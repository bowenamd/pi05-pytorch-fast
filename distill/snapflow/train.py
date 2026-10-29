# Derived from FastCrest Tether. Copyright (c) 2026 FastCrest.
# SPDX-License-Identifier: BUSL-1.1
# See LICENSE and NOTICE.md in this directory.
"""Run standalone default pi0.5 SnapFlow on real local LeRobot data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .loss import snapflow_loss_step
from .model import enable_snapflow
from .runtime import (build_dataloader, build_preprocessor, iter_batches, load_policy,
                      prepare_batch, save_checkpoint, velocity_fn)

LOG = logging.getLogger(__name__)


def summarize(rows: list[dict]) -> dict:
    """Compare nonoverlapping early/late windows; noisy steps need not be monotonic."""
    if len(rows) < 2:
        return {"loss_decreased": None, "reason": "At least two steps required"}
    window = min(50, len(rows) // 2)
    early = sum(r["total"] for r in rows[:window]) / window
    late = sum(r["total"] for r in rows[-window:]) / window
    return {"window_steps": window, "early_mean_loss": early, "late_mean_loss": late,
            "loss_drop_pct": 100 * (early - late) / early if early else None,
            "loss_decreased": late < early,
            "all_losses_finite": all(math.isfinite(r[k]) for r in rows
                                     for k in ("total", "flow_matching", "consistency"))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True, help="Local undistilled PI05 checkpoint")
    p.add_argument("--tokenizer", type=Path, help="Override teacher tokenizer with a local directory")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-repo", default="lerobot/libero")
    p.add_argument("--dataset-revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    p.add_argument("--output", type=Path, required=True, help="New or empty run directory")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--consistency-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--gradient-check-every", type=int, default=10)
    p.add_argument("--require-loss-decrease", action="store_true",
                   help="Fail the short-run validation if late-window loss does not decrease")
    return p


def main():
    args = parser().parse_args()
    if min(args.steps, args.batch_size, args.checkpoint_every, args.gradient_check_every) <= 0:
        raise ValueError("Steps, batch size, and check/save intervals must be positive")
    if args.num_workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid workers or learning rate")
    if not math.isfinite(args.consistency_alpha) or args.consistency_alpha < 0:
        raise ValueError("consistency_alpha must be finite and nonnegative")
    args.teacher = args.teacher.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A ROCm PyTorch build and available AMD GPU are required")
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "torchvision", "lerobot", "transformers", "av", "numpy", "datasets")}
    if versions["lerobot"] != "0.6.1" or versions["transformers"] != "5.5.4":
        raise RuntimeError("Use the isolated distill/snapflow/setup_conda.sh environment")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    manifest = {
        "method": "snapflow", "variant": "pi05-default-expert-only",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "versions": versions, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "extraction_source": json.loads((source / "source_manifest.json").read_text()),
        "component_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source.glob("*.py")},
        "teacher_config_sha256": hashlib.sha256((args.teacher / "config.json").read_bytes()).hexdigest(),
        "dataset_info_sha256": hashlib.sha256((args.dataset_root / "meta/info.json").read_bytes()).hexdigest(),
        "optimizer": {"type": "AdamW", "lr": args.learning_rate,
                      "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "exact_resume_supported": False,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    completed_steps = 0
    result = {"status": "failed", "completed_steps": 0}
    started = time.perf_counter()
    try:
        LOG.info("Loading frozen teacher and fresh student from %s", args.teacher)
        teacher = load_policy(args.teacher).eval()
        teacher.requires_grad_(False)
        student = load_policy(args.teacher).train()
        student.requires_grad_(True)
        enable_snapflow(teacher.model)
        teacher.requires_grad_(False)
        enable_snapflow(student.model)
        student.model.paligemma_with_expert.paligemma.requires_grad_(False)
        parameters = [p for p in student.parameters() if p.requires_grad]
        LOG.info("Trainable parameters: %d / %d", sum(p.numel() for p in parameters),
                 sum(p.numel() for p in student.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        preprocessor = build_preprocessor(args.teacher, tokenizer=args.tokenizer)
        loader = build_dataloader(
            args.dataset_root, repo_id=args.dataset_repo, revision=args.dataset_revision,
            chunk_size=teacher.config.chunk_size, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed,
        )
        teacher_velocity = velocity_fn(teacher)
        student_velocity = velocity_fn(student)
        training_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with (args.output / "training_log.jsonl").open("w", buffering=1) as log:
            for step, batch in iter_batches(loader, args.steps):
                step_started = time.perf_counter()
                action, noise, t, obs = prepare_batch(
                    preprocessor(batch), chunk_size=teacher.config.chunk_size,
                    max_action_dim=teacher.config.max_action_dim,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, snapshot = snapflow_loss_step(
                    student_velocity, teacher_velocity, action=action, noise=noise, t=t,
                    obs_kwargs=obs, consistency_alpha=args.consistency_alpha,
                )
                if not all(math.isfinite(v) for v in asdict(snapshot).values()):
                    raise RuntimeError(f"Non-finite loss at step {step}: {snapshot}")
                loss.backward()
                grad_norm = None
                if step == 1 or step % args.gradient_check_every == 0:
                    grads = [p.grad for p in parameters if p.grad is not None]
                    if not grads:
                        raise RuntimeError("Student has no gradients")
                    grad_norm = float(torch.stack([g.float().norm() for g in grads]).norm().item())
                    if not math.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite gradients at step {step}")
                optimizer.step()
                torch.cuda.synchronize()
                completed_steps = step
                row = {"step": step, **asdict(snapshot), "gradient_norm": grad_norm,
                       "step_time_s": time.perf_counter() - step_started,
                       "training_elapsed_s": time.perf_counter() - training_started,
                       "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                if step == 1 or step % 10 == 0:
                    LOG.info("step %d/%d loss=%.4f fm=%.4f consistency=%.4f time=%.2fs",
                             step, args.steps, snapshot.total, snapshot.flow_matching,
                             snapshot.consistency, row["step_time_s"])
                if step % args.checkpoint_every == 0 or step == args.steps:
                    checkpoint = save_checkpoint(student, preprocessor, args.teacher,
                                                 args.output, step, manifest)
                    LOG.info("Saved %s", checkpoint)
        result = {"status": "ok", "completed_steps": completed_steps,
                  "checkpoint": str(checkpoint), **summarize(rows),
                  "training_time_s": time.perf_counter() - training_started,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        if args.require_loss_decrease and result["loss_decreased"] is not True:
            result["status"] = "loss_validation_failed"
            raise RuntimeError("Late-window mean loss did not decrease; see result.json")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      completed_steps=completed_steps)
        raise
    finally:
        result["wall_time_s"] = time.perf_counter() - started
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
