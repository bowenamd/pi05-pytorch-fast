#!/usr/bin/env python3
"""Cosine similarity of predicted action chunks vs. a baseline (same inputs + same noise).

  source env.sh
  # 1) record real LIBERO model inputs once (fixed noise is generated per sample)
  python scripts/compare_cosine.py capture --out eval_out/cosine/inputs.pt
  # 2) one process per config; config is chosen by the usual PI05_* env knobs
  python scripts/compare_cosine.py dump --upstream --dtype float32 --out eval_out/cosine/ref_fp32.pt
  PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 python scripts/compare_cosine.py dump --out eval_out/cosine/fast.pt
  # 3) compare
  python scripts/compare_cosine.py compare --ref eval_out/cosine/ref_fp32.pt eval_out/cosine/*.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

DEFAULT_POLICY = Path.home() / "model_data" / "pi05_snapflow_1nfe"
DEFAULT_TEACHER = Path.home() / "model_data" / "pi05_libero_v044"


def _prepare_checkpoint(policy_id: str) -> Path:
    from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint
    from pi05_fast.snapflow_pi05 import checkpoint_has_snapflow_mlp, install_pi05_snapflow_eval_hooks

    ckpt = Path(policy_id).expanduser()
    teacher = Path(os.environ.get("SNAPFLOW_TEACHER_POLICY", DEFAULT_TEACHER))
    for note in sanitize_snapflow_checkpoint(ckpt, teacher):
        print(f"policy checkpoint: {note}", flush=True)
    if checkpoint_has_snapflow_mlp(ckpt):
        install_pi05_snapflow_eval_hooks()
    return ckpt


def cmd_capture(a) -> None:
    sys.path.insert(0, str(PKG_ROOT / "scripts"))
    import libero_eval as le  # sets MuJoCo/LIBERO env + sys.path
    import torch

    le._check_deps()
    ckpt = _prepare_checkpoint(a.policy_id)
    from pi05_fast.rocm_pi05_optim import install_rocm_pi05_eval_hooks

    install_rocm_pi05_eval_hooks()
    le._patch_paligemma_tokenizer(Path(os.environ.get("PALIGEMMA_TOKENIZER_PATH", a.tokenizer_path)))

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    rec: dict[str, list] = {"images": [], "img_masks": [], "tokens": [], "masks": []}
    state = {"calls": 0, "cfg": None}

    def _record(model):
        inner = model.sample_actions
        state["cfg"] = model.config

        def recording(images, img_masks, tokens, masks, *args, **kwargs):
            i = state["calls"]
            state["calls"] += 1
            if i % a.stride == 0 and len(rec["tokens"]) < a.max_samples:
                rec["images"].append(torch.stack([im.detach().to("cpu", torch.float16) for im in images]))
                rec["img_masks"].append(torch.stack([m.detach().cpu() for m in img_masks]))
                rec["tokens"].append(tokens.detach().cpu())
                rec["masks"].append(masks.detach().cpu())
            return inner(images, img_masks, tokens, masks, *args, **kwargs)

        model.sample_actions = recording

    hooked = PI05Policy.from_pretrained.__func__

    @classmethod
    def from_pretrained_recording(cls, *args, **kwargs):
        policy = hooked(cls, *args, **kwargs)
        _record(policy.model)
        return policy

    PI05Policy.from_pretrained = from_pretrained_recording

    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    eval_args = argparse.Namespace(
        output_dir=out.parent / "capture_rollout",
        n_action_steps=10,
        tasks=a.tasks,
        task_ids=a.task_ids,
        episodes=a.episodes,
        n_inference_steps=1,
        seed=a.seed,
    )
    info, wall_s = le._run_eval(eval_args, str(ckpt))
    le._print_summary(info, wall_s)

    cfg = state["cfg"]
    n = len(rec["tokens"])
    gen = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(n, 1, cfg.chunk_size, cfg.max_action_dim, generator=gen, dtype=torch.float32)
    torch.save(
        {
            "images": torch.stack(rec["images"]),
            "img_masks": torch.stack(rec["img_masks"]),
            "tokens": torch.stack(rec["tokens"]),
            "masks": torch.stack(rec["masks"]),
            "noise": noise,
            "action_dim": int(cfg.output_features["action"].shape[0]),
            "meta": {"tasks": a.tasks, "episodes": a.episodes, "seed": a.seed, "stride": a.stride,
                     "calls_seen": state["calls"]},
        },
        out,
    )
    print(f"captured {n} samples (of {state['calls']} calls) -> {out}", flush=True)


def cmd_dump(a) -> None:
    import torch

    ckpt = _prepare_checkpoint(a.policy_id)
    if not a.upstream:
        from pi05_fast.rocm_pi05_optim import install_rocm_pi05_eval_hooks

        install_rocm_pi05_eval_hooks()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(str(ckpt))
    model = policy.model
    if a.upstream:
        policy.to("cuda")
        if a.dtype:
            model.to(getattr(torch, a.dtype))
        model.eval()
    dev = next(model.parameters()).device
    # Mirrors the patched predict_action_chunk: bf16 autocast unless the prefix LM runs fp16 (W4A4).
    autocast = not a.upstream and getattr(model, "_prefix_lm_dtype", None) is None

    data = torch.load(a.inputs)
    n = data["tokens"].shape[0] if a.limit <= 0 else min(a.limit, data["tokens"].shape[0])
    ncam = data["images"].shape[1]
    outs = []
    t0 = time.perf_counter()
    for i in range(n):
        images = [data["images"][i, c].to(dev, torch.float32) for c in range(ncam)]
        img_masks = [data["img_masks"][i, c].to(dev) for c in range(ncam)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            act = model.sample_actions(
                images,
                img_masks,
                data["tokens"][i].to(dev),
                data["masks"][i].to(dev),
                noise=data["noise"][i].to(dev),
            )
        outs.append(act.float().cpu())
        if i == 0:
            print(f"first sample done ({time.perf_counter() - t0:.1f}s)", flush=True)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith("PI05_")}
    torch.save(
        {
            "actions": torch.stack(outs),
            "action_dim": data["action_dim"],
            "label": a.label or out.stem,
            "upstream": a.upstream,
            "dtype": a.dtype,
            "env": knobs,
        },
        out,
    )
    print(f"dumped {n} action chunks -> {out}", flush=True)


def cmd_compare(a) -> None:
    import torch
    import torch.nn.functional as F

    ref = torch.load(a.ref)
    d = ref["action_dim"]
    r_all = ref["actions"].double()
    rows = []
    for p in a.runs:
        if Path(p).resolve() == Path(a.ref).resolve():
            continue
        run = torch.load(p)
        x_all = run["actions"].double()
        n = min(len(x_all), len(r_all))
        x = x_all[:n, ..., :d].reshape(n, -1)
        r = r_all[:n, ..., :d].reshape(n, -1)
        cos = F.cosine_similarity(x, r, dim=1)
        rel = (x - r).norm(dim=1) / r.norm(dim=1)
        # Per-timestep cosine of each single 7-D action.
        cos_step = F.cosine_similarity(x_all[:n, 0, :, :d], r_all[:n, 0, :, :d], dim=-1)
        rows.append(
            {
                "run": run.get("label", Path(p).stem),
                "n": n,
                "cos_mean": cos.mean().item(),
                "cos_min": cos.min().item(),
                "cos_p5": cos.quantile(0.05).item(),
                "step_cos_mean": cos_step.mean().item(),
                "step_cos_p1": cos_step.flatten().quantile(0.01).item(),
                "rel_l2_mean": rel.mean().item(),
                "mse": ((x - r) ** 2).mean().item(),
                "max_abs": (x - r).abs().max().item(),
            }
        )
    label = ref.get("label", Path(a.ref).stem)
    print(f"\nreference: {label}   (normalized actions, first {d} dims x {r_all.shape[-2]} steps)")
    hdr = f"{'run':<24}{'n':>5}{'cos mean':>10}{'cos min':>10}{'cos p5':>10}" \
          f"{'step mean':>11}{'step p1':>9}{'relL2':>9}{'MSE':>11}{'max|d|':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:<24}{r['n']:>5}{r['cos_mean']:>10.5f}{r['cos_min']:>10.5f}{r['cos_p5']:>10.5f}"
            f"{r['step_cos_mean']:>11.5f}{r['step_cos_p1']:>9.4f}{r['rel_l2_mean']:>9.4f}"
            f"{r['mse']:>11.2e}{r['max_abs']:>9.4f}"
        )
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps({"reference": label, "rows": rows}, indent=2) + "\n")
        print(f"Wrote {a.json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="record model inputs from LIBERO rollouts")
    c.add_argument("--out", required=True)
    c.add_argument("--policy-id", default=str(DEFAULT_POLICY))
    c.add_argument("--tasks", default="libero_10")
    c.add_argument("--task-ids", default=None)
    c.add_argument("--episodes", type=int, default=1)
    c.add_argument("--stride", type=int, default=3, help="record every Nth inference call")
    c.add_argument("--max-samples", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tokenizer-path", default=str(Path.home() / "model_data/paligemma2-3b-pt-224"))

    d = sub.add_parser("dump", help="run one config on captured inputs and save actions")
    d.add_argument("--inputs", default="eval_out/cosine/inputs.pt")
    d.add_argument("--out", required=True)
    d.add_argument("--label", default=None)
    d.add_argument("--policy-id", default=str(DEFAULT_POLICY))
    d.add_argument("--upstream", action="store_true", help="stock lerobot PI05 (no ROCm optimizations)")
    d.add_argument("--dtype", default=None, choices=["float32", "bfloat16"], help="cast model (upstream only)")
    d.add_argument("--limit", type=int, default=0)

    m = sub.add_parser("compare", help="cosine similarity of runs vs. a reference dump")
    m.add_argument("--ref", required=True)
    m.add_argument("--json", default=None)
    m.add_argument("runs", nargs="+")

    a = p.parse_args()
    {"capture": cmd_capture, "dump": cmd_dump, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
