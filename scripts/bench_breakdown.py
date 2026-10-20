#!/usr/bin/env python3
"""1-NFE latency breakdown: Vision encoder / LLM prefill / Denoiser (1 step).

Each phase is compiled on its own (same Inductor mode as the E2E path) and timed
with CUDA events; the fullgraph E2E median is printed alongside for reference.
Phase compile splits the fullgraph, so the phase sum is slightly above E2E.

  Vision   = embed_prefix: batched SigLIP + projector (+ lang embedding lookup, cat)
  LLM      = prefix masks + PaliGemma prefix LM prefill (W4A4), builds KV cache
  Denoiser = denoise_step (KV clone + action expert + action proj), 1 step

Env: same as bench_latency.py, plus PI05_BREAKDOWN_E2E=0 to skip the E2E run.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import torch

from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint
from pi05_fast.rocm_pi05_optim import _lang_token_budget, install_rocm_pi05_eval_hooks
from pi05_fast.snapflow_pi05 import checkpoint_has_snapflow_mlp, install_pi05_snapflow_eval_hooks


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return statistics.median(xs), xs[max(0, n // 10)], xs[min(n - 1, 9 * n // 10)]


def main() -> None:
    ckpt = Path(os.environ.get("POLICY_ID", Path.home() / "model_data/pi05_snapflow_1nfe")).expanduser()
    load = Path(os.environ.get("POLICY_LOAD", ckpt)).expanduser()
    n = int(os.environ.get("PI05_BENCH_ITERS", "30"))
    w = int(os.environ.get("PI05_BENCH_WARMUP", "8"))
    do_e2e = os.environ.get("PI05_BREAKDOWN_E2E", "1") != "0"
    compile_on = os.environ.get("PI05_COMPILE", "1") != "0"
    mode = os.environ.get("PI05_COMPILE_MODE", "max-autotune-no-cudagraphs")
    teacher = Path(os.environ.get("SNAPFLOW_TEACHER_POLICY", Path.home() / "model_data/pi05_libero_v044"))
    for note in sanitize_snapflow_checkpoint(load, teacher):
        print(f"policy checkpoint: {note}", flush=True)

    if checkpoint_has_snapflow_mlp(ckpt) or checkpoint_has_snapflow_mlp(load):
        install_pi05_snapflow_eval_hooks()
    install_rocm_pi05_eval_hooks()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks

    print("loading", load, flush=True)
    policy = PI05Policy.from_pretrained(str(load))
    model = policy.model
    device = next(model.parameters()).device
    cs, ad = model.config.chunk_size, model.config.max_action_dim

    images = [torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32) for _ in range(3)]
    img_masks = [torch.ones(1, device=device, dtype=torch.bool) for _ in range(3)]
    tokens = torch.randint(1, 1000, (1, 200), device=device)
    lang_len = max(1, min(int(os.environ.get("PI05_BENCH_LANG_LEN", "52")), 200))
    masks = torch.zeros(1, 200, device=device, dtype=torch.bool)
    masks[:, :lang_len] = True
    noise = model.sample_noise((1, cs, ad), device)

    # Phase path bypasses the sample_actions wrapper, so apply the lang trim here.
    budget = _lang_token_budget()
    p_tokens, p_masks = tokens, masks
    if 0 < budget < tokens.shape[1] and not bool(masks[:, budget:].any()):
        p_tokens, p_masks = tokens[:, :budget].contiguous(), masks[:, :budget].contiguous()

    def vision(images, img_masks, tokens, masks):
        return model.embed_prefix(images, img_masks, tokens, masks)

    def llm(prefix_embs, prefix_pad_masks, prefix_att_masks):
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = model._prepare_attention_masks_4d(att_2d)
        _, pkv = model.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=pos, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True,
        )
        return pkv

    def denoiser(prefix_pad_masks, pkv, noise):
        t = torch.ones(noise.shape[0], dtype=torch.float32, device=noise.device)
        x_t = noise.to(model.action_in_proj.weight.dtype)
        v_t = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks, past_key_values=pkv,
            x_t=x_t, timestep=t, target_time=t,
        )
        return (x_t - v_t).to(noise.dtype)

    if compile_on:
        fullgraph = os.environ.get("PI05_COMPILE_FULLGRAPH", "1") != "0"
        vision = torch.compile(vision, mode=mode, fullgraph=fullgraph)
        llm = torch.compile(llm, mode=mode, fullgraph=fullgraph)
        denoiser = torch.compile(denoiser, mode=mode, fullgraph=fullgraph)
        print(f"breakdown: per-phase torch.compile({mode}, fullgraph={fullgraph})", flush=True)
    else:
        print("breakdown: eager phases (PI05_COMPILE=0)", flush=True)

    names = ["vision", "llm", "denoiser"]

    def run_phases(evs=None):
        with torch.no_grad():
            if evs:
                evs[0].record()
            pe, ppm, pam = vision(images, img_masks, p_tokens, p_masks)
            if evs:
                evs[1].record()
            pkv = llm(pe, ppm, pam)
            if evs:
                evs[2].record()
            out = denoiser(ppm, pkv, noise)
            if evs:
                evs[3].record()
        return out, ppm.shape[1]

    print("phase warmup / compile...", flush=True)
    for _ in range(w):
        _, n_prefix = run_phases()
    torch.cuda.synchronize()
    print(f"prefix tokens = {n_prefix}", flush=True)

    acc = {k: [] for k in names + ["sum_wall"]}
    for _ in range(n):
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_phases(evs)
        torch.cuda.synchronize()
        acc["sum_wall"].append((time.perf_counter() - t0) * 1e3)
        for i, k in enumerate(names):
            acc[k].append(evs[i].elapsed_time(evs[i + 1]))

    e2e = None
    if do_e2e:
        def step():
            with torch.no_grad():
                return model.sample_actions(images, img_masks, tokens, masks, noise=noise)

        print("E2E warmup / compile...", flush=True)
        for _ in range(w):
            step()
        torch.cuda.synchronize()
        e2e = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1e3)

    tot = statistics.median(acc["sum_wall"])
    labels = {"vision": "Vision Encoder", "llm": "LLM (prefix prefill)", "denoiser": "Denoiser (1 step)",
              "sum_wall": "Phases total (wall)"}
    print(f"\n=== latency breakdown (n={n}, ms) ===", flush=True)
    print(f"{'phase':24s} {'median':>8s} {'p10':>8s} {'p90':>8s} {'share':>7s}", flush=True)
    for k in names + ["sum_wall"]:
        m, p10, p90 = _stats(acc[k])
        print(f"{labels[k]:24s} {m:8.2f} {p10:8.2f} {p90:8.2f} {100 * m / tot:6.1f}%", flush=True)
    if e2e:
        m, p10, p90 = _stats(e2e)
        print(f"{'E2E fullgraph (ref)':24s} {m:8.2f} {p10:8.2f} {p90:8.2f}", flush=True)


if __name__ == "__main__":
    main()
