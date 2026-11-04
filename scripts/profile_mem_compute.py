#!/usr/bin/env python3
"""Per-kernel memory traffic vs. compute during pi05 1-NFE inference (rocprofv3, gfx1151).

  collect  run the pi05 path under rocprofv3: 1 timing pass + 4 counter passes
           (+ optional thread-trace pass). Only steady-state iterations are profiled
           (roctxProfilerResume/Pause around them, --selected-regions).
  report   join the passes per kernel dispatch -> summary.md, kernels.csv,
           phases.csv, timeline.png (+ att.csv with --att data).
  target   the profiled workload (spawned by collect; not meant to be run directly).

Kernels are attributed to phases (bw_probe / vision / llm / denoiser) with roctx
ranges; each phase is compiled separately, as in bench_breakdown.py.

Traffic = GL2C->EA requests, i.e. L2 misses. On gfx1151 they are served by the
32 MB MALL (Infinity Cache) or DRAM; no per-kernel counter separates the two.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
PHASES = ["vision", "llm", "denoiser"]
PROBE = "bw_probe"
# One counter group per pass: each set fits the gfx1151 counter hardware in one pass.
COUNTER_PASSES = {
    "sq": ["GRBM_GUI_ACTIVE", "SQ_WAVE_CYCLES", "SQ_WAIT_ANY", "SQ_INSTS_VALU",
           "SQ_INSTS_LDS", "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQ_WAVES"],
    "rd": ["GL2C_EA_RDREQ_32B_sum", "GL2C_EA_RDREQ_64B_sum",
           "GL2C_EA_RDREQ_96B_sum", "GL2C_EA_RDREQ_128B_sum"],
    "wr": ["GL2C_MC_WRREQ_sum", "GL2C_EA_WRREQ_64B_sum", "GL2C_HIT_sum", "GL2C_MISS_sum"],
    "mu": ["MemUnitBusy"],
}
SPEC_BW_GBS = 256.0  # LPDDR5X-8000 x 256-bit


# ----------------------------------------------------------------------------- target
class _Roctx:
    """roctx via ctypes; every call is a no-op when the library is missing."""

    def __init__(self) -> None:
        self.lib = None
        cands = []
        try:
            import _rocm_sdk_core

            cands.append(str(Path(_rocm_sdk_core.__file__).parent / "lib/librocprofiler-sdk-roctx.so.1"))
        except ImportError:
            pass
        cands.append("librocprofiler-sdk-roctx.so.1")
        for c in cands:
            try:
                self.lib = ctypes.CDLL(c)
                self.lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
                break
            except OSError:
                continue

    def push(self, name: str) -> None:
        if self.lib:
            self.lib.roctxRangePushA(name.encode())

    def pop(self) -> None:
        if self.lib:
            self.lib.roctxRangePop()

    def resume(self) -> None:
        if self.lib:
            self.lib.roctxProfilerResume(0)

    def pause(self) -> None:
        if self.lib:
            self.lib.roctxProfilerPause(0)


def target() -> None:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    sys.path.insert(0, str(PKG_ROOT))
    import torch

    from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint
    from pi05_fast.rocm_pi05_optim import _lang_token_budget, install_rocm_pi05_eval_hooks
    from pi05_fast.snapflow_pi05 import checkpoint_has_snapflow_mlp, install_pi05_snapflow_eval_hooks

    rx = _Roctx()
    ckpt = Path(os.environ.get("POLICY_ID", Path.home() / "model_data/pi05_snapflow_1nfe")).expanduser()
    load = Path(os.environ.get("POLICY_LOAD", ckpt)).expanduser()
    teacher = Path(os.environ.get("SNAPFLOW_TEACHER_POLICY", Path.home() / "model_data/pi05_libero_v044"))
    iters = int(os.environ.get("PI05_MC_ITERS", "3"))
    warmup = int(os.environ.get("PI05_BENCH_WARMUP", "4"))
    probe_mb = int(os.environ.get("PI05_MC_PROBE_MB", "1024"))
    mode = os.environ.get("PI05_COMPILE_MODE", "max-autotune-no-cudagraphs")
    for note in sanitize_snapflow_checkpoint(load, teacher):
        print(f"policy checkpoint: {note}", flush=True)
    if checkpoint_has_snapflow_mlp(ckpt) or checkpoint_has_snapflow_mlp(load):
        install_pi05_snapflow_eval_hooks()
    install_rocm_pi05_eval_hooks()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks

    print("loading", load, flush=True)
    model = PI05Policy.from_pretrained(str(load)).model
    device = next(model.parameters()).device
    cs, ad = model.config.chunk_size, model.config.max_action_dim
    images = [torch.randn(1, 3, 224, 224, device=device) for _ in range(3)]
    img_masks = [torch.ones(1, device=device, dtype=torch.bool) for _ in range(3)]
    tokens = torch.randint(1, 1000, (1, 200), device=device)
    lang_len = max(1, min(int(os.environ.get("PI05_BENCH_LANG_LEN", "52")), 200))
    masks = torch.zeros(1, 200, device=device, dtype=torch.bool)
    masks[:, :lang_len] = True
    noise = model.sample_noise((1, cs, ad), device)
    budget = _lang_token_budget()
    if 0 < budget < tokens.shape[1] and not bool(masks[:, budget:].any()):
        tokens, masks = tokens[:, :budget].contiguous(), masks[:, :budget].contiguous()

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
        v_t = model.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=pkv,
                                 x_t=x_t, timestep=t, target_time=t)
        return (x_t - v_t).to(noise.dtype)

    if os.environ.get("PI05_COMPILE", "1") != "0":
        vision, llm, denoiser = (torch.compile(f, mode=mode, fullgraph=True) for f in (vision, llm, denoiser))

    def run(marked: bool):
        def phase(name, fn, *a):
            if marked:
                rx.push(name)
            out = fn(*a)
            if marked:
                # sync inside the range so every kernel of the phase lies within it
                torch.cuda.synchronize()
                rx.pop()
            return out

        with torch.no_grad():
            pe, ppm, pam = phase("vision", vision, images, img_masks, tokens, masks)
            pkv = phase("llm", llm, pe, ppm, pam)
            return phase("denoiser", denoiser, ppm, pkv, noise), ppm.shape[1]

    src = torch.empty(probe_mb << 20, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    print("warmup / compile...", flush=True)
    for _ in range(warmup):
        _, n_prefix = run(False)
        dst.copy_(src)
    torch.cuda.synchronize()
    print(f"prefix tokens = {n_prefix}; profiled iterations = {iters}", flush=True)
    rx.resume()
    rx.push(PROBE)
    dst.copy_(src)
    torch.cuda.synchronize()
    rx.pop()
    for i in range(iters):
        rx.push(f"iter{i}")
        run(True)
        rx.pop()
    rx.pause()
    print("target done", flush=True)


# ---------------------------------------------------------------------------- collect
def _rocprof() -> str:
    # The venv rocprofv3 matches the venv HIP runtime; /opt/rocm/bin/rocprofv3 (first in
    # PATH after env.sh) loads the host HSA runtime and fails to start torch.
    p = Path(sys.executable).parent / "rocprofv3"
    return str(p) if p.exists() else (shutil.which("rocprofv3") or "rocprofv3")


def collect(args) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    passes = [p for p in args.passes.split(",") if p]
    if args.att_lib:
        passes.append("att")
    base_env = dict(os.environ, PI05_MC_ITERS=str(args.iters), PI05_BENCH_WARMUP=str(args.warmup))
    for p in passes:
        d = out / p
        shutil.rmtree(d, ignore_errors=True)
        # ATT ignores --selected-regions beyond the first matching dispatch, so it selects
        # dispatches by occurrence index instead (warmup occurrences come first).
        cmd = [_rocprof(), "-d", str(d), "-o", "r"] + ([] if p == "att" else ["--selected-regions"])
        env = base_env
        if p == "trace":
            cmd += ["--kernel-trace", "--marker-trace", "--memory-copy-trace",
                    "--output-format", "csv", "pftrace"]
        elif p == "att":
            a, b = (int(x) for x in args.att_index.split("-"))
            skip = args.warmup * args.att_per_iter
            cmd += ["--att", "--att-library-path", args.att_lib,
                    "--kernel-include-regex", args.att_regex,
                    "--kernel-iteration-range", f"[{skip + a}-{skip + b}]", "--output-format", "csv"]
            env = dict(base_env, PI05_MC_ITERS="1")
        else:
            cmd += ["--pmc", *COUNTER_PASSES[p], "--output-format", "csv"]
        cmd += ["--", sys.executable, str(Path(__file__).resolve()), "target"]
        log = out / f"{p}.log"
        print(f"[{p}] {' '.join(cmd[:-3])} ... (log: {log})", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            sys.exit(f"[{p}] rocprofv3 exited with {rc}; see {log}")
    print(f"done -> python {Path(__file__).name} report --out {out}", flush=True)


# ----------------------------------------------------------------------------- report
def _short(name: str, n: int = 64) -> str:
    s = re.sub(r"\(.*", "", str(name)).replace("void ", "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _counter_pass(d: Path):
    import pandas as pd

    df = pd.read_csv(d / "r_counter_collection.csv")
    g = df.groupby(["Dispatch_Id", "Kernel_Name", "Counter_Name"])["Counter_Value"]
    # derived percentages are averaged over duplicates, raw counts are summed
    v = g.sum().where(~g.sum().index.get_level_values(2).isin(["MemUnitBusy"]), g.mean())
    w = v.unstack("Counter_Name").reset_index().sort_values("Dispatch_Id")
    w["occ"] = w.groupby("Kernel_Name").cumcount()
    return w.drop(columns="Dispatch_Id")


def _assign_ranges(ts, ranges):
    """Name of the (non-overlapping) range containing each timestamp, else ''."""
    import numpy as np

    ranges = ranges.sort_values("Start_Timestamp")
    st, en = ranges["Start_Timestamp"].to_numpy(), ranges["End_Timestamp"].to_numpy()
    nm = ranges["Function"].to_numpy()
    i = np.searchsorted(st, ts, side="right") - 1
    ok = (i >= 0) & (ts <= en[np.clip(i, 0, None)])
    return np.where(ok, nm[np.clip(i, 0, None)], "")


def _load_dispatches(out: Path):
    import pandas as pd

    kt = pd.read_csv(out / "trace/r_kernel_trace.csv").sort_values("Dispatch_Id")
    kt["occ"] = kt.groupby("Kernel_Name").cumcount()
    mk = pd.read_csv(out / "trace/r_marker_api_trace.csv")
    mk = mk[mk["Domain"] == "MARKER_CORE_RANGE_API"]
    ts = kt["Start_Timestamp"].to_numpy()
    kt["phase"] = _assign_ranges(ts, mk[mk["Function"].isin(PHASES + [PROBE])])
    kt["iter"] = _assign_ranges(ts, mk[mk["Function"].str.match(r"iter\d+$")])
    kt.loc[kt["phase"] == PROBE, "iter"] = PROBE
    df = kt
    for p in COUNTER_PASSES:
        if (out / p / "r_counter_collection.csv").exists():
            df = df.merge(_counter_pass(out / p), on=["Kernel_Name", "occ"], how="left")
        else:
            print(f"warning: pass {p} missing", file=sys.stderr)
    for c in sum(COUNTER_PASSES.values(), []):
        if c not in df:
            df[c] = float("nan")
    agent = pd.read_csv(out / "trace/r_agent_info.csv")
    simds = int(agent.loc[agent["Agent_Type"] == "GPU", "Simd_Count"].iloc[0])

    df["dur_us"] = (df["End_Timestamp"] - df["Start_Timestamp"]) / 1e3
    df["rd_MB"] = (32 * df["GL2C_EA_RDREQ_32B_sum"] + 64 * df["GL2C_EA_RDREQ_64B_sum"]
                   + 96 * df["GL2C_EA_RDREQ_96B_sum"] + 128 * df["GL2C_EA_RDREQ_128B_sum"]) / 1e6
    wr64 = df["GL2C_EA_WRREQ_64B_sum"]
    df["wr_MB"] = (64 * wr64 + 32 * (df["GL2C_MC_WRREQ_sum"] - wr64)) / 1e6
    df["fabric_GBs"] = (df["rd_MB"] + df["wr_MB"]) * 1e3 / df["dur_us"]
    df["l2_hit_pct"] = 100 * df["GL2C_HIT_sum"] / (df["GL2C_HIT_sum"] + df["GL2C_MISS_sum"])
    df["mem_unit_busy_pct"] = df["MemUnitBusy"]
    df["valu_issue_pct"] = 100 * df["SQ_INSTS_VALU"] / (simds * df["GRBM_GUI_ACTIVE"])
    df["wave_wait_pct"] = 100 * df["SQ_WAIT_ANY"] / df["SQ_WAVE_CYCLES"]
    df["short"] = df["Kernel_Name"].map(_short)
    return df


def _wavg(x, w):
    m = x.notna() & w.notna()
    return float((x[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float("nan")


def _classify(r, peak):
    if r["fabric_GBs"] >= 0.7 * peak:
        return "L2-miss BW bound"
    if r["mem_unit_busy_pct"] >= 70:
        return "mem-pipe (L2/LDS)"
    return "compute/latency"


def report(args) -> None:
    import pandas as pd

    out = Path(args.out).resolve()
    df = _load_dispatches(out)
    probe = df[df["phase"] == PROBE]
    peak = float(probe["fabric_GBs"].max()) if len(probe) else SPEC_BW_GBS
    run = df[df["iter"].str.startswith("iter") & df["phase"].isin(PHASES)].copy()
    iters = run["iter"].nunique()
    run["fabric_util_pct"] = 100 * run["fabric_GBs"] / peak
    run.to_csv(out / "kernels.csv", index=False)

    rows = []
    for ph, g in list(run.groupby("phase", sort=False)) + [("TOTAL", run)]:
        t = g["dur_us"].sum() / iters
        mb = (g["rd_MB"].sum() + g["wr_MB"].sum()) / iters
        rows.append({
            "phase": ph, "kernels/iter": len(g) / iters, "gpu_ms": t / 1e3,
            "read_MB": g["rd_MB"].sum() / iters, "write_MB": g["wr_MB"].sum() / iters,
            "avg_GBs": mb * 1e3 / t, "fabric_util_pct": 100 * mb * 1e3 / t / peak,
            "mem_unit_busy_pct": _wavg(g["mem_unit_busy_pct"], g["dur_us"]),
            "valu_issue_pct": _wavg(g["valu_issue_pct"], g["dur_us"]),
            "wave_wait_pct": _wavg(g["wave_wait_pct"], g["dur_us"]),
        })
    phases = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(PHASES + ["TOTAL"])}
    phases = phases.sort_values("phase", key=lambda s: s.map(order))
    phases.to_csv(out / "phases.csv", index=False)

    total_us = run["dur_us"].sum()
    krows = []
    for (ph, nm), g in run.groupby(["phase", "short"]):
        krows.append({
            "phase": ph, "kernel": nm, "calls/iter": len(g) / iters,
            "ms/iter": g["dur_us"].sum() / iters / 1e3, "share_pct": 100 * g["dur_us"].sum() / total_us,
            "avg_us": g["dur_us"].mean(), "MB/iter": (g["rd_MB"].sum() + g["wr_MB"].sum()) / iters,
            "GBs": (g["rd_MB"].sum() + g["wr_MB"].sum()) * 1e3 / g["dur_us"].sum(),
            "l2_hit_pct": _wavg(g["l2_hit_pct"], g["dur_us"]),
            "mem_unit_busy_pct": _wavg(g["mem_unit_busy_pct"], g["dur_us"]),
            "valu_issue_pct": _wavg(g["valu_issue_pct"], g["dur_us"]),
            "wave_wait_pct": _wavg(g["wave_wait_pct"], g["dur_us"]),
        })
    kt = pd.DataFrame(krows).sort_values("ms/iter", ascending=False)
    kt["fabric_util_pct"] = 100 * kt["GBs"] / peak
    kt["class"] = kt.apply(lambda r: _classify({"fabric_GBs": r["GBs"], "mem_unit_busy_pct": r["mem_unit_busy_pct"]}, peak), axis=1)
    kt.to_csv(out / "kernel_summary.csv", index=False)

    att = _att_report(out) if (out / "att").exists() else None
    _plot_timeline(run, peak, out / "timeline.png")

    def fmt(d):
        cell = lambda v: (f"{v:.1f}" if abs(v) >= 1 else f"{v:.3f}") if isinstance(v, float) else str(v)  # noqa: E731
        rows = [list(d.columns)] + [[cell(v) for v in r] for r in d.itertuples(index=False)]
        return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                         + ["| " + " | ".join(r) + " |" for r in rows[1:]])

    lines = [
        "# pi05 per-kernel memory traffic vs compute",
        "",
        f"iterations: {iters}; GPU kernel time/iter: {total_us / iters / 1e3:.1f} ms; "
        f"measured copy bandwidth (bw_probe): {peak:.0f} GB/s (spec {SPEC_BW_GBS:.0f} GB/s)",
        "",
        "## Phases (per iteration)",
        fmt(phases),
        "",
        f"## Top {args.top} kernels (per iteration)",
        fmt(kt.head(args.top)),
    ]
    if att is not None and len(att):
        lines += ["", "## Thread trace (ATT): compute issued while global loads are in flight",
                  fmt(att)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.md, phases.csv, kernel_summary.csv, kernels.csv, timeline.png", flush=True)


def _plot_timeline(run, peak, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = sorted(run["iter"].unique())[len(run["iter"].unique()) // 2]
    g = run[run["iter"] == it].sort_values("Start_Timestamp")
    t0 = g["Start_Timestamp"].min()
    x = (g["Start_Timestamp"] - t0).to_numpy() / 1e6
    w = (g["End_Timestamp"] - g["Start_Timestamp"]).to_numpy() / 1e6
    colors = {"vision": "tab:blue", "llm": "tab:orange", "denoiser": "tab:green"}
    c = g["phase"].map(colors).to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    ax[0].bar(x, g["fabric_GBs"], width=w, align="edge", color=c, linewidth=0)
    ax[0].axhline(peak, ls="--", c="k", lw=0.8)
    ax[0].text(0, peak, f" copy peak {peak:.0f} GB/s", va="bottom", fontsize=8)
    ax[0].set_ylabel("L2-miss GB/s\n(MALL + DRAM)")
    ax[1].bar(x, g["mem_unit_busy_pct"], width=w, align="edge", color=c, linewidth=0)
    ax[1].set_ylabel("Memory unit busy %")
    ax[1].set_ylim(0, 105)
    ax[2].bar(x, g["valu_issue_pct"], width=w, align="edge", color=c, linewidth=0)
    ax[2].set_ylabel("VALU issue %\n(lower bound)")
    ax[2].set_xlabel(f"time in {it} (ms, phases separated by sync)")
    for p, col in colors.items():
        ax[0].bar(0, 0, color=col, label=p)
    ax[0].legend(loc="upper right")
    fig.suptitle("pi05 1-NFE: per-kernel L2-miss bandwidth vs. compute activity (gfx1151)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# -------------------------------------------------------------------------------- ATT
_LOAD = re.compile(r"^(buffer|global|flat|scratch)_(load|atomic)")
_STALL_MIN = 8  # an s_waitcnt lasting longer than this was actually waiting


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _covered(times, merged):
    import numpy as np

    if not merged or not len(times):
        return 0
    st = np.array([m[0] for m in merged])
    en = np.array([m[1] for m in merged])
    i = np.searchsorted(st, times, side="right") - 1
    return int(((i >= 0) & (times < en[np.clip(i, 0, None)])).sum())


def _att_dispatch(d: Path):
    """Overlap of VALU issue with outstanding global loads on the traced SIMD.

    Load i is outstanding from issue until the s_waitcnt vmcnt(N) that retires it
    (loads retire in order). If that wait actually stalled, the newest retired load
    was provably in flight for the whole interval ("definite").
    """
    import numpy as np

    code_j = json.load(open(d / "code.json"))["code"] or []
    code = {x[2]: x[0] for x in code_j if len(x) > 2}
    kname = str(code_j[0][0]).lstrip("; ").strip() if code_j else d.name
    simds: dict = {}
    for f in glob.glob(str(d / "se*_sm*_sl*_wv*.json")):
        w = json.load(open(f))["wave"]
        se = Path(f).name.split("_")[0]
        s = simds.setdefault((se, w["cu"], w["simd"]), {"waves": [], "bound": [], "definite": [],
                                                        "blocked": [], "valu": []})
        s["waves"].append((w["begin"], w["end"]))
        loads = deque()
        for t, _ty, _stall, du, pc in w["instructions"]:
            txt = code.get(pc, "")
            m = txt.split(" ", 1)[0]
            if m.startswith("v_"):
                s["valu"].append(t)
            elif _LOAD.match(m) and "store" not in m:
                loads.append(t)
            elif m == "s_waitcnt" and "vmcnt" in txt:
                n = int(re.search(r"vmcnt\((\d+)\)", txt).group(1))
                tend = t + du
                waited = du > _STALL_MIN and "lgkmcnt" not in txt
                if waited:
                    s["blocked"].append((t, tend))
                newest = None
                while len(loads) > n:
                    newest = loads.popleft()
                    s["bound"].append((newest, tend))
                if newest is not None and waited:
                    s["definite"].append((newest, tend))
        s["bound"] += [(t, w["end"]) for t in loads]
    rows = []
    for key, s in simds.items():
        if not s["waves"]:
            continue
        b0 = min(a for a, _ in s["waves"])
        span = max(b for _, b in s["waves"]) - b0
        valu = np.sort(np.array(s["valu"]))
        B, D = _merge(s["bound"]), _merge(s["definite"])
        cov = lambda m: sum(b - a for a, b in m)  # noqa: E731
        # exposed memory time: every resident wave is blocked on vmcnt
        ev = [(a, 1, 0) for a, _ in s["waves"]] + [(b, -1, 0) for _, b in s["waves"]]
        ev += [(a, 0, 1) for a, _ in s["blocked"]] + [(b, 0, -1) for _, b in s["blocked"]]
        ev.sort()
        res = blk = 0
        exposed = 0
        prev = None
        for t, dr, db in ev:
            if prev is not None and res > 0 and blk >= res:
                exposed += t - prev
            res += dr
            blk += db
            prev = t
        rows.append({
            "kernel": _short(kname, 48), "simd": "/".join(map(str, key)), "waves": len(s["waves"]),
            "load_in_flight_pct": 100 * cov(B) / span,
            "load_in_flight_definite_pct": 100 * cov(D) / span,
            "valu_per_cycle": len(valu) / span,
            "valu_during_load_pct": 100 * _covered(valu, B) / max(len(valu), 1),
            "valu_during_definite_load_pct": 100 * _covered(valu, D) / max(len(valu), 1),
            "exposed_mem_wait_pct": 100 * exposed / span,
        })
    return rows


def _att_report(out: Path):
    import pandas as pd

    rows = []
    for d in sorted((out / "att").glob("ui_output_agent_*_dispatch_*")):
        for r in _att_dispatch(d):
            r["dispatch"] = int(d.name.rsplit("_", 1)[1])
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("dispatch")
    df.to_csv(out / "att.csv", index=False)
    cols = ["dispatch", "kernel", "waves", "load_in_flight_pct", "load_in_flight_definite_pct",
            "valu_per_cycle", "valu_during_load_pct", "valu_during_definite_load_pct", "exposed_mem_wait_pct"]
    return df[cols]


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default=str(PKG_ROOT / "eval_out/memcompute"))
    c.add_argument("--iters", type=int, default=3, help="profiled steady-state iterations")
    c.add_argument("--passes", default="trace," + ",".join(COUNTER_PASSES))
    c.add_argument("--att-lib", default=os.environ.get("ROCPROF_ATT_LIBRARY_PATH", ""),
                   help="dir with librocprof-trace-decoder.so; enables the thread-trace pass")
    c.add_argument("--warmup", type=int, default=4, help="unprofiled warmup iterations")
    c.add_argument("--att-regex", default="gemm_iu4", help="kernel(s) to thread-trace")
    c.add_argument("--att-per-iter", type=int, default=122,
                   help="calls/iter of the traced kernel (kernel_summary.csv); used to skip warmup")
    c.add_argument("--att-index", default="1-7",
                   help="1-based calls within the profiled iteration (1-7 = first LLM layer)")
    r = sub.add_parser("report")
    r.add_argument("--out", default=str(PKG_ROOT / "eval_out/memcompute"))
    r.add_argument("--top", type=int, default=25)
    sub.add_parser("target")
    a = ap.parse_args()
    {"collect": collect, "report": report, "target": lambda _a: target()}[a.cmd](a)


if __name__ == "__main__":
    main()
