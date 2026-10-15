# Fastest gfx1151 PyTorch recipe for OpenPI π0.5 SnapFlow (1-NFE)

## Measured results (2026-09-02)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`).

| Metric | Value |
|--------|--------|
| E2E action chunk latency | **104 ms** default, **81 ms** with `PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1` (median; `scripts/bench_latency.py`, 1-NFE `sample_actions`) |
| LIBERO-10 accuracy | **20/20 = 100%** (2 episodes × 10 tasks, seed 0); unchanged with the knobs above (39/40 over seeds 0+1 either way) |
| `n_action_steps` / `num_inference_steps` | 10 / 1 |
| Prefix tokens | 776 (batched SigLIP + empty-camera pool 256→64); **640** with `PI05_LANG_TOKENS=64` |

Two episodes per task is a **smoke** for quantization/compile correctness, not a full 500-episode LIBERO score.

## Software (ROCm)

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon 8060S (`gfx1151`) |
| Host ROCm (`/opt/rocm`) | **7.2.4** (70204-93) |
| `rocm-smi` | 4.0.0+97f5574fe2 |
| Python | 3.12 |
| PyTorch (venv) | **2.15.0a0+rocm10.1.0a20260822** |
| HIP in PyTorch (`torch.version.hip`) | **7.16.26332** |
| Wheel extra | `torch[device-gfx1151]` from `https://rocm.nightlies.amd.com/whl-multi-arch/` |
| Arch override | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |

## What this is

Closed-loop **LeRobot PI05Policy** inference:

1. **SnapFlow 1-NFE** student (`Rylinjames/pi05-snapflow-distill-1nfe`)
2. **W4A4** packed INT4 GEMM on the **PaliGemma prefix LM only** (HIP custom op)
3. Prefix LM weights/activations in **fp16** (match W4A4; SigLIP stays bf16)
4. Shared activation quant for QKV and gate/up (`int4_gemm3` / `int4_gemm2`)
5. Batched SigLIP (3 cameras in one forward); empty-camera pool **256→64** → prefix **776**
6. **`torch.compile(max-autotune-no-cudagraphs, fullgraph=True)`** + gfx1151 L2 antialias
7. LM / expert attention **eager** (Inductor); SigLIP **SDPA**
8. `n_action_steps=10`, `num_inference_steps=1`

**pure PyTorch** path we have on this GPU.

## Hardware / OS

| Item | Value |
|------|--------|
| GPU | gfx1151 (e.g. Radeon 8060S / Ryzen AI Max+ 395) |
| ROCm | Host **7.2.4** (`/opt/rocm`) + `rocm[devel]` in the venv (see table above) |
| Python | 3.12 |
| Tools | `uv`, `git`, HIP compiler (for the INT4 extension) |

Set `HSA_OVERRIDE_GFX_VERSION=11.5.1` if the runtime does not detect gfx1151.

## One-time setup

```bash
cd pi05-pytorch-fast

# 1) Venv (does not touch onnx-infer / pi05-migraphx)
ROCM_ARCH=gfx1151 bash scripts/setup.sh -y
source env.sh

# 2) Weights + tokenizer + LIBERO sim (clones if missing)
# Must run after setup.sh: it needs the `hf` CLI installed into the venv.
bash scripts/download_checkpoints.sh all
# Needs HF token for gated paligemma tokenizer if google/paligemma2-3b-pt-224 is gated.
# LIBERO defaults to ~/model_data/libero; reuses ~/openpi/third_party/libero when present.
```

If you already have `~/.local/venvs/pi05-migraphx` from the parent project, you *can*
point `--venv` at it, but the intended distribute path is a **fresh**
`~/.local/venvs/pi05-pytorch-fast`.

## Pack W4A4 weights (once per checkpoint)

```bash
source env.sh
python scripts/pack_w4a4_prefix.py \
  --model ~/model_data/pi05_snapflow_1nfe \
  --out models/w4a4_prefix
```

`env.sh` sets `PI05_W4A4_PACK` to `models/w4a4_prefix`. First inference JIT-compiles
`pi05_fast/w4a4/csrc/int4_gemm.cu` (needs `ninja` + ROCm devel, already in setup).

## LIBERO eval (accuracy)

```bash
source env.sh
export MUJOCO_GL=egl

python scripts/libero_eval.py \
  --policy-id ~/model_data/pi05_snapflow_1nfe \
  --tasks libero_10 --episodes 2 --seed 0 \
  --n-inference-steps 1 --n-action-steps 10 \
  --output-dir eval_out/libero10_ep2

# same command with prefix trimming (also 20/20)
PI05_LANG_TOKENS=64 python scripts/libero_eval.py ... --output-dir eval_out/libero10_lang64
```

Videos: `eval_out/libero10_ep2/videos/libero_10_{id}/eval_episode_{0,1}.mp4`

`--resume` evaluates one task at a time (survives crashes; recompiles per task).

When using `PI05_LANG_TOKENS`, check the log says `lang tokens trimmed to 64` and does
**not** warn `instruction needs N tokens`: on that warning the prompt did not fit, the full
200 tokens were used, and the run measured the default path rather than the trimmed one.

## Latency (synthetic, not sim)

```bash
source env.sh

python scripts/bench_latency.py                        # 104 ms
PI05_LANG_TOKENS=64 python scripts/bench_latency.py    #  89 ms
PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 \
    python scripts/bench_latency.py                    #  81 ms
```

First call pays `torch.compile` autotune (minutes). Median after warmup is the number
to compare (**104 ms** on 8060S, **81 ms** with both knobs).

The synthetic prompt fills 52 of the 200 padded language tokens, matching LIBERO-10
(`PI05_BENCH_LANG_LEN` overrides it).

`download_checkpoints.sh` rewrites SnapFlow `config.json` once (drops `_reflex_*`
keys that stock LeRobot rejects). Later eval/bench runs are a no-op if already clean.

## Output fidelity vs. baseline (cosine similarity)

`scripts/compare_cosine.py` feeds identical real LIBERO inputs and fixed noise to each
config, then compares the action chunks with a baseline (normalized actions, 7 dims x 50 steps).

```bash
source env.sh
D=eval_out/cosine

# 1) Record real model inputs from LIBERO-10 rollouts (1 ep/task, every 3rd call -> 83 samples)
python scripts/compare_cosine.py capture --out $D/inputs.pt

# 2) Dump actions, one process per config (env knobs select the config)
python scripts/compare_cosine.py dump --upstream --dtype float32 --out $D/ref_fp32.pt   # stock lerobot fp32
python scripts/compare_cosine.py dump --upstream --out $D/upstream_bf16.pt              # stock lerobot bf16
PI05_W4A4_PACK= PI05_COMPILE=0 PI05_PREFIX_LAYOUT=openpi \
    python scripts/compare_cosine.py dump --out $D/rocm_bf16_eager.pt
python scripts/compare_cosine.py dump --out $D/rocm_w4a4.pt                             # default
PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 \
    python scripts/compare_cosine.py dump --out $D/fast.pt

# 3) Compare against the fp32 baseline (also writes JSON)
python scripts/compare_cosine.py compare --ref $D/ref_fp32.pt --json $D/vs_ref_fp32.json \
    $D/upstream_bf16.pt $D/rocm_bf16_eager.pt $D/rocm_w4a4.pt $D/fast.pt
```

Measured 2026-10-15 (83 samples, ref = stock lerobot fp32):

| Config | cos mean | cos min | cos P5 | relL2 |
|---|---|---|---|---|
| stock lerobot bf16 | 0.99996 | 0.99975 | 0.99991 | 0.8% |
| ROCm bf16 eager | 0.99995 | 0.99982 | 0.99988 | 1.0% |
| W4A4 + compile (default) | 0.98751 | 0.87691 | 0.97079 | 14.8% |
| fast (W4A4 + `LANG_TOKENS=64` + `CACHE_EMPTY_CAM=1`) | 0.98791 | 0.86084 | 0.96810 | 14.4% |

Nearly all deviation comes from W4A4 on the prefix LM; bf16/compile/layout changes are
lossless (>= 0.9999), and token trimming/camera cache add none.

## Resource utilization (CPU / GPU / DRAM)

`scripts/monitor_util.py` runs any command as a child process and samples every
100 ms. It only reads `/proc` and amdgpu sysfs: no root, no `rocprofv3`, no
change to the measured process.

```bash
source env.sh

# 1) Monitor the fast config. Summary is printed for the timed loop only.
PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 PI05_BENCH_ITERS=400 \
    python scripts/monitor_util.py --csv eval_out/util/bench_fast.csv -- \
    python scripts/bench_latency.py

# 2) Line charts: full run (each phase shaded) and timed loop only.
python scripts/plot_util.py eval_out/util/bench_fast.csv            # -> bench_fast.all.png
python scripts/plot_util.py eval_out/util/bench_fast.csv --phase steady   # -> bench_fast.steady.png
```

Use `PI05_BENCH_ITERS=400` (~30 s) so the window is much longer than the 1 s SMU
filter. Each sample's `phase` is set by `bench_latency.py` stdout lines:

| Phase | Starts at line | What runs |
|-------|----------------|-----------|
| `init` | process start | Python/torch imports, checkpoint sanitize, hook install |
| `load` | `loading ...` | `PI05Policy.from_pretrained`, W4A4 pack load + Linear swap, move to GPU |
| `compile` | `warmup / compile...` | first `sample_actions` call: `torch.compile` + max-autotune |
| `warmup` | `first call done` | remaining `PI05_BENCH_WARMUP - 1` calls |
| `steady` | `timed loop start` | timed loop (the latency number) |
| `done` | `n=...` | result print, teardown |

The monitor prints a per-phase table and a detailed `steady` summary, dropping
the first `--skip-after-start` seconds (default 2) of `steady`. To wrap another
command, set `--start-marker` / `--end-marker` (regex on its stdout for
`steady` / `done`); markers that never appear are skipped.

| Column | Source |
|--------|--------|
| `sys_cpu_pct` | `psutil.cpu_percent()` (`/proc/stat`), whole machine |
| `proc_cpu_pct` | `psutil` per-process CPU summed over the process tree (100 = 1 core) |
| `proc_rss_mb`, `host_mem_used_mb` | `psutil` RSS / `virtual_memory()` (unified memory: RSS includes GPU-mapped pages) |
| `gpu_busy_pct` | sysfs `gpu_busy_percent` |
| `vram_used_mb`, `gtt_used_mb` | sysfs `mem_info_vram_used` / `mem_info_gtt_used` |
| `gfx_activity`, `dram_reads/writes` (MB/s), `*_power` (W), `gfxclk/fclk/uclk` (MHz) | sysfs `gpu_metrics` parsed as `struct gpu_metrics_v3_0` (`kgd_pp_interface.h`) |

`gpu_metrics` values are SMU time-filtered (~1 s), so they show averages, not
per-inference phases (use `rocprofv3` counters for per-kernel detail). DRAM
bandwidth is SoC-wide (CPU + GPU); `dram_bw_pct` is relative to the 256 GB/s
LPDDR5X-8000 x 256-bit spec, not a measured peak. `amd-smi metric` reports N/A
for most of these on gfx1151, hence the direct sysfs read.

Measured on 8060S, fast config (81 ms median), per phase (warm Inductor cache;
a cold first compile takes minutes):

| Phase | Start (s) | Duration (s) | CPU mean / peak (cores) | GPU busy (%) | DRAM (GB/s) | RSS end (GB) | GTT end (GB) |
|-------|----------:|-------------:|------------------------:|-------------:|------------:|-------------:|-------------:|
| init | 0.0 | 3.3 | 3.4 / 24.1 | 0.0 | 1.5 | 1.0 | 0.0 |
| load | 3.3 | 63.7 | 2.8 / 31.3 | 0.7 | 6.0 | 6.7 | 5.7 |
| compile | 67.0 | 23.0 | 1.6 / 26.7 | 7.4 | 5.1 | 22.6 | 6.6 |
| warmup | 90.0 | 0.6 | 2.0 / 2.1 | 45.0 | 25.0 | 22.6 | 6.6 |
| steady | 90.6 | 32.3 | 2.0 / 2.2 | 97.7 | 73.3 | 22.6 | 6.6 |
| done | 122.9 | 2.6 | 1.9 / 2.1 | 35.9 | 27.6 | 0.0 | 0.1 |
| **total** | | **125.5** | | | | | |

GPU/DRAM means of short phases (`warmup`, `done`) and the start of `steady` are
smeared by the 1 s SMU filter. Timed loop (`steady`, first 2 s dropped):

| Metric | Mean |
|--------|------|
| GPU busy | 98.8% (GPU-bound, no launch gaps) |
| CPU | ~2 cores (6.7% of 32 logical) |
| DRAM bandwidth | 74.6 GB/s (46.7 read + 27.9 write) ≈ 29% of spec, ~6 GB / inference |
| Memory | host used 39 GB, process RSS 23 GB, GTT 6.8 GB, VRAM 1.3 GB |
| Power | socket 140 W, gfx 55 W, gfxclk 2900 MHz |

## Layout

```
pi05-pytorch-fast/
├── README.md
├── RESULTS.md
├── constraints-rocm.txt
├── requirements-lerobot-runtime.txt
├── env.sh                      # generated by setup.sh
├── pi05_fast/                  # runtime hooks (no ONNX)
│   ├── rocm_pi05_optim.py
│   ├── rocm_antialias.py
│   ├── snapflow_pi05.py
│   └── w4a4/               
└── scripts/
    ├── setup.sh
    ├── download_checkpoints.sh
    ├── pack_w4a4_prefix.py
    ├── libero_eval.py
    ├── bench_latency.py
    ├── compare_cosine.py       # action cosine similarity vs. baseline
    ├── monitor_util.py         # CPU/GPU/DRAM sampler -> CSV + summary
    └── plot_util.py            # CSV -> line charts
```

## Environment knobs

| Env | Default | Meaning |
|-----|---------|---------|
| `PI05_W4A4_PACK` | `$ROOT/models/w4a4_prefix` | Packed INT4 LM weights |
| `PI05_COMPILE` | `1` | `torch.compile` |
| `PI05_COMPILE_MODE` | `max-autotune-no-cudagraphs` | Inductor mode |
| `PI05_PREFIX_LM_FP16` | `1` | Cast prefix LM to fp16 after W4A4 |
| `PI05_W4A4_FUSE_QUANT` | `1` | Shared QKV / gate-up quant |
| `PI05_INDUCTOR_GEMM_BACKENDS` | `ATEN` | Inductor GEMM backend. ROCm 10.1 disables the origami heuristic, so Inductor mis-ranks triton over hipBLASLt; `ATEN` is ~2% faster. Set empty to restore triton autotune |
| `PI05_LANG_TOKENS` | unset (off) | Trim trailing **padded** language tokens to this bucket (prefix = 576 + N). Real tokens are never dropped: if the prompt does not fit, the full 200 are used and a warning is printed. `64` → prefix 640 = exactly 5 `gemm_iu4` `BM=128` tiles (−14% E2E, LIBERO-10 still 20/20) |
| `PI05_CACHE_EMPTY_CAM` | `0` (off) | Cache the empty-camera SigLIP embedding. With `empty_cameras=1` the placeholder camera is a constant `-1` image, so its embedding never changes, yet a full vision forward runs on it every frame. Caching it batches only the real cameras (−7 ms). Not bit-exact: bf16 SigLIP output depends on batch size (rel err ~4e-3) |
| `SNAPFLOW_TEACHER_POLICY` | `~/model_data/pi05_libero_v044` | Processor json/safetensors |
| `PALIGEMMA_TOKENIZER_PATH` | `~/model_data/paligemma2-3b-pt-224` | Local tokenizer |

## What we tried and did **not** keep

- SigLIP fp16 (slower than bf16 on this GPU)
- SigLIP / LM / expert FlashAttention-2 (no win; FA2 + 4D mask crashes)
- LM SDPA (slower than eager with additive 4D masks)
- Quantizing SigLIP 4-bit (MLP `K=4304` not multiple of 64; attn projections cost ~1 LIBERO
  episode for only −1.7% latency)
- W4A4 on the action expert (at `M=50` the INT4 path is *slower* than bf16 hipBLASLt)
- ONNX / MIGraphX / ORT (parent `pi05-migraphx` package)

## License / provenance

- W4A4 kernel layout (Hadamard on selected K).
- SnapFlow eval class adapted from Reflex VLA (Apache-2.0).
- Antialias pass from AMD rocm-scripts gfx1151 tests.
- Policy weights: Hugging Face (see `download_checkpoints.sh`).
