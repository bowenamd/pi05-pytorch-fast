# Fastest gfx1151 PyTorch recipe for OpenPI π0.5 SnapFlow (1-NFE)

## SnapFlow distillation from a fresh machine

Follow the **[complete SnapFlow distillation guide](distill/snapflow/DISTILLATION.md)**
to start from a fresh Ubuntu 24.04 AMD GPU machine: install the GPU driver and
Anaconda, create a new conda environment with pinned ROCm 10.1 / PyTorch 2.15
nightly dependencies, download the teacher/tokenizer/LIBERO data, run 100 training
steps, and inspect the loss report. The guide also covers longer training,
checkpoint loading, and an alternative Ubuntu ROCm container workflow.

[Validation results](distill/snapflow/VERIFICATION.md) record the fresh conda setup
and successful 100-step training. [Container image details](distill/snapflow/IMAGES.md)
record the available images and their versions; the container workflow is documented
but has not been validated locally.

The standalone component requires no Tether checkout or personal fork. Extracted
code retains its [BUSL-1.1 license](distill/snapflow/LICENSE); the inference setup
below is separate.

## Published 10,000-step student

The completed bf16 SnapFlow student is available at
[haojie233/pi05-snapflow-10k](https://huggingface.co/haojie233/pi05-snapflow-10k). Download the fixed release:

```bash
HF_ENDPOINT=https://huggingface.co hf download haojie233/pi05-snapflow-10k \
  --revision d002e044a4c2b87f2ff669eb2f37d91bfdedcc2e \
  --local-dir "$HOME/model_data/pi05-snapflow-10k"
```

This release uses the converted LIBERO-90 mix teacher and **QUANTILES**
normalization. Keep its packaged processors and tokenizer together with the
weights. The existing latency/LIBERO measurements below retain their original
reference-checkpoint provenance. See the model card and
[standalone loading instructions](distill/snapflow/DISTILLATION.md#7-download-the-published-10000-step-student)
for this release's provenance and target-time-aware 1-NFE loading.

## Measured results (2026-09-02)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`).

| Metric | Value |
|--------|--------|
| E2E action chunk latency | **104 ms** default, **81 ms** with `PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1` (median; `scripts/bench_latency.py`, 1-NFE `sample_actions`) |
| E2E latency, fastest config (2026-11-26) | **64.8 ms** with the fast knobs + `PI05_W4A4_KERNEL=v2 PI05_W4A4_GLU=1` + TunableOp (on by default in `env.sh`); see [Latency](#latency-synthetic-not-sim) |
| Memory, fastest config (2026-11-26) | **7.6 GB** system memory above idle, peak = steady (was ~17 GB peak / ~12 GB steady); model load ~20 s (was ~80 s); see [Memory footprint](#memory-footprint) |
| LIBERO-10 accuracy | **20/20 = 100%** (2 episodes × 10 tasks, seed 0); unchanged with the knobs above (39/40 over seeds 0+1 either way). Fastest config: 19/20 at seed 0, same as the fast config |
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
   - opt-in `PI05_W4A4_KERNEL=v2`: bit-exact INT4 GEMM with 128-bit loads (weight rows padded to `K/8+4`)
   - opt-in `PI05_W4A4_GLU=1` (needs v2): `gelu_tanh(gate)·up` fused into the gate/up GEMM epilogue
   - K=16384 (down_proj) activation quant uses a bit-exact transpose-based FWHT kernel (always on)
   - TunableOp picks the fastest hipBLASLt/rocBLAS kernel per bf16/fp16 GEMM shape (`configs/tunableop/`)
5. Batched SigLIP (3 cameras in one forward); empty-camera pool **256→64** → prefix **776**
6. **`torch.compile(max-autotune-no-cudagraphs, fullgraph=True)`** + gfx1151 L2 antialias
7. LM / expert attention **eager** (Inductor); SigLIP **SDPA**
8. `n_action_steps=10`, `num_inference_steps=1`
9. Low-peak loader (`pi05_fast/fast_load.py`): meta-device init, W4A4 swap and unused `lm_head`s dropped
   before any weight is read, weights streamed tensor by tensor from safetensors to the GPU

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

Fastest config (TunableOp is already enabled by `env.sh`):

```bash
PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 PI05_W4A4_KERNEL=v2 PI05_W4A4_GLU=1 \
    python scripts/bench_latency.py                    #  65 ms
```

Check that the load log prints `kernel=v2 gelu-fused MLPs=18`.

Step by step, measured 2026-11-26 (fast config, `PI05_BENCH_ITERS=100`, median of 2
interleaved runs; GPU perf level `high`, CPU governor `performance`, raised `ryzenadj`
power limits, so absolute numbers are a few ms below the 2026-09 table):

| Change | E2E median | Output |
|---|---|---|
| fast config, original `gemm_iu4` | 79.5 ms | — |
| `PI05_W4A4_KERNEL=v2`: 128-bit global/LDS loads, one barrier per K tile, LDS-staged epilogue | 75.5 ms | bit-exact |
| `PI05_W4A4_GLU=1`: gelu·up in the gate/up epilogue (no fp16 gate/up round trip, no separate Triton kernel) | 68.5 ms | ≤1 fp16 ulp on 3.5e-5 of elements |
| K=16384 activation quant: 2 LDS transposes instead of 8 LDS passes (170 → 126 µs per call, copy-bandwidth floor) | 67.7 ms | bit-exact |
| TunableOp CSV (`configs/tunableop/gfx1151_fast0.csv`, 23 shapes: SigLIP, expert, attention bmm) | **64.8 ms** | different bf16 kernels |

To re-tune after a ROCm / PyTorch / hipBLASLt upgrade (the CSV records library
versions), delete `configs/tunableop/gfx1151_fast0.csv` and run the fastest config once
with `PYTORCH_TUNABLEOP_TUNING=1` (a few minutes); existing entries are otherwise kept. Shapes not in the CSV (e.g. the default 776-token prefix) fall back
to the default kernels. Disable with `PYTORCH_TUNABLEOP_ENABLED=0`.

v2 padding matters: with an unpadded power-of-2 weight row stride the same kernel was
1.7-3.4× slower (DRAM channel aliasing on gfx1151).

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
| fast + v2 + GLU (2026-11-25) | 0.98872 | 0.89258 | 0.97133 | 14.2% |
| fast + v2 + GLU + TunableOp (2026-11-26) | 0.98851 | 0.91143 | 0.97210 | 14.3% |

Nearly all deviation comes from W4A4 on the prefix LM; bf16/compile/layout changes are
lossless (>= 0.9999), and token trimming/camera cache add none.

Compiled runs are not bit-reproducible across processes: two processes of the same
config, loading the same cached compiled graph, agree only to cos ≈ 0.995 (max |Δ| ≈
0.07-0.15 in normalized actions), and W4A4 amplifies the difference. Within one process
and in eager mode (`PI05_COMPILE=0`) results are bit-identical, so use eager runs to show
that two versions are equal (e.g. v1 vs v2 kernels, stock vs fast loader), and compare
compiled configs only against the fp32 reference.

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

`proc_rss_mb` sums the whole process tree, including ~33 Inductor compile workers. Each
worker maps the same shared libraries, so their RSS (~16 GB together) is counted many
times while their private memory is ~0.1 GB in total. Use `host_mem_used_mb` above idle
for the real footprint.

### Memory footprint

System memory above idle, fastest config (fast knobs + v2 + GLU + TunableOp), measured
2026-11-26 with `monitor_util.py` around `bench_latency.py`:

| Change | Peak | Steady | Notes |
|---|---:|---:|---|
| before | ~17 GB | ~12 GB | stock LeRobot load: full model built in fp32 with random init, moved to the GPU (8.7 GB), then the whole checkpoint read into host memory |
| glibc mmap threshold pinned before loading | 16.9 GB | 9.1 GB | the freed ~4 GB load-time state dict stayed in the heap (`malloc_trim` cannot return it) |
| `PI05_DROP_LM_HEAD=1` | 16.9 GB | 7.7 GB | PaliGemma (0.98 GB) and expert (0.49 GB) `lm_head`s are never used for actions |
| `PI05_FAST_LOAD=1` | **7.6 GB** | **7.6 GB** | parameters on the meta device, W4A4 swap and head drop before loading, per-tensor copy to the GPU; load 80 s → 21 s |

All three are on by default and do not change the loaded parameters (all 947 tensors
bit-identical to the stock loader; eager actions bit-identical; LIBERO-10 seed 0 20/20;
latency unchanged). The steady 7.6 GB is ~3.5 GB of weights, ~1 GB of padded weight
copies made by the antialias freezing pass, activations/allocator slack, and ~2.6 GB of
CPU-side PyTorch runtime.

## Memory traffic vs. compute per kernel (rocprofv3)

`scripts/profile_mem_compute.py` profiles the steady-state 1-NFE path (each phase
compiled separately, kernels tagged `vision` / `llm` / `denoiser` with roctx ranges)
and joins one timing pass with four hardware-counter passes per kernel dispatch.
An optional thread-trace (ATT) pass shows, at instruction level, whether compute
issues while global loads are outstanding.

```bash
source env.sh
export PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1
# ~12 min (5 passes). Add --att-lib <dir with librocprof-trace-decoder.so> for the ATT pass.
python scripts/profile_mem_compute.py collect --out eval_out/memcompute
python scripts/profile_mem_compute.py report  --out eval_out/memcompute
# -> summary.md, phases.csv, kernel_summary.csv, kernels.csv (per dispatch),
#    timeline.png, att.csv, trace/r_results.pftrace (open in ui.perfetto.dev)
```

| Metric | Counters | Meaning |
|--------|----------|---------|
| `fabric_GBs` | `GL2C_EA_RDREQ_{32,64,96,128}B`, `GL2C_MC_WRREQ`, `GL2C_EA_WRREQ_64B` | L2-miss traffic / kernel time. Served by the 32 MB MALL **or** DRAM; no per-kernel counter separates them |
| `mem_unit_busy_pct` | `MemUnitBusy` (TA busy / GPU busy) | time the vector memory pipeline has work, incl. stalls |
| `valu_issue_pct` | `SQ_INSTS_VALU / (SIMDs x GRBM_GUI_ACTIVE)` | VALU issue rate; lower bound on ALU activity (one WMMA counts once but occupies many cycles) |
| `wave_wait_pct` | `SQ_WAIT_ANY / SQ_WAVE_CYCLES` | share of wave time spent waiting |
| ATT `valu_during_load_pct` | thread trace, traced SIMD | VALU/WMMA instructions issued while a global load is outstanding (`_definite`: load proven in flight because its `s_waitcnt vmcnt` stalled) |
| ATT `exposed_mem_wait_pct` | thread trace | time every resident wave on the SIMD is blocked on `s_waitcnt vmcnt`, i.e. memory latency not hidden by compute |

Counter passes serialize kernels, so timings come from the counter-free pass. Kernels on
one stream never overlap each other; overlap of memory and compute happens **inside** a
kernel, across waves. Measured 2026-11-04, fast config, 79.8 ms GPU time/iteration:

| Phase | GPU ms | L2-miss read / write MB | GB/s | Mem unit busy | VALU issue |
|-------|-------:|------------------------:|-----:|--------------:|-----------:|
| vision | 18.2 | 2111 / 378 | 137 | 66% | 8% |
| llm | 54.5 | 11781 / 1655 | 246 | 87% | 16% |
| denoiser | 7.0 | 1088 / 1 | 155 | 71% | 2% |

The 1 GB copy probe reaches 199 GB/s, but `gemm_iu4` sustains 268 GB/s of L2-miss
traffic: part of the weight/activation re-reads hit the MALL. (The SoC-wide DRAM
average from `monitor_util.py` is 75 GB/s.) ATT on the first LLM layer's 6 `gemm_iu4`
calls: loads are outstanding 75-99% of the time, 67-99% of WMMA/VALU instructions issue
while loads are outstanding (9-67% provably), and only 3-15% of time is exposed memory
wait.

## Layout

```
pi05-pytorch-fast/
├── README.md
├── RESULTS.md
├── constraints-rocm.txt
├── requirements-lerobot-runtime.txt
├── env.sh                      # generated by setup.sh
├── configs/tunableop/          # TunableOp GEMM selections (gfx1151, fast config)
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
    ├── profile_mem_compute.py  # per-kernel L2-miss traffic vs compute (rocprofv3)
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
| `PI05_W4A4_KERNEL` | `v1` | `v2`: pad packed weight rows to `K/8+4` at load and use the v2 INT4 GEMM (bit-exact, −4 ms) |
| `PI05_W4A4_GLU` | `0` | `1` (needs `v2`): fuse `gelu_tanh(gate)·up` into the gate/up GEMM (−7 ms). Only applied if `act_fn` matches `F.gelu(approximate="tanh")` |
| `PYTORCH_TUNABLEOP_ENABLED` / `_TUNING` / `_FILENAME` | `1` / `0` / `$ROOT/configs/tunableop/gfx1151_fast%d.csv` | Reuse tuned GEMM selections (−3 ms); `TUNING=1` re-tunes |
| `PI05_INDUCTOR_GEMM_BACKENDS` | `ATEN` | Inductor GEMM backend. ROCm 10.1 disables the origami heuristic, so Inductor mis-ranks triton over hipBLASLt; `ATEN` is ~2% faster. Set empty to restore triton autotune |
| `PI05_LANG_TOKENS` | unset (off) | Trim trailing **padded** language tokens to this bucket (prefix = 576 + N). Real tokens are never dropped: if the prompt does not fit, the full 200 are used and a warning is printed. `64` → prefix 640 = exactly 5 `gemm_iu4` `BM=128` tiles (−14% E2E, LIBERO-10 still 20/20) |
| `PI05_FAST_LOAD` | `1` | Low-peak loader (meta init, W4A4 swap before loading, streamed weights). `0` = stock `PI05Policy.from_pretrained` |
| `PI05_DROP_LM_HEAD` | `1` | Drop the unused PaliGemma and expert `lm_head` weights (−1.47 GB GPU) |
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
- `gemm_iu4` grid reorder for L2 weight reuse (≤2%: the kernel is limited by its own load
  instructions, not DRAM)
- GLU tile sweep (10 configs): the current `128×64×64` tile is already the fastest
- ONNX / MIGraphX / ORT (parent `pi05-migraphx` package)

## License / provenance

- W4A4 kernel layout (Hadamard on selected K).
- SnapFlow eval class adapted from Reflex VLA (Apache-2.0).
- Antialias pass from AMD rocm-scripts gfx1151 tests.
- Policy weights: Hugging Face (see `download_checkpoints.sh`).
