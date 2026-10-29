# Local validation — 2026-10-28

The extracted component completed 100 steps on AMD Radeon 8060S Graphics in the independent
`pi05-snapflow-rocm` conda environment. No Tether package or source path was used.
Training was offline using the original undistilled pi05 LIBERO teacher and real
LeRobot v3 LIBERO frames/actions.

| Setting / result | Observed value |
|---|---|
| PyTorch | 2.15.0a0+rocm10.1.0a20260902 |
| torchvision | 0.30.0a0+rocm10.1.0a20260902 |
| HIP | 7.16.26351 |
| LeRobot / Transformers | 0.6.1 / 5.5.4 |
| Steps / batch / seed | 100 / 4 / 42 |
| Precision / LR / alpha | bf16 / 1e-4 / 1.0 |
| First 50-step mean loss | 3.128750 |
| Last 50-step mean loss | 0.753047 |
| Loss decrease | 75.93% |
| Peak GPU allocation | 20.22 GiB |
| Training loop + final checkpoint | 448.95 seconds |
| Loading + training + save wall time | 577.37 seconds |

All logged losses and checked gradient norms were finite. The saved checkpoint
was strictly reloaded; the target-time MLP output projection had learned nonzero
weights. A real preprocessed dataset input produced finite raw actions with shape
`[1, 50, 32]` with exactly one denoiser call at `target_time=1`.

Six component tests passed, covering loss/shortcut math, teacher gradient isolation,
fp32 sampling time with bf16 projection, finite-epoch iteration, missing local
metadata, pinned local dataset/action chunks, and loss-window reporting. Python
syntax and shell syntax checks passed. The core file hashes match those recorded
before the 100-step run. The copied license and upstream source hashes were verified.

Local command (paths are existing inputs on the validation host):

```bash
conda activate pi05-snapflow-rocm
cd pi05-pytorch-fast
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/tether/hf \
  bash distill/snapflow/train.sh \
  --teacher /data/tether/checkpoints/pi05_libero_finetuned_v044 \
  --tokenizer /data/tether/tokenizers/paligemma-3b-pt-224 \
  --dataset-root /data/tether/lerobot/lerobot/libero \
  --output /data/pi05-pytorch-fast/snapflow-smoke-100-20261028 \
  --steps 100 --batch-size 4 --require-loss-decrease
```

Use a **new output directory** when repeating. The recorded run already exists.
The run artifacts are under `/data/pi05-pytorch-fast/snapflow-smoke-100-20261028`:
`training_log.jsonl`, `run_manifest.json`, `result.json`, `loss.png`, `report.md`,
`checkpoint_verification.json`, and `checkpoints/00000100/pretrained_model`.
The model is approximately 7.72 GiB and is not copied into the source repository.

This is a short-run stability and checkpoint/sampling smoke. It does not establish
LIBERO rollout success, converged policy quality, or fast-inference performance.


## From-zero installer and current pinned build — 20260822

The rewritten installer created a fresh Python 3.12 conda prefix at
`/data/pi05-pytorch-fast/conda-from-zero-20261028` using conda-forge, without
cloning or inheriting a source GPU environment. It installed the ROCm 10.1 /
PyTorch 2.15 cohort in `rocm_stack.json`, installed the training requirements,
and passed policy imports, exact-version checks, GPU detection, and bf16 GPU
matmul. No Tether package was present in the new environment. A setup record
and pip freeze are saved in that prefix.

A complete 100-step training run in this new environment succeeded:

| Result | Observed value |
|---|---|
| PyTorch | 2.15.0a0+rocm10.1.0a20260822 |
| torchvision | 0.30.0a0+rocm10.1.0a20260822 |
| ROCm package | 10.1.0a20260822 |
| HIP build | 7.16.26332 |
| First 50-step mean loss | 3.128750 |
| Last 50-step mean loss | 0.753047 |
| Loss decrease | 75.93% |
| Peak GPU allocation | 20.22 GiB |
| Training loop + final checkpoint | 640.82 seconds |

All losses and checked gradients were finite. The new checkpoint was strictly
reloaded, its target-time output projection had nonzero learned weights, and
one target-time-aware denoiser call produced finite `[1, 50, 32]` raw actions.
Core source hashes match the recorded experiment. Timing reflects the shared
validation host workload.

Run evidence: `/data/pi05-pytorch-fast/snapflow-from-zero-100-20261028`, including
loss curve, raw per-step logs, manifest, result, checkpoint, and reload evidence.

The independent downloader was also validated:

- It downloaded all five tokenizer files to an initially empty asset directory,
  using public access. Sizes, upstream hashes, and the original Google reference
  hashes passed. An interrupted TLS transfer resumed successfully.
- Its offline verification passed with network access disabled and an unusable
  Hub endpoint, demonstrating no hidden online requirement.
- The pinned teacher inventory (9 files) and dataset inventory (457 files) were
  fetched from the public Hub. All cached files passed their upstream sizes and
  LFS SHA-256 / Git-blob SHA-1 checks. Those approximately 8.76 GiB of existing
  teacher/data files were not downloaded again for this check.
- Twelve component tests passed, including checksum vectors, corruption/missing
  file rejection, path containment, pinned download requests, offline validation,
  failure-before-manifest behavior, and transient inventory retry.

Download evidence: `/data/pi05-pytorch-fast/assets-from-zero-20261028`, including
`assets_manifest.json` for the real tokenizer download,
`existing_assets_verification.json` for teacher/data inventory checks, and test logs.

Fresh Ubuntu driver/kernel and Anaconda installation commands were checked
against their official documentation, but were not executed on this already
configured host. No complete Docker image was pulled or executed; `IMAGES.md`
records registry/documentation checks separately.


## Published 10,000-step checkpoint

The previously completed Tether run was published to
[haojie233/pi05-snapflow-10k](https://huggingface.co/haojie233/pi05-snapflow-10k)
at immutable Hub revision `d002e044a4c2b87f2ff669eb2f37d91bfdedcc2e`.
Anonymous Hub access confirmed the repository is public. All release-manifest
files matched remote byte sizes, and LFS objects matched their SHA-256 hashes.
The unchanged model weights have SHA-256
`1aa6198111d8b05a2957811c1dbbbe788d6c28bd21589d41a2cc478b5a5c9f26`.
Packaged tokenizer encodings matched the training tokenizer on representative
English and Chinese inputs.

This run completed 10,000 steps with the converted LIBERO-90 mix teacher,
QUANTILES normalization, and the original LeRobot v3 LIBERO data. Its first and
last 50-step mean losses were 2.14640625 and 0.041123046875. These are recorded
training results, not a new rollout or latency evaluation. This teacher differs
from the default standalone recipe; the published model card records provenance.
Publication evidence is saved outside Git in
`/data/pi05-pytorch-fast/hf_pi05_snapflow_10k_publication.json`.
