# pi0.5 SnapFlow: reproduce from a fresh Ubuntu machine

This directory contains the standalone default pi0.5 SnapFlow trainer, its
from-zero conda installer, pinned asset downloader, and training/report commands.
No existing ROCm conda environment, Tether checkout, personal fork, or downloaded
student is required. Extracted code retains BUSL-1.1: see `LICENSE`, `NOTICE.md`,
and `source_manifest.json`.

The reference platform is **Ubuntu 24.04 x86_64, AMD Radeon 8060S / Ryzen AI Max+
(gfx1151), Python 3.12**. A usable AMD GPU kernel driver is required. The scripts
install the **ROCm 10.1 userspace SDK and PyTorch 2.15 into conda**, not a kernel
driver. The SDK's package version and `torch.version.hip` are different version
identifiers; the latter can report a 7.16 build even with this ROCm 10.1 SDK.

Allow space for the environment, approximately 8.78 GiB of reference inputs,
and approximately 7.72 GiB per saved student. At least 50 GiB free disk space is
recommended. The earlier 100-step test allocated about 20.22 GiB on the GPU;
allow additional CPU/UMA memory for loading two full policies and decoding data.

## Alternative: reproduce inside an Ubuntu ROCm container

You can also reproduce the training workflow inside an AMD Ubuntu container.
The host still needs working AMD GPU drivers and permissions from step 1.
The official image below provides Ubuntu 24.04 and ROCm 10.1 with PyTorch 2.14;
inside it, use `setup_conda.sh` to install this project's pinned PyTorch 2.15
training environment. This container workflow is documented but has not been
validated locally. See `IMAGES.md` for the image versions and registry evidence.

From the project root on the host:

```bash
export SNAPFLOW_CONTAINER_DATA="$HOME/model_data/pi05-snapflow"
mkdir -p "${SNAPFLOW_CONTAINER_DATA}"
docker pull rocm/pytorch@sha256:099f59a3bee9c436f5cc57db6bc6076fdeda579c57e9e22ae40f856d5905b3e4

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --shm-size=8g \
  -v "$PWD:/workspace/pi05-pytorch-fast" \
  -v "${SNAPFLOW_CONTAINER_DATA}:/data" \
  -w /workspace/pi05-pytorch-fast \
  rocm/pytorch@sha256:099f59a3bee9c436f5cc57db6bc6076fdeda579c57e9e22ae40f856d5905b3e4 \
  bash
```

Inside the container, use its existing conda installation if available; otherwise
install Anaconda as in step 2. Then create the pinned environment on the mounted
data volume so the environment, downloaded inputs, and checkpoints survive exit:

```bash
SNAPFLOW_CONDA_PREFIX=/data/conda/pi05-snapflow \
  bash distill/snapflow/setup_conda.sh
conda activate /data/conda/pi05-snapflow
export SNAPFLOW_DATA_DIR=/data
export HF_HOME="${SNAPFLOW_DATA_DIR}/hf-cache"
```

Continue with steps 4–6 inside the container: download assets, run `train.sh`,
and generate the loss report. Host driver installation/reboot commands in
step 1 are for the host, not the container.

## 1. Prepare Ubuntu and GPU access

On a fresh Ubuntu 24.04 machine:

```bash
sudo apt update
sudo apt install -y ca-certificates curl wget git build-essential libnuma1 libgl1 libglib2.0-0
```

For Ryzen APUs including gfx1151, AMD's Ubuntu 24.04 instructions specify the
OEM kernel. Install it on the fresh host if needed:

```bash
sudo apt install -y linux-oem-24.04d
```

If the host does not yet have a usable AMD GPU driver, install the driver using
the official ROCm 10.1 installer. The following selects **the driver only**;
the conda setup in step 3 installs the userspace ROCm stack separately:

```bash
mkdir -p "$HOME/Downloads/rocm-setup"
cd "$HOME/Downloads/rocm-setup"
curl -fL --retry 3 -O https://stable.repo.amd.com/rocm/core/linux-installers/rocm-installer-10.1.0-5.run
sudo bash rocm-installer-10.1.0-5.run deps=install amdgpu
sudo usermod -a -G render,video "$USER"
sudo reboot
```

If the host already has a working driver, keep it and configure GPU group access
as needed. After reboot/re-login, check:

```bash
uname -r
ls -l /dev/kfd /dev/dri/renderD*
id
# /dev/kfd and a render node must be accessible to the training user.
test -r /dev/kfd && test -w /dev/kfd
```

The conda installer does not modify the host kernel, `/opt/rocm`, or driver
configuration. Use AMD's [ROCm 10.1 installation guide](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html)
and [compatibility matrix](https://rocm.docs.amd.com/en/docs-10.1.0/compatibility/compatibility-matrix.html)
for the GPU/OS-specific host requirements. Host driver/kernel installation is
not covered by a conda-environment test on an already configured machine.

## 2. Install Anaconda

Follow the official [Linux installation guide](https://www.anaconda.com/docs/getting-started/anaconda/install/linux-install).
The following pins its currently documented Linux x86_64 installer:

```bash
mkdir -p "$HOME/Downloads/anaconda-setup"
cd "$HOME/Downloads/anaconda-setup"
curl -fL --retry 3 -O https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Linux-x86_64.sh
sha256sum Anaconda3-2026.07-1-Linux-x86_64.sh
# Compare with this installer's SHA-256 in https://repo.anaconda.com/archive/.
bash Anaconda3-2026.07-1-Linux-x86_64.sh
```

Accept the installer terms if applicable and select the default
`$HOME/anaconda3` prefix. Initialize the current shell:

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda --version
```

An existing Miniconda/Miniforge/Anaconda installation also works; it does not need
any pre-existing GPU environment. The setup creates Python from `conda-forge`
with `--no-default-packages`, rather than cloning another environment.

## 3. Create conda and install ROCm 10.1, PyTorch 2.15, and other dependencies

Obtain the project's source release/check-out, then run from its root:

```bash
cd /path/to/pi05-pytorch-fast
bash distill/snapflow/setup_conda.sh
conda activate pi05-snapflow-rocm
```

`setup_conda.sh` executes these stages:

1. Create a dedicated, empty conda environment with Python 3.12 and pip.
2. Install the exact AMD torch/torchvision GPU extras, ROCm SDK/libraries, and
   Triton specified in `rocm_stack.json` from AMD's public wheel index.
3. Install LeRobot 0.6.1 with `--no-deps`, then the fixed Python training
   dependencies in `requirements.txt`. This includes Transformers 5.5.4, PyAV
   15.1.0, dataset processing, plotting, and pytest.
4. Validate versions, policy/dataset imports, AMD GPU detection, and bf16 GPU
   matrix multiplication. Write `snapflow_environment.json` and
   `snapflow_pip_freeze.txt` in the new conda prefix.

The default stack is fixed, not `latest`:

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.15.0a0+rocm10.1.0a20260822 |
| torchvision | 0.30.0a0+rocm10.1.0a20260822 |
| ROCm SDK and runtime libraries | 10.1.0a20260822 |
| LeRobot | 0.6.1 |
| Transformers | 5.5.4 |
| PyAV | 15.1.0 |

These are **nightly builds**, not the PyTorch 2.15 stable release. The earlier
100-step test used a 20260902 build. That date is no longer advertised in the
public index inspected on 2026-10-28, so the from-zero recipe pins the available
20260822 cohort. See `VERIFICATION.md` for validation evidence by build.
LeRobot's declared torch/torchvision upper bounds exclude these nightlies;
installing it with `--no-deps` is deliberate. The installer pins the ROCm stack
throughout dependency installation and checks the resulting versions.

Alternative environment name and GPU architecture:

```bash
SNAPFLOW_CONDA_ENV=pi05-snapflow-clean SNAPFLOW_GPU_ARCH=gfx1151 \
  bash distill/snapflow/setup_conda.sh
conda activate pi05-snapflow-clean
```

Use `SNAPFLOW_CONDA_PREFIX=/path/to/new/conda-prefix` to place the environment
on another disk. `SNAPFLOW_ROCM_INDEX` and `SNAPFLOW_PYPI_INDEX` support an
organization's package mirrors. A mirror must contain the exact pinned cohort.
If a future public index removes it, preserve these artifacts in an organization
repository; the installer intentionally does not switch to another nightly.

`SNAPFLOW_SKIP_GPU_CHECK=1` is available for CPU-only image builders. That only
checks installation/imports and does **not** validate GPU operation. For training,
keep the default GPU check. Do not activate the fast inference `env.sh` here:
it selects a separate environment and inference optimizations.

## 4. Download the undistilled teacher, tokenizer, and real training data

Choose a data directory outside the source checkout. No local inputs need to
exist before this command:

```bash
export SNAPFLOW_DATA_DIR="$HOME/model_data/pi05-snapflow"
export HF_HOME="${SNAPFLOW_DATA_DIR}/hf-cache"
export HF_ENDPOINT=https://huggingface.co
bash distill/snapflow/download_assets.sh --data-dir "${SNAPFLOW_DATA_DIR}"
```

The script downloads fixed snapshots, resumes interrupted downloads, and verifies
**every file's byte size and upstream LFS SHA-256 or Git-blob SHA-1**. It writes
`assets_manifest.json` only after verification of each component. The sources are:

| Input | Source and fixed revision | Local directory under data root |
|---|---|---|
| Undistilled teacher | `lerobot/pi05_libero_finetuned_v044` / `7152f13e34c6c115d808aea0e5d08aca74eac14b` | `checkpoints/pi05_libero_finetuned_v044` |
| LeRobot v3 LIBERO training data | `lerobot/libero` / `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` | `datasets/libero` |
| Tokenizer | `leo009/paligemma-3b-pt-224` / `39996beb6fb17c5d16a50d3ef8f7a96ad9d03986` | `tokenizers/paligemma-3b-pt-224` |

The tokenizer download selects only its five tokenizer files, not the model
weights in that repository. Their hashes are additionally checked against the
original Google reference used by the extracted experiment. The public copy
avoids requiring a developer's gated-model credentials for this reference
recipe; tokenizer/model/data licensing remains that of the original assets.
The downloader uses public unauthenticated access, not saved personal tokens.

Verify previously downloaded inputs without network access:

```bash
bash distill/snapflow/download_assets.sh \
  --data-dir "${SNAPFLOW_DATA_DIR}" --verify-only
```

Download/verify just one component with `--only teacher`, `--only dataset`, or
`--only tokenizer`. If a file is corrupted, repeat its download with
`--force-download`. Transient inventory transport/rate-limit errors are retried
three times by default (`--retries`); file transfers use Hub resume/retry support.
No student checkpoint is fetched.

## 5. Run the 100-step smoke and inspect loss

Use a fresh output directory for every run:

```bash
export SNAPFLOW_RUN_DIR="${SNAPFLOW_DATA_DIR}/runs/smoke-100-$(date +%Y%m%d-%H%M%S)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash distill/snapflow/train.sh \
  --teacher "${SNAPFLOW_DATA_DIR}/checkpoints/pi05_libero_finetuned_v044" \
  --tokenizer "${SNAPFLOW_DATA_DIR}/tokenizers/paligemma-3b-pt-224" \
  --dataset-root "${SNAPFLOW_DATA_DIR}/datasets/libero" \
  --output "${SNAPFLOW_RUN_DIR}" \
  --steps 100 --batch-size 4 --learning-rate 1e-4 --seed 42 \
  --require-loss-decrease

python -m distill.snapflow.report "${SNAPFLOW_RUN_DIR}"
cat "${SNAPFLOW_RUN_DIR}/result.json"
```

Defaults: bf16, consistency alpha 1, PyAV, 2 loader workers, AdamW betas
(0.9, 0.999), eps 1e-8, and weight decay 0.01. The teacher is frozen. The default
student's VLM prefix is frozen because its KV cache is detached; its action
expert, time/action projections, and target-time MLP train. Loss/time conventions
are retained from Tether.

The loader reads real 50-step future action chunks (teacher chunk size), applies
teacher normalization, and pads the action dimension. Single-frame actions are
rejected rather than tiled. `--require-loss-decrease` compares the first and
last 50-step mean losses, not individual noisy steps. Non-finite losses and
checked gradients fail the run.

## 6. Outputs, longer training, and tests

Outputs include `training_log.jsonl`, `run_manifest.json`, `result.json`,
`loss.png`, `report.md`, and the final
`checkpoints/00000100/pretrained_model/`. Each checkpoint has full student
weights including target-time MLP, native PI05 config and processor state, and
separate provenance. Existing runs are not overwritten.

For a longer training run, select a new output and replace the training options
with `--steps 10000 --checkpoint-every 5000`. Checkpoints do not include optimizer,
RNG, or loader state and do not support exact training resume.

```bash
PYTHONPATH=. python -m pytest distill/snapflow/tests -q
```

For target-time-aware single-step sampling in this training environment,
`distill.snapflow.runtime.load_policy(path, student=True).eval()` strictly loads
a saved student; invoke `policy.model.sample_actions_1step(images, image_masks,
tokens, language_masks)`. Stock LeRobot sampling does not activate the extra
target-time conditioning. The fast inference environment has separate hooks
and packing; its quality and latency need separate validation.

A short-run loss decrease establishes training stability only, not LIBERO
success, converged policy quality, or fast-inference speedup. See
`VERIFICATION.md` for recorded runs and `IMAGES.md` for Ubuntu container research.


## 7. Download the published 10,000-step student

Model URL: **[haojie233/pi05-snapflow-10k](https://huggingface.co/haojie233/pi05-snapflow-10k)**.
This is the previously completed 10,000-step student; its model tensor bytes
were retained unchanged. Native configs select 1-NFE and disable compilation
and gradient checkpointing. The upload includes tokenizer files, preprocessing,
postprocessing, normalizer state, training summary, and release hashes.

After preparing the conda environment, download the immutable release:

```bash
export SNAPFLOW_STUDENT_DIR="$HOME/model_data/pi05-snapflow-10k"
HF_ENDPOINT=https://huggingface.co hf download haojie233/pi05-snapflow-10k \
  --revision d002e044a4c2b87f2ff669eb2f37d91bfdedcc2e \
  --local-dir "${SNAPFLOW_STUDENT_DIR}"
```

This existing run used teacher `aayushis196/pi05-libero90mix-taco-checkpoints`
at revision `af13ddd7bbd028f81cb52645ae676efba2a98f3b`, converted from OpenPI
checkpoint `pi05_l90mix_pretrain_bs64_lr2.5e-5_10k/9999`. Distillation used the
original `lerobot/libero` v3 input, batch 4, LR 1e-4, seed 42, and bf16.
The model uses **QUANTILES** normalization from that teacher. The default
from-zero recipe in this guide uses the different `pi05_libero_finetuned_v044`
teacher; it describes a fresh experiment and does not promise to recreate this
published checkpoint. Detailed provenance and input/model terms are in the
Hub model card.

Load the student with the SnapFlow-aware loader, using the packaged local
tokenizer and processor statistics:

```python
import os
from pathlib import Path
from distill.snapflow.runtime import load_policy, build_preprocessor

checkpoint = Path(os.environ["SNAPFLOW_STUDENT_DIR"])
policy = load_policy(checkpoint, student=True).eval()
preprocessor = build_preprocessor(checkpoint, tokenizer=checkpoint)
```

Use `policy.model.sample_actions_1step(...)` for target-time-aware inference.
Stock LeRobot's loader/sampler needs the corresponding SnapFlow extension.
The student is an inference artifact, not an undistilled `--teacher` input;
it also does not include optimizer/RNG state for exact training resume.
