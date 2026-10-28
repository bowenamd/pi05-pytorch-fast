#!/usr/bin/env bash
# Install from an empty conda environment; no source environment is cloned.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-conda}"
TARGET_ENV="${SNAPFLOW_CONDA_ENV:-pi05-snapflow-rocm}"
GPU_ARCH="${SNAPFLOW_GPU_ARCH:-gfx1151}"
ROCM_INDEX="${SNAPFLOW_ROCM_INDEX:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
PYPI_INDEX="${SNAPFLOW_PYPI_INDEX:-https://pypi.org/simple}"
SKIP_GPU="${SNAPFLOW_SKIP_GPU_CHECK:-0}"
if [[ "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: bash distill/snapflow/setup_conda.sh
Creates Python 3.12 + ROCm 10.1 + PyTorch 2.15 + training dependencies.
Environment variables:
  SNAPFLOW_CONDA_ENV       Target name (default: pi05-snapflow-rocm)
  SNAPFLOW_CONDA_PREFIX    Use a target prefix instead of a name
  SNAPFLOW_GPU_ARCH        GPU architecture (default: gfx1151)
  SNAPFLOW_ROCM_INDEX      AMD package index or an organization mirror
  SNAPFLOW_PYPI_INDEX      Python package index or an organization mirror
  SNAPFLOW_SKIP_GPU_CHECK  1 for CPU-only image builders (training still needs GPU)
EOF
    exit 0
fi
if [[ $# != 0 ]]; then echo "Unknown arguments; use --help" >&2; exit 2; fi
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    echo "Install Anaconda/conda first; see distill/snapflow/README.md" >&2; exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This pinned ROCm recipe requires Linux x86_64" >&2; exit 1
fi
if [[ ! "${GPU_ARCH}" =~ ^gfx[0-9a-z]+$ ]]; then echo "Invalid GPU architecture" >&2; exit 2; fi
CONDA_BASE_PREFIX="$("${CONDA_BIN}" info --base)"
if [[ "${TARGET_ENV}" == base ]] || { [[ -n "${SNAPFLOW_CONDA_PREFIX:-}" ]] &&
    [[ "$(realpath -m "${SNAPFLOW_CONDA_PREFIX}")" == "$(realpath -m "${CONDA_BASE_PREFIX}")" ]]; }; then
    echo "Use a dedicated environment, not base" >&2; exit 2
fi
if [[ "${SKIP_GPU}" != 0 && "${SKIP_GPU}" != 1 ]]; then
    echo "SNAPFLOW_SKIP_GPU_CHECK must be 0 or 1" >&2; exit 2
fi
if [[ -n "${SNAPFLOW_CONDA_PREFIX:-}" ]]; then
    TARGET_ARGS=(-p "${SNAPFLOW_CONDA_PREFIX}")
else
    TARGET_ARGS=(-n "${TARGET_ENV}")
fi
if ! "${CONDA_BIN}" run "${TARGET_ARGS[@]}" python -c 'import sys' >/dev/null 2>&1; then
    "${CONDA_BIN}" create -y "${TARGET_ARGS[@]}" --no-default-packages \
        --override-channels -c conda-forge python=3.12 pip
fi
"${CONDA_BIN}" run --no-capture-output "${TARGET_ARGS[@]}" python - \
    "${SCRIPT_DIR}" "${GPU_ARCH}" "${ROCM_INDEX}" "${PYPI_INDEX}" "${SKIP_GPU}" <<'PY'
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

script, arch, rocm_index, pypi_index, skip_gpu = sys.argv[1:]
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Target environment must use Python 3.12; choose a new environment name")
lock = json.loads((Path(script) / "rocm_stack.json").read_text())
pip = [sys.executable, "-m", "pip", "--isolated", "install", "--index-url", pypi_index]
with tempfile.TemporaryDirectory(prefix="snapflow-setup-") as tmp:
    constraints = Path(tmp) / "rocm-constraints.txt"
    constraints.write_text("\n".join(f"{n}=={v}" for n, v in lock["packages"].items()) + "\n")
    # Explicit extras install the GPU-specific torch/vision payloads and ROCm SDK.
    # The exact versions avoid silently substituting CUDA wheels or newer nightlies.
    subprocess.run(pip + ["--extra-index-url", rocm_index, "--pre", "-c", str(constraints),
                        f"torch[device-{arch}]=={lock['packages']['torch']}",
                        f"torchvision[device-{arch}]=={lock['packages']['torchvision']}",
                        f"rocm-bootstrap=={lock['packages']['rocm-bootstrap']}"], check=True)
    # LeRobot's upstream torch bounds exclude the deliberately selected nightly.
    subprocess.run(pip + ["--no-deps", "lerobot==0.6.1"], check=True)
    subprocess.run(pip + ["-c", str(constraints), "-r", str(Path(script) / "requirements.txt")], check=True)
    import importlib.metadata as metadata
    actual = {n: metadata.version(n) for n in lock["packages"]}
    if actual != lock["packages"]:
        raise SystemExit(f"Unexpected ROCm stack: {actual}")
    # Execute in a fresh process so no pre-install modules can mask a broken install.
    subprocess.run([sys.executable, str(Path(script) / "check_environment.py"),
                    "--skip-gpu" if skip_gpu == "1" else "--gpu", "--arch", arch], check=True)
    prefix = Path(sys.prefix)
    (prefix / "snapflow_environment.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "arch": arch,
        "stack": actual, "gpu_check_requested": skip_gpu != "1",
        "rocm_index": rocm_index, "pypi_index": pypi_index,
    }, indent=2) + "\n")
    with (prefix / "snapflow_pip_freeze.txt").open("w") as output:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=output, check=True)
print(f"Installed in {sys.prefix}")
PY
if [[ -n "${SNAPFLOW_CONDA_PREFIX:-}" ]]; then
    echo "Ready: conda activate ${SNAPFLOW_CONDA_PREFIX}"
else
    echo "Ready: conda activate ${TARGET_ENV}"
fi
