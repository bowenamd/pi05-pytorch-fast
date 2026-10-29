#!/usr/bin/env bash
# Run after setup_conda.sh and conda activation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
python -m distill.snapflow.download_assets "$@"
