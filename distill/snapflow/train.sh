#!/usr/bin/env bash
# Run inside the separately prepared conda environment.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
# Exclude unrelated inherited source trees from module lookup.
export PYTHONPATH="${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false
python -m distill.snapflow.train "$@"
