#!/usr/bin/env bash
# Download SnapFlow weights, tokenizer, and (for eval) clone LIBERO.
#
# Default roots:
#   MODEL_DATA=~/model_data
#   LIBERO_ROOT=~/model_data/libero   (or reuse ~/openpi/third_party/libero if present)
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DATA="${MODEL_DATA:-$HOME/model_data}"
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"

resolve_libero_root() {
  if [[ -n "${LIBERO_ROOT:-}" ]]; then
    printf '%s' "${LIBERO_ROOT/#\~/$HOME}"
    return
  fi
  local d
  for d in \
    "${HOME}/openpi/third_party/libero" \
    "${MODEL_DATA}/libero" \
    "${PKG_ROOT}/.vendor/libero"; do
    if [[ -d "${d}/libero" ]]; then
      printf '%s' "$d"
      return
    fi
  done
  printf '%s' "${MODEL_DATA}/libero"
}

download_v044() {
  local dir="$MODEL_DATA/pi05_libero_v044"
  if [[ -f "$dir/model.safetensors" && -f "$dir/policy_preprocessor.json" ]]; then
    echo "skip v044 (already at $dir)"
    return
  fi
  mkdir -p "$dir"
  echo "=== lerobot/pi05_libero_finetuned_v044 → $dir"
  hf download lerobot/pi05_libero_finetuned_v044 \
    config.json model.safetensors \
    policy_preprocessor.json policy_preprocessor_step_2_normalizer_processor.safetensors \
    policy_postprocessor.json policy_postprocessor_step_0_unnormalizer_processor.safetensors \
    --local-dir "$dir"
}

sanitize_snapflow() {
  local dir="$MODEL_DATA/pi05_snapflow_1nfe"
  local teacher="$MODEL_DATA/pi05_libero_v044"
  [[ -f "$dir/config.json" ]] || return 0
  PYTHONPATH="${PKG_ROOT}${PYTHONPATH:+:$PYTHONPATH}" python3 - <<PY
from pathlib import Path
from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint
notes = sanitize_snapflow_checkpoint(Path("$dir"), Path("$teacher"))
for n in notes:
    print("snapflow ckpt:", n)
if not notes:
    print("snapflow ckpt: already LeRobot-compatible ($dir)")
PY
}

download_snapflow() {
  local dir="$MODEL_DATA/pi05_snapflow_1nfe"
  if [[ -f "$dir/model.safetensors" && -f "$dir/config.json" ]]; then
    echo "skip snapflow download (already at $dir)"
    sanitize_snapflow
    return
  fi
  mkdir -p "$dir"
  echo "=== Rylinjames/pi05-snapflow-distill-1nfe → $dir"
  hf download Rylinjames/pi05-snapflow-distill-1nfe \
    config.json model.safetensors distill_provenance.json \
    --local-dir "$dir"
  sanitize_snapflow
}

download_tokenizer() {
  local dir="${PALIGEMMA_TOKENIZER_PATH:-$MODEL_DATA/paligemma2-3b-pt-224}"
  if [[ -f "$dir/tokenizer.json" || -f "$dir/tokenizer.model" ]]; then
    echo "skip tokenizer (already at $dir)"
    return
  fi
  mkdir -p "$dir"
  echo "=== google/paligemma2-3b-pt-224 tokenizer → $dir"
  hf download google/paligemma2-3b-pt-224 --local-dir "$dir"
}

clone_libero() {
  local dest
  dest="$(resolve_libero_root)"
  if [[ -d "$dest/libero" ]]; then
    echo "skip LIBERO (already at $dest)"
    echo "LIBERO_ROOT=$dest"
    return
  fi
  command -v git >/dev/null 2>&1 || { echo "ERROR: git not found (needed to clone LIBERO)"; exit 1; }
  echo "=== clone LIBERO → $dest"
  mkdir -p "$(dirname "$dest")"
  git clone --depth 1 "${LIBERO_REPO}" "$dest"
  echo "LIBERO_ROOT=$dest"
}

need_hf() {
  command -v hf >/dev/null 2>&1 || { echo "ERROR: install huggingface_hub CLI (hf)"; exit 1; }
}

TARGET="${1:-all}"
case "$TARGET" in
  v044|libero-weights)
    need_hf
    download_v044
    ;;
  snapflow)
    need_hf
    download_snapflow
    ;;
  tokenizer)
    need_hf
    download_tokenizer
    ;;
  libero|libero-src)
    clone_libero
    ;;
  all)
    clone_libero
    need_hf
    download_v044
    download_snapflow
    download_tokenizer
    ;;
  *)
    echo "Usage: $0 [all|v044|snapflow|tokenizer|libero]" >&2
    exit 1
    ;;
esac
echo "Done."
