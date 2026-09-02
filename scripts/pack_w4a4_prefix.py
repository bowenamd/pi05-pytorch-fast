#!/usr/bin/env python3
"""Pack PaliGemma language-model Linear weights for W4A4 (no full policy load)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _weight_to_module_name(key: str) -> str | None:
    name = key
    if name.startswith("model."):
        name = name[len("model.") :]
    if not name.endswith(".weight"):
        return None
    name = name[: -len(".weight")]
    prefix = "paligemma_with_expert.paligemma.model.language_model.layers."
    if not name.startswith(prefix):
        return None
    if name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate_proj", ".up_proj", ".down_proj")):
        return name
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path.home() / "model_data" / "pi05_snapflow_1nfe")
    p.add_argument("--out", type=Path, default=Path("models/w4a4_prefix"))
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pi05_fast.w4a4.pack import pack_linear_weight, should_rotate

    st = args.model.expanduser() / "model.safetensors"
    if not st.is_file():
        sys.exit(f"missing {st}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}
    with safe_open(str(st), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            mod = _weight_to_module_name(key)
            if mod is None:
                continue
            w = f.get_tensor(key)
            n, k = w.shape
            if k % 64 != 0:
                print(f"skip {mod}: K={k} not multiple of 64")
                continue
            rotate = should_rotate(k)
            packed = pack_linear_weight(w, rotate=rotate)
            tensors[f"{mod}.packed"] = packed.contiguous()
            bias_key = key[: -len("weight")] + "bias"
            has_bias = bias_key in keys
            if has_bias:
                tensors[f"{mod}.bias"] = f.get_tensor(bias_key).contiguous()
            layers[mod] = {
                "in_features": k,
                "out_features": n,
                "rotate": rotate,
                "has_bias": has_bias,
            }
            print(f"packed {mod}  [{n}, {k}] rotate={rotate}")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "packed.safetensors"))
    meta = {"format": "embedl-int4-gemm-v1", "source": str(args.model.expanduser()), "layers": layers}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(layers)} layers → {args.out}")


if __name__ == "__main__":
    main()
