"""Low-peak PI05Policy loader.

Stock PI05Policy.from_pretrained builds the full model in fp32 with random init, moves it to
the GPU (8.7 GB) and then reads the whole checkpoint into host memory. This loader builds the
model with parameters on the meta device (buffers stay real), swaps in the W4A4 prefix-LM
linears and drops the unused lm_heads before any weight is read, then copies the remaining
weights tensor by tensor from the safetensors file straight to the GPU. Parameter dtypes and
values match the stock path, so the policy is bit-identical.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn


@contextlib.contextmanager
def _meta_parameters():
    """New parameters go to the meta device; Module.to keeps dtype changes but ignores devices."""
    orig_register = nn.Module.register_parameter
    orig_to = nn.Module.to

    def register_parameter(self, name, param):
        orig_register(self, name, param)
        p = self._parameters.get(name)
        if p is not None and not p.is_meta:
            self._parameters[name] = type(p)(p.to("meta"), requires_grad=p.requires_grad)

    def to_dtype_only(self, *args, **kwargs):
        _device, dtype, non_blocking, _fmt = torch._C._nn._parse_to(*args, **kwargs)
        return orig_to(self, dtype=dtype, non_blocking=non_blocking) if dtype is not None else self

    nn.Module.register_parameter = register_parameter
    nn.Module.to = to_dtype_only
    try:
        yield
    finally:
        nn.Module.register_parameter = orig_register
        nn.Module.to = orig_to


class _Key:
    """Stands in for a checkpoint tensor while LeRobot remaps keys (it only renames and clones)."""

    def __init__(self, key: str):
        self.key = key

    def clone(self):
        return self


def _materialize(module: nn.Module, device: torch.device) -> None:
    new: dict[int, nn.Parameter] = {}
    for mod in module.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None and p.is_meta:
                if id(p) not in new:  # keep tied parameters tied
                    new[id(p)] = nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device=device),
                                              requires_grad=p.requires_grad)
                mod._parameters[name] = new[id(p)]


def fast_from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = True, **kwargs):
    from safetensors import safe_open
    from transformers.utils import cached_file

    from lerobot.policies.pi05 import modeling_pi05 as m
    from pi05_fast.rocm_pi05_optim import drop_unused_lm_heads
    from pi05_fast.w4a4.apply import apply_w4a4_from_dir

    hub = {k: kwargs.pop(k) for k in ("force_download", "resume_download", "proxies", "token", "cache_dir",
                                      "local_files_only", "revision") if k in kwargs}
    if config is None:
        config = m.PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_name_or_path, **hub, **kwargs)
    with _meta_parameters():
        policy = cls(config, **kwargs)
    model = policy.model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if os.environ.get("PI05_DROP_LM_HEAD", "1") != "0":
        drop_unused_lm_heads(model)
    pack = os.environ.get("PI05_W4A4_PACK", "").strip()
    if pack:
        apply_w4a4_from_dir(model, pack, device=device)
    _materialize(policy, device)

    resolved = cached_file(str(pretrained_name_or_path), "model.safetensors",
                           **{k: v for k, v in hub.items() if k != "resume_download"})
    targets = dict(policy.named_parameters(remove_duplicate=False))
    targets.update(policy.named_buffers(remove_duplicate=False))
    # checkpoint keys of the swapped/dropped modules are expected to be unused
    swapped = tuple(n.removesuffix("packed") for n in targets if n.endswith(".packed"))
    loaded, unexpected = set(), []
    loaded_ids = set()
    with safe_open(resolved, framework="pt", device="cpu") as f:
        fixed = policy._fix_pytorch_state_dict_keys({k: _Key(k) for k in f.keys()}, policy.config)
        for key, src in fixed.items():
            name = key if key.startswith("model.") else f"model.{key}"
            if name.startswith(swapped):
                continue
            t = targets.get(name)
            if t is None:
                unexpected.append(name)
                continue
            with torch.no_grad():
                t.copy_(f.get_tensor(src.key))
            loaded.add(name)
            loaded_ids.add(id(t))
    unexpected = [k for k in unexpected if ".lm_head." not in k]  # dropped heads
    missing = [n for n, p in policy.named_parameters() if id(p) not in loaded_ids]
    if strict and (missing or unexpected):
        raise RuntimeError(f"fast load: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} "
                           f"({len(unexpected)})")
    policy.to(device)
    print(f"fast-load: {len(loaded)} tensors from {Path(resolved).name} -> {device} "
          f"(meta init, W4A4 {'on' if pack else 'off'})", flush=True)
    return policy
