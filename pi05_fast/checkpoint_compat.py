"""One-time SnapFlow checkpoint fixes for stock LeRobot PI05Config.

Idempotent. Mutates ``config.json`` in the checkpoint directory (not a per-run copy).
"""

from __future__ import annotations

import json
from pathlib import Path

_V044_PROCESSOR_FILES = (
    "policy_preprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def sanitize_snapflow_checkpoint(ckpt: Path, teacher: Path | None = None) -> list[str]:
    """Strip ``_reflex_*`` keys, disable LeRobot compile, optionally link processors.

    Returns human-readable notes (empty if nothing changed).
    """
    ckpt = ckpt.expanduser().resolve()
    notes: list[str] = []
    cfg_path = ckpt / "config.json"
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text())
        drop = [k for k in list(data) if str(k).startswith("_reflex_")]
        changed = False
        for k in drop:
            del data[k]
            changed = True
        if data.get("compile_model") is True:
            data["compile_model"] = False
            changed = True
        if changed:
            cfg_path.write_text(json.dumps(data, indent=2) + "\n")
            if drop:
                notes.append(f"stripped {drop} from {cfg_path}")
            notes.append(f"compile_model=false in {cfg_path}")

    if teacher is None:
        return notes
    teacher = teacher.expanduser()
    if not teacher.is_dir():
        return notes
    for name in _V044_PROCESSOR_FILES:
        dest = ckpt / name
        if dest.exists() or dest.is_symlink():
            continue
        src = teacher / name
        if src.is_file():
            dest.symlink_to(src.resolve())
            notes.append(f"linked {name} <- {src}")
    return notes
