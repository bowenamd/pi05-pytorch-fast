# Derived from FastCrest Tether. Copyright (c) 2026 FastCrest.
# SPDX-License-Identifier: BUSL-1.1
# See LICENSE and NOTICE.md in this directory.
"""Local pi0.5 inputs, velocity adapter, and portable student checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import enable_snapflow


def load_policy(path: Path, *, student: bool = False):
    """Load local weights strictly, attaching the student MLP before loading."""
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors import safe_open
    from safetensors.torch import load_file, load_model

    path = path.expanduser().resolve()
    if json.loads((path / "config.json").read_text()).get("type") != "pi05":
        raise ValueError("Only pi05 checkpoints are supported")
    with safe_open(str(path / "model.safetensors"), framework="pt") as weights:
        has_mlp = any(k.startswith("model.target_time_embed_mlp.") for k in weights.keys())
    if has_mlp != student:
        raise ValueError("Teacher must be undistilled; student must contain target_time_embed_mlp")
    config = PI05Config.from_pretrained(str(path), local_files_only=True)
    config.device = "cpu"
    config.dtype = "bfloat16"
    config.compile_model = False
    config.gradient_checkpointing = False
    config.push_to_hub = False
    policy = PI05Policy(config)
    if student:
        enable_snapflow(policy.model)
    if student:
        # Native save_pretrained omits shared tensor aliases.
        load_model(policy, str(path / "model.safetensors"), strict=True, device="cpu")
    else:
        # Public teachers use LeRobot's OpenPI key remapping (including the
        # shared embedding/lm_head). Unlike its from_pretrained wrapper, errors
        # propagate instead of returning an accidentally random policy.
        weights = load_file(str(path / "model.safetensors"))
        weights = policy._fix_pytorch_state_dict_keys(weights, config)
        weights = {k if k.startswith("model.") else f"model.{k}": v for k, v in weights.items()}
        policy.load_state_dict(weights, strict=True)
    config.device = "cuda"
    return policy.to(device="cuda", dtype=torch.bfloat16)


def build_preprocessor(teacher: Path, *, tokenizer: Path | None = None):
    # Registers the state-in-language processor even before a policy is loaded.
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor.pipeline import DataProcessorPipeline

    overrides = {"device_processor": {"device": "cuda"}}
    if tokenizer is not None:
        tokenizer = tokenizer.expanduser().resolve()
        if not (tokenizer / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Local tokenizer missing: {tokenizer}")
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer)}
    return DataProcessorPipeline.from_pretrained(
        str(teacher), config_filename="policy_preprocessor.json", overrides=overrides,
    )


def build_dataloader(root: Path, *, repo_id: str, revision: str, chunk_size: int,
                     batch_size: int, num_workers: int, seed: int):
    """Use real future actions and locally downloaded frames; never tile actions."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = root.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Local dataset metadata missing: {root}")
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    dataset = LeRobotDataset(
        repo_id, root=root, revision=revision, video_backend="pyav",
        delta_timestamps={"action": [i / meta.fps for i in range(chunk_size)]},
    )
    if len(dataset) < batch_size:
        raise ValueError("Dataset has fewer frames than batch_size (drop_last=True)")
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=True, generator=torch.Generator().manual_seed(seed),
    )


def iter_batches(loader, steps: int):
    """Restart finite epochs without retaining decoded batches in memory."""
    step = 0
    while step < steps:
        nonempty = False
        for batch in loader:
            nonempty = True
            step += 1
            yield step, batch
            if step == steps:
                return
        if not nonempty:
            raise RuntimeError(f"Empty training epoch at {step}/{steps}")


def prepare_batch(batch: dict, *, chunk_size: int, max_action_dim: int):
    action = batch["action"].to(device="cuda", dtype=torch.bfloat16)
    if action.ndim != 3 or action.shape[1] != chunk_size:
        raise ValueError(f"Expected real (B, {chunk_size}, D) action chunks, got {action.shape}")
    if action.shape[-1] > max_action_dim:
        raise ValueError("Dataset action dimension exceeds policy max_action_dim")
    action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    noise = torch.randn_like(action)
    t = torch.rand(action.shape[0], device="cuda", dtype=torch.float32).to(action.dtype)
    obs = {k: v for k, v in batch.items() if k != "action"}
    return action, noise, t, obs


def velocity_fn(policy):
    """Mirror Tether's pi05 adapter; prefix KV stays detached from training."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    def velocity(x, t, target_time=None, **obs):
        model = policy.model
        # Default SnapFlow freezes this entire prefix. Explicit no_grad avoids
        # building discarded graphs, while expert and time MLP remain trainable.
        with torch.no_grad():
            images, image_masks = policy._preprocess_images(obs)
            embs, pad, att = model.embed_prefix(
                images, image_masks, obs[OBS_LANGUAGE_TOKENS], obs[OBS_LANGUAGE_ATTENTION_MASK],
            )
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
            model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
            _, cache = model.paligemma_with_expert.forward(
                attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None, inputs_embeds=[embs, None], use_cache=True,
            )
        return model.denoise_step(
            prefix_pad_masks=pad, past_key_values=cache,
            x_t=x.to(model.action_in_proj.weight.dtype), timestep=t, target_time=target_time,
        )
    return velocity


def save_checkpoint(student, preprocessor, teacher: Path, output: Path, step: int,
                    provenance: dict) -> Path:
    from lerobot.processor.pipeline import DataProcessorPipeline

    checkpoint = output / "checkpoints" / f"{step:08d}" / "pretrained_model"
    student.config.num_inference_steps = 1
    student.save_pretrained(str(checkpoint))
    preprocessor.save_pretrained(checkpoint)
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(teacher), config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    postprocessor.save_pretrained(checkpoint)
    (checkpoint / "distill_provenance.json").write_text(
        json.dumps({**provenance, "completed_steps": step}, indent=2) + "\n",
    )
    return checkpoint
