"""SnapFlow 1-NFE student support for LeRobot PI05Pytorch (eval-only).

The distilled checkpoint has extra ``target_time_embed_mlp`` weights that
stock PI05Policy.from_pretrained rejects (strict load). Enable the MLP
before load, then run a single denoise at target_time=1.

Adapted from Reflex VLA SnapFlowPI05Pytorch (Apache-2.0).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SNAPFLOW_PI05_CLASS: type | None = None
_PATCHED = False


def checkpoint_has_snapflow_mlp(path) -> bool:
    from pathlib import Path

    sf = Path(path).expanduser() / "model.safetensors"
    if not sf.is_file():
        return False
    from safetensors import safe_open

    with safe_open(str(sf), framework="pt") as f:
        return any(k.startswith("model.target_time_embed_mlp.") for k in f.keys())


def enable_snapflow_pi05(model: Any) -> None:
    import torch
    import torch.nn as nn
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    if not isinstance(model, PI05Pytorch):
        raise TypeError(f"enable_snapflow_pi05 expects PI05Pytorch, got {type(model)}")
    if hasattr(model, "target_time_embed_mlp"):
        model.__class__ = _resolve_snapflow_pi05_class()
        return

    dim = model.action_in_proj.out_features
    hidden_dim = max(dim, 256)
    mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
    with torch.no_grad():
        mlp[-1].weight.zero_()
        mlp[-1].bias.zero_()
    ref = next(model.parameters())
    mlp = mlp.to(dtype=ref.dtype, device=ref.device)
    model.add_module("target_time_embed_mlp", mlp)
    model.__class__ = _resolve_snapflow_pi05_class()
    logger.info("snapflow: attached target_time_embed_mlp dim=%d hidden=%d", dim, hidden_dim)


def _resolve_snapflow_pi05_class() -> type:
    global _SNAPFLOW_PI05_CLASS
    if _SNAPFLOW_PI05_CLASS is not None:
        return _SNAPFLOW_PI05_CLASS

    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi05.modeling_pi05 import (
        PI05Pytorch,
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )
    from pi05_fast.rocm_pi05_optim import clone_kv_cache

    class SnapFlowPI05Pytorch(PI05Pytorch):
        def embed_suffix(self, noisy_actions, timestep, target_time=None):
            embs = []
            pad_masks = []
            att_masks = []

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=timestep.device,
            )
            suffix_dtype = self.time_mlp_in.weight.dtype
            time_emb = time_emb.to(suffix_dtype)

            if target_time is not None:
                tt = target_time
                if tt.ndim == 0:
                    tt = tt.expand(time_emb.shape[0])
                tt_emb = create_sinusoidal_pos_embedding(
                    tt,
                    self.action_in_proj.out_features,
                    min_period=self.config.min_period,
                    max_period=self.config.max_period,
                    device=tt.device,
                )
                mlp_dtype = self.target_time_embed_mlp[0].weight.dtype
                mlp_out = self.target_time_embed_mlp(tt_emb.to(mlp_dtype))
                time_emb = time_emb + mlp_out.to(suffix_dtype)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            adarms_cond = time_emb
            embs.append(action_emb)
            bsize, action_time_dim = action_emb.shape[:2]
            pad_masks.append(
                torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
            )
            att_masks += [1] + ([0] * (self.config.chunk_size - 1))
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))
            return embs, pad_masks, att_masks, adarms_cond

        def denoise_step(
            self,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            target_time=None,
        ):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
            expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
            past_key_values = clone_kv_cache(past_key_values, dtype=expert_dtype)
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(self, images, img_masks, tokens, masks, noise=None):
            bsize = tokens.shape[0]
            device = tokens.device
            if noise is None:
                noise = self.sample_noise(
                    (bsize, self.config.chunk_size, self.config.max_action_dim), device
                )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
                "eager"  # noqa: SLF001
            )
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            time = torch.ones(bsize, dtype=torch.float32, device=device)
            x_t = noise.to(self.action_in_proj.weight.dtype)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
            )
            return (x_t - v_t).to(noise.dtype)

        @torch.no_grad()
        def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
            return self.sample_actions_1step(images, img_masks, tokens, masks, noise=noise)

    _SNAPFLOW_PI05_CLASS = SnapFlowPI05Pytorch
    return SnapFlowPI05Pytorch


def install_pi05_snapflow_eval_hooks() -> None:
    """Attach SnapFlow MLP in PI05Policy.__init__ so from_pretrained can load student weights."""
    global _PATCHED
    if _PATCHED:
        return
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    orig_init = PI05Policy.__init__

    def wrapped_init(self, config, **kwargs):
        orig_init(self, config, **kwargs)
        enable_snapflow_pi05(self.model)

    PI05Policy.__init__ = wrapped_init
    _PATCHED = True
    print("snapflow: PI05Policy.__init__ patched (target_time 1-NFE)", flush=True)
