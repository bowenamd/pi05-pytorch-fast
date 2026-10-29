# Derived from FastCrest Tether. Copyright (c) 2026 FastCrest.
# SPDX-License-Identifier: BUSL-1.1
# See LICENSE and NOTICE.md in this directory.

"""pi0.5 target-time conditioning and detached-prefix denoising."""
from __future__ import annotations
from typing import Any
_SNAPFLOW_PI05_CLASS = None

def _detach_kv_cache_deep(past_kv: Any) -> Any:
    """Walk a KV cache and detach every tensor found, in place.

    Used by SnapFlowPI05Pytorch.denoise_step before a ``copy.deepcopy``
    that would otherwise raise on graph-attached tensors (pytorch#103001).
    Returns the same object with tensors detached.

    Handles:
      - bare tuples / lists / dicts
      - HF DynamicCache (key_cache + value_cache attrs)
      - any object with a __dict__ containing nested tensors
    """
    import torch

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        for attr in ("key_cache", "value_cache", "key_states", "value_states"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, _walk(getattr(obj, attr)))
                except AttributeError:
                    pass  # read-only property
        if hasattr(obj, "__dict__"):
            for k, v in list(obj.__dict__.items()):
                if isinstance(v, (torch.Tensor, list, tuple, dict)):
                    obj.__dict__[k] = _walk(v)
        return obj

    return _walk(past_kv)


def _resolve_snapflow_pi05_class() -> type:
    """Build (or return cached) SnapFlowPI05Pytorch subclass of lerobot's
    PI05Pytorch. Lazy so importing this module doesn't force lerobot import.

    Architectural differences vs pi0:
      - pi05 has NO state_proj. State is not in the suffix embedding.
      - pi05 uses AdaRMSNorm for time conditioning: time_emb flows through
        time_mlp_in → silu → time_mlp_out → silu, and the result becomes
        adarms_cond passed to the gemma_expert via the AdaRMS layer.
      - target_time injects ADDITIVELY to time_emb BEFORE the time_mlp
        (so the zero-init MLP starts identity-equivalent to teacher).
    """
    global _SNAPFLOW_PI05_CLASS
    if _SNAPFLOW_PI05_CLASS is not None:
        return _SNAPFLOW_PI05_CLASS

    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from lerobot.policies.common.vla_utils import (
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    class SnapFlowPI05Pytorch(PI05Pytorch):
        """PI05Pytorch + target_time embed_suffix + deepcopy-free denoise_step.

        Not constructed directly — produced by enable_snapflow() via
        __class__-swap on a loaded PI05Pytorch instance.
        """

        def embed_suffix(
            self,
            noisy_actions,
            timestep,
            target_time=None,
        ):
            """Parent embed_suffix + optional zero-init target_time contribution.

            target_time (when given) is sinusoidally encoded and routed
            through ``target_time_embed_mlp`` (zero-init), then ADDED to
            time_emb BEFORE the existing time_mlp_{in,out} chain. This way
            the AdaRMS conditioning vector (adarms_cond) carries the
            shortcut-velocity signal at target_time=1.
            """
            import torch.nn.functional as F

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
            time_emb = time_emb.type(dtype=timestep.dtype)

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
                # Cast to MLP's dtype (bf16 during training/inference),
                # not time_emb's dtype (which follows timestep, often fp32).
                # Otherwise bf16 model + fp32 timestep → F.linear dtype
                # mismatch at MLP's first Linear layer.
                mlp_dtype = self.target_time_embed_mlp[0].weight.dtype
                tt_emb = tt_emb.to(mlp_dtype)
                mlp_out = self.target_time_embed_mlp(tt_emb)
                time_emb = time_emb + mlp_out.to(time_emb.dtype)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

            def time_mlp_func(time_emb):
                # Sampling uses fp32 timesteps even when the learned time
                # projections are bf16. Match the actual linear weight dtype.
                time_emb = time_emb.to(self.time_mlp_in.weight.dtype)
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

            embs.append(action_time_emb)
            bsize, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(
                bsize, action_time_dim, dtype=torch.bool, device=timestep.device,
            )
            pad_masks.append(action_time_mask)

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
            state=None,  # accepted for API compat with pi0 path; ignored
        ):
            """Parent denoise_step minus the fp32 cast; with DETACHED past_kv.

            Unlike the pi0 override (which keeps past_kv graph-attached so
            VLM gradients flow to the student), pi0.5's
            ``paligemma_with_expert.forward`` needs ``past_key_values`` in a
            state the DynamicCache can inspect — in practice this means it
            must be re-parseable as a cache object, which requires going
            through a ``copy.deepcopy`` that fails on graph-attached
            tensors (pytorch#103001).

            Workaround: detach-in-place + deepcopy before the forward.
            Consequence: pi0.5 student trains action expert + projections +
            target_time_embed_mlp, but VLM weights stay frozen (same as
            pi0 in v0.3 pre-v0.3.1). Full-VLM training is v0.3.2 work.

            ``state`` kwarg is accepted for cross-family API compatibility
            but unused — pi0.5 doesn't put state in the suffix embedding.
            """
            import copy

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2,
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            # Detach past_kv tensors so deepcopy succeeds on graph-attached
            # cache objects. Walks the cache recursively.
            past_key_values = _detach_kv_cache_deep(past_key_values)
            past_key_values = copy.deepcopy(past_key_values)

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(
            self,
            images,
            img_masks,
            tokens,
            masks,
            noise=None,
        ):
            """SnapFlow 1-NFE inference for pi0.5 (no state argument).

            Mirror of PI05Pytorch.sample_actions but with num_steps=1 and
            target_time=1 threaded into the single denoise_step call.
            """
            bsize = tokens.shape[0]
            device = tokens.device

            if noise is None:
                actions_shape = (
                    bsize, self.config.chunk_size, self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            time = torch.ones(bsize, dtype=torch.float32, device=device)
            action_dtype = self.action_in_proj.weight.dtype
            x_t = noise.to(action_dtype)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
            )
            return (x_t - v_t).to(noise.dtype)

    _SNAPFLOW_PI05_CLASS = SnapFlowPI05Pytorch
    return SnapFlowPI05Pytorch


def enable_snapflow(model):
    """Attach the zero-initialized target-time MLP before training or weight loading."""
    import torch
    from torch import nn
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    if not isinstance(model, PI05Pytorch):
        raise TypeError(f"Expected PI05Pytorch, got {type(model).__name__}")
    if not hasattr(model, "target_time_embed_mlp"):
        dim = model.action_in_proj.out_features
        mlp = nn.Sequential(nn.Linear(dim, max(dim, 256)), nn.SiLU(),
                            nn.Linear(max(dim, 256), dim))
        with torch.no_grad():
            mlp[-1].weight.zero_()
            mlp[-1].bias.zero_()
        ref = model.action_in_proj.weight
        model.add_module("target_time_embed_mlp", mlp.to(device=ref.device, dtype=ref.dtype))
    model.__class__ = _resolve_snapflow_pi05_class()
