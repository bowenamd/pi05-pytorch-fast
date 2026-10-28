# Derived from FastCrest Tether. Copyright (c) 2026 FastCrest.
# SPDX-License-Identifier: BUSL-1.1
# See LICENSE and NOTICE.md in this directory.

"""SnapFlow loss, extracted without changing Tether's math convention."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
DEFAULT_CONSISTENCY_ALPHA = 1.0

@dataclass
class SnapFlowLosses:
    """Per-step loss breakdown. Backend logs these to the training log
    for observability + later VERIFICATION.md rendering."""

    flow_matching: float
    consistency: float
    total: float


def flow_matching_interp(
    noise: "torch.Tensor",
    action: "torch.Tensor",
    t: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Linear flow-matching interpolation.

    Returns (x_t, v_target) where:
      x_t      = (1-t) * noise + t * action
      v_target = action - noise     (the analytic velocity)

    Shapes:
      noise, action: (B, chunk, action_dim)
      t:             (B,) — random per-sample time
      x_t, v_target: (B, chunk, action_dim)

    `t` is broadcast along chunk + action_dim axes. Caller should have
    sampled t ~ Uniform(0, 1) per batch element.
    """
    import torch

    assert noise.shape == action.shape, f"noise {noise.shape} vs action {action.shape}"
    # Broadcast t to (B, 1, 1) so it multiplies per-sample
    while t.ndim < noise.ndim:
        t = t.unsqueeze(-1)
    x_t = (1.0 - t) * noise + t * action
    v_target = action - noise  # independent of t (linear interp)
    return x_t, v_target


def two_step_euler_shortcut(
    teacher_velocity_fn: Callable[..., "torch.Tensor"],
    x_t: "torch.Tensor",
    t: "torch.Tensor",
    *,
    obs_kwargs: dict,
) -> "torch.Tensor":
    """Compute SnapFlow's 2-step Euler shortcut velocity from the teacher.

    The shortcut estimates "what one-step velocity moves x_t to action"
    using TWO small Euler steps with the teacher velocity field:

        v_t         = teacher(x_t,   t)
        x_mid       = x_t + 0.5*(1-t) * v_t                         # Euler half-step
        v_mid       = teacher(x_mid, t + 0.5*(1-t))
        v_shortcut  = 0.5 * (v_t + v_mid)                           # avg

    Expects `teacher_velocity_fn(x, t, **obs_kwargs) -> velocity`.
    Runs under `torch.no_grad()` inside — caller doesn't need to wrap.

    Returns a tensor shaped like x_t.
    """
    import torch

    with torch.no_grad():
        v_t = teacher_velocity_fn(x_t, t, **obs_kwargs)
        # Broadcast t to match v_t dims for the half-step addition
        t_broadcast = t
        while t_broadcast.ndim < x_t.ndim:
            t_broadcast = t_broadcast.unsqueeze(-1)
        half_step = 0.5 * (1.0 - t_broadcast)
        x_mid = x_t + half_step * v_t
        t_mid = t + 0.5 * (1.0 - t)
        v_mid = teacher_velocity_fn(x_mid, t_mid, **obs_kwargs)
        v_shortcut = 0.5 * (v_t + v_mid)
    return v_shortcut


def snapflow_loss_step(
    student_velocity_fn: Callable[..., "torch.Tensor"],
    teacher_velocity_fn: Callable[..., "torch.Tensor"],
    *,
    action: "torch.Tensor",
    noise: "torch.Tensor",
    t: "torch.Tensor",
    obs_kwargs: dict,
    teacher_obs_kwargs: dict | None = None,
    consistency_alpha: float = DEFAULT_CONSISTENCY_ALPHA,
) -> tuple["torch.Tensor", SnapFlowLosses]:
    """Compute one SnapFlow training-step loss.

    Args:
      student_velocity_fn: callable `(x, t, target_time=None, **obs_kwargs) -> v`.
        When `target_time=None`, behaves as standard velocity prediction.
        When `target_time=1.0`, produces the one-step generation output.
      teacher_velocity_fn: callable `(x, t, **obs_kwargs) -> v` (frozen).
      action: ground-truth action chunk (B, chunk, action_dim).
      noise: Gaussian noise, same shape as action.
      t: random time per sample (B,), sampled Uniform(0, 1).
      obs_kwargs: VLM conditioning inputs for the STUDENT (image, state,
        language). For the default variant, used for both student and
        teacher. For v0.5 'state_out' variant, this is the student-side
        dict with stripped lang + explicit state; teacher gets its own
        via ``teacher_obs_kwargs``.
      teacher_obs_kwargs: optional separate conditioning for the teacher.
        When None (default), teacher reuses ``obs_kwargs``. Set this when
        student and teacher have different preprocessed forms (v0.5
        state-out student).
      consistency_alpha: mix weight for the consistency term.

    Returns (loss_tensor, SnapFlowLosses snapshot). Caller does
    loss.backward() and optimizer.step().
    """
    import torch
    import torch.nn.functional as F

    t_obs = teacher_obs_kwargs if teacher_obs_kwargs is not None else obs_kwargs

    # (a) Flow-matching term: student must match the analytic velocity
    # at the random interpolation point.
    x_t, v_target = flow_matching_interp(noise, action, t)
    v_student = student_velocity_fn(x_t, t, target_time=None, **obs_kwargs)
    fm_loss = F.mse_loss(v_student, v_target)

    # (b) Consistency term: student at target_time=1 should match the
    # teacher's 2-step Euler shortcut from x_t.
    v_shortcut = two_step_euler_shortcut(
        teacher_velocity_fn, x_t, t, obs_kwargs=t_obs,
    )
    target_time_one = torch.ones_like(t)
    v_student_one = student_velocity_fn(
        x_t, t, target_time=target_time_one, **obs_kwargs,
    )
    consistency_loss = F.mse_loss(v_student_one, v_shortcut)

    total = fm_loss + consistency_alpha * consistency_loss
    snapshot = SnapFlowLosses(
        flow_matching=float(fm_loss.item()),
        consistency=float(consistency_loss.item()),
        total=float(total.item()),
    )
    return total, snapshot
