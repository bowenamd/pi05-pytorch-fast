"""Behavior gates for the extracted component; no model downloads."""
from types import SimpleNamespace

import pytest
import torch

from distill.snapflow.loss import snapflow_loss_step
from distill.snapflow.model import _resolve_snapflow_pi05_class
from distill.snapflow.runtime import build_dataloader, iter_batches
from distill.snapflow.train import summarize


def test_loss_matches_shortcut_and_only_student_receives_gradients():
    student_weight = torch.nn.Parameter(torch.tensor(0.3))
    teacher_weight = torch.nn.Parameter(torch.tensor(0.7))
    action = torch.ones(2, 3, 4)
    noise = torch.zeros_like(action)
    t = torch.tensor([0.2, 0.6])
    target_times = []

    def student(x, t, target_time=None, **obs):
        target_times.append(target_time)
        return student_weight * x

    def teacher(x, t, **obs):
        return teacher_weight * x

    loss, row = snapflow_loss_step(student, teacher, action=action, noise=noise,
                                  t=t, obs_kwargs={}, consistency_alpha=1.0)
    x = t[:, None, None].expand_as(action)
    v0 = 0.7 * x
    vmid = 0.7 * (x + 0.5 * (1 - t[:, None, None]) * v0)
    expected = ((0.3 * x - 1) ** 2).mean() + ((0.3 * x - (v0 + vmid) / 2) ** 2).mean()
    torch.testing.assert_close(loss, expected)
    assert row.total == pytest.approx(float(expected))
    assert target_times[0] is None
    torch.testing.assert_close(target_times[1], torch.ones_like(t))
    loss.backward()
    assert student_weight.grad is not None and torch.isfinite(student_weight.grad)
    assert teacher_weight.grad is None


def test_bf16_time_projection_accepts_fp32_sampling_timestep():
    model = SimpleNamespace(
        action_in_proj=torch.nn.Linear(4, 8).to(torch.bfloat16),
        time_mlp_in=torch.nn.Linear(8, 8).to(torch.bfloat16),
        time_mlp_out=torch.nn.Linear(8, 8).to(torch.bfloat16),
        target_time_embed_mlp=torch.nn.Sequential(torch.nn.Linear(8, 8)).to(torch.bfloat16),
        config=SimpleNamespace(min_period=0.004, max_period=4.0, chunk_size=3),
        _apply_checkpoint=lambda fn, *args: fn(*args),
    )
    out = _resolve_snapflow_pi05_class().embed_suffix(
        model, torch.randn(1, 3, 4, dtype=torch.bfloat16),
        torch.ones(1, dtype=torch.float32), target_time=torch.ones(1, dtype=torch.float32),
    )
    assert out[0].dtype == out[3].dtype == torch.bfloat16
    assert torch.isfinite(out[3]).all()


def test_epoch_restart_reaches_exact_step_count():
    assert list(iter_batches(["a", "b"], 5)) == [(1, "a"), (2, "b"), (3, "a"), (4, "b"), (5, "a")]
    with pytest.raises(RuntimeError, match="Empty training epoch"):
        list(iter_batches([], 1))


def test_missing_local_dataset_fails_before_loading(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("Missing local metadata must not trigger download")
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", unexpected)
    with pytest.raises(FileNotFoundError, match="metadata missing"):
        build_dataloader(tmp_path, repo_id="lerobot/libero", revision="pinned",
                         chunk_size=50, batch_size=4, num_workers=0, seed=42)


def test_local_dataset_uses_revision_and_real_future_actions(monkeypatch, tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    calls = []

    def meta(repo, **kwargs):
        calls.append((repo, kwargs))
        return SimpleNamespace(fps=10)

    def dataset(repo, **kwargs):
        calls.append((repo, kwargs))
        return [{"action": torch.zeros(50, 7)}]

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", meta)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", dataset)
    loader = build_dataloader(tmp_path, repo_id="lerobot/libero", revision="pinned",
                              chunk_size=50, batch_size=1, num_workers=0, seed=42)
    assert next(iter(loader))["action"].shape == (1, 50, 7)
    assert calls[0][1] == {"root": tmp_path, "revision": "pinned"}
    assert calls[1][1]["video_backend"] == "pyav"
    assert calls[1][1]["delta_timestamps"]["action"][-1] == 4.9


def test_loss_summary_uses_windows_and_detects_nonfinite_values():
    rows = [{"total": v, "flow_matching": v, "consistency": 0.0} for v in [4.0, 3.0, 2.0, 1.0]]
    summary = summarize(rows)
    assert summary["early_mean_loss"] == 3.5
    assert summary["late_mean_loss"] == 1.5
    assert summary["loss_decreased"] is True
    assert summarize(list(reversed(rows)))["loss_decreased"] is False
    rows[-1]["total"] = float("nan")
    assert summarize(rows)["all_losses_finite"] is False
