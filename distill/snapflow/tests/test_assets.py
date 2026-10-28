"""Integrity and offline-reproduction gates for the asset downloader."""
from types import SimpleNamespace

import pytest

from distill.snapflow.download_assets import SOURCES, checked_path, checksum, fetch_inventory, run, verify_files

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hashes_match_known_sha256_and_git_blob_vectors(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert checksum(empty, "git-sha1") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    payload = tmp_path / "payload"
    payload.write_bytes(b"abc")
    assert checksum(payload, "sha256") == ABC_SHA256


def test_same_size_corruption_and_missing_files_are_rejected(tmp_path):
    entries = [{"path": "model.safetensors", "bytes": 3,
                "algorithm": "sha256", "checksum": ABC_SHA256}]
    with pytest.raises(RuntimeError, match="Missing or wrong-sized"):
        verify_files(tmp_path, entries)
    (tmp_path / "model.safetensors").write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_files(tmp_path, entries)
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    verify_files(tmp_path, entries)


def test_asset_paths_cannot_escape_download_root(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        checked_path(root, "../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        checked_path(root, "/absolute")
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        checked_path(root, "outside/file")


def install_fake_hub(monkeypatch, *, corrupt=False):
    calls = []

    class API:
        def __init__(self, token):
            assert token is False

        def repo_info(self, repo, **kwargs):
            calls.append((repo, kwargs))
            return SimpleNamespace(siblings=[SimpleNamespace(
                rfilename="model.safetensors", size=3, blob_id=None,
                lfs=SimpleNamespace(sha256=ABC_SHA256),
            )])

    def download(repo, **kwargs):
        assert kwargs["token"] is False
        root = kwargs["local_dir"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "model.safetensors").write_bytes(b"abd" if corrupt else b"abc")

    monkeypatch.setattr("huggingface_hub.HfApi", API)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    return calls


def test_download_pins_source_and_offline_verification_needs_no_hub(monkeypatch, tmp_path):
    calls = install_fake_hub(monkeypatch)
    manifest = run(tmp_path, only="teacher")
    assert calls[0][1]["revision"] == SOURCES["teacher"]["revision"]
    assert manifest["assets"]["teacher"]["verified"] is True
    assert (tmp_path / "assets_manifest.json").is_file()

    def fail(*args, **kwargs):
        raise AssertionError("Offline verification must not call the Hub")

    monkeypatch.setattr("huggingface_hub.HfApi", fail)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    run(tmp_path, only="teacher", verify_only=True)
    (tmp_path / SOURCES["teacher"]["directory"] / "model.safetensors").write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        run(tmp_path, only="teacher", verify_only=True)


def test_failed_download_does_not_create_verified_manifest(monkeypatch, tmp_path):
    install_fake_hub(monkeypatch, corrupt=True)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        run(tmp_path, only="teacher")
    assert not (tmp_path / "assets_manifest.json").exists()


def test_transient_hub_failure_retries_without_changing_revision(monkeypatch):
    import httpx

    calls = []
    def request(repo, **kwargs):
        calls.append(kwargs["revision"])
        if len(calls) == 1:
            raise httpx.ConnectError("temporary TLS failure")
        return "inventory"
    monkeypatch.setattr("distill.snapflow.download_assets.time.sleep", lambda seconds: None)
    result = fetch_inventory(SimpleNamespace(repo_info=request), SOURCES["teacher"])
    assert result == "inventory"
    assert calls == [SOURCES["teacher"]["revision"]] * 2
