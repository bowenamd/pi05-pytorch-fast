"""Download and verify pinned undistilled teacher, LIBERO data, and tokenizer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

SOURCES = {
    'teacher': {
        'repo': 'lerobot/pi05_libero_finetuned_v044', 'repo_type': 'model',
        'revision': '7152f13e34c6c115d808aea0e5d08aca74eac14b',
        'directory': 'checkpoints/pi05_libero_finetuned_v044',
    },
    'dataset': {
        'repo': 'lerobot/libero', 'repo_type': 'dataset',
        'revision': 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',
        'directory': 'datasets/libero',
    },
    'tokenizer': {
        'repo': 'leo009/paligemma-3b-pt-224', 'repo_type': 'model',
        'revision': '39996beb6fb17c5d16a50d3ef8f7a96ad9d03986',
        'directory': 'tokenizers/paligemma-3b-pt-224',
    },
}
# The public tokenizer copy is byte-identical to the pinned Google files used
# by the source experiment. Availability does not change the original terms.
TOKENIZER_HASHES = {
    'added_tokens.json': ('git-sha1', 'ff9246e420971b10c510b39a0de289132c2bf23f'),
    'special_tokens_map.json': ('git-sha1', '0c18fdd94629ea652e2ef36ed9a422474ad0a8e8'),
    'tokenizer_config.json': ('git-sha1', '960dcec0bccc6e594d865eceb63c8078e164cfb2'),
    'tokenizer.json': ('sha256', 'ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563'),
    'tokenizer.model': ('sha256', '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'),
}


def checked_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or not name.parts:
        raise ValueError(f'Unsafe asset path: {relative}')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes destination: {relative}')
    return path


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.sha1() if algorithm == 'git-sha1' else hashlib.new(algorithm)
    if algorithm == 'git-sha1':
        digest.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, files: list[dict]) -> None:
    for entry in files:
        path = checked_path(root, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Missing or wrong-sized asset: {path}')
        if checksum(path, entry['algorithm']) != entry['checksum']:
            raise RuntimeError(f'Checksum mismatch: {path}. Retry download with --force-download')


def expected_file(sibling) -> dict:
    if sibling.size is None:
        raise ValueError(f'Missing upstream size for {sibling.rfilename}')
    if sibling.lfs is not None:
        algorithm, expected = 'sha256', sibling.lfs.sha256
    else:
        algorithm, expected = 'git-sha1', sibling.blob_id
    if not expected:
        raise ValueError(f'Missing upstream checksum for {sibling.rfilename}')
    return {'path': sibling.rfilename, 'bytes': sibling.size,
            'algorithm': algorithm, 'checksum': expected}


def validate_tokenizer(root: Path) -> None:
    for name, (algorithm, expected) in TOKENIZER_HASHES.items():
        path = root / name
        if not path.is_file() or checksum(path, algorithm) != expected:
            raise RuntimeError(f'Tokenizer differs from the original reference: {name}')


def fetch_inventory(api, source: dict, *, retries: int = 3):
    """Retry transient transport/rate-limit errors, retaining the exact revision."""
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(retries):
        try:
            return api.repo_info(source['repo'], repo_type=source['repo_type'],
                                 revision=source['revision'], files_metadata=True)
        except (httpx.TransportError, HfHubHTTPError) as exc:
            response = getattr(exc, 'response', None)
            if response is not None and response.status_code != 429 and response.status_code < 500:
                raise
            if attempt + 1 == retries:
                raise
            print(f"Transient Hub error for {source['repo']}; retry {attempt+2}/{retries}", flush=True)
            time.sleep(min(2 ** (attempt + 1), 16))


def run(data_dir: Path, *, only: str = 'all', verify_only: bool = False,
        force_download: bool = False, max_workers: int = 4, retries: int = 3) -> dict:
    data_dir = data_dir.expanduser().resolve()
    selected = list(SOURCES) if only == 'all' else [only]
    manifest_path = data_dir / 'assets_manifest.json'
    if verify_only and not manifest_path.is_file():
        raise FileNotFoundError(f'Download first; offline manifest missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {'assets': {}}
    if not verify_only:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi(token=False)
        data_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        source = SOURCES[name]
        root = checked_path(data_dir, source['directory'])
        if verify_only:
            record = manifest['assets'].get(name)
            if not record or any(record.get(k) != source[k] for k in ('repo', 'repo_type', 'revision', 'directory')):
                raise ValueError(f'Missing or mismatched pinned manifest for {name}')
            files = record['files']
        else:
            info = fetch_inventory(api, source, retries=retries)
            siblings = [s for s in info.siblings if name != 'tokenizer' or s.rfilename in TOKENIZER_HASHES]
            files = [expected_file(s) for s in siblings]
            if not files:
                raise ValueError(f'Empty upstream inventory for {name}')
            if name == 'tokenizer' and {f['path'] for f in files} != set(TOKENIZER_HASHES):
                raise ValueError('Upstream tokenizer is missing required files')
            for entry in files:
                checked_path(root, entry['path'])
            print(f"Download {name}: {source['repo']}@{source['revision']} "
                  f"({sum(f['bytes'] for f in files)/2**30:.2f} GiB) -> {root}", flush=True)
            snapshot_download(
                source['repo'], repo_type=source['repo_type'], revision=source['revision'],
                local_dir=root, allow_patterns=[f['path'] for f in files],
                max_workers=max_workers, force_download=force_download, token=False,
            )
        verify_files(root, files)
        if name == 'tokenizer':
            validate_tokenizer(root)
        if not verify_only:
            manifest['assets'][name] = {**source, 'path': str(root), 'files': files,
                                        'bytes': sum(f['bytes'] for f in files), 'verified': True}
            manifest['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
            temporary = manifest_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.replace(manifest_path)
        print(f'Verified {name}: {len(files)} files', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.home() / 'model_data/pi05-snapflow')
    parser.add_argument('--only', choices=['all', *SOURCES], default='all')
    parser.add_argument('--verify-only', action='store_true', help='Validate existing manifest and files, offline')
    parser.add_argument('--force-download', action='store_true')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--retries', type=int, default=3, help='Attempts for transient Hub inventory errors')
    args = parser.parse_args()
    if args.max_workers < 1 or args.retries < 1 or args.verify_only and args.force_download:
        parser.error('Invalid worker count or conflicting download/verify options')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    run(args.data_dir, only=args.only, verify_only=args.verify_only,
        force_download=args.force_download, max_workers=args.max_workers, retries=args.retries)
    print('Pinned assets are ready.', flush=True)


if __name__ == '__main__':
    main()
