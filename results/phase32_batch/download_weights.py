#!/usr/bin/env python3
"""Restore the original checkpoint directory tree from GitHub Release assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    for index, item in enumerate(manifest['files'], 1):
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe relative path in manifest')
        target = destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Destination escapes output directory')
        if target.exists():
            if target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']:
                print(f'[{index}/{len(manifest["files"])}] Verified {relative}', flush=True)
                continue
            raise FileExistsError(f'Existing file differs; no overwrite performed: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.cxr-download-', dir=target.parent) as temp:
            subprocess.run(['gh', 'release', 'download', manifest['release_tag'],
                            '--repo', manifest['repository'], '--pattern', item['asset_name'],
                            '--dir', temp], check=True)
            downloaded = Path(temp) / item['asset_name']
            if downloaded.stat().st_size != item['bytes'] or sha256(downloaded) != item['sha256']:
                raise ValueError(f'Download size or SHA-256 mismatch: {relative}')
            # Atomic creation preserves any file concurrently created by another process.
            os.link(downloaded, target)
        print(f'[{index}/{len(manifest["files"])}] Restored {relative}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Result root into which experiments/... files are restored')
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('binary_release_manifest.json'))
    args = parser.parse_args()
    restore(args.manifest, args.output)
