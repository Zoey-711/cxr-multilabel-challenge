"""Immutable verified backup/restore; never merge into an existing destination."""
import json
import shutil
from pathlib import Path

from cxr.batch.plan import digest, write_json


def backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Backup destination must be outside source')
    if (source/'queue.lock').exists():
        raise RuntimeError('Queue active; use per-task frozen results or drain before full backup')
    if destination.exists():
        raise FileExistsError(destination)
    files={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source,destination)
    if any(digest(destination/rel)!=h for rel,h in files.items()):
        raise ValueError('Backup verification failed')
    write_json(destination/'backup_manifest.json',{'files':files})
    return {'files':len(files),'bytes':sum((destination/r).stat().st_size for r in files)}


def restore(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest=json.loads((source/'backup_manifest.json').read_text())
    for rel,h in manifest['files'].items():
        p=(source/rel).resolve()
        if source not in p.parents or digest(p)!=h:
            raise ValueError('Unsafe or corrupt backup member')
    shutil.copytree(source,destination)
    if any(digest(destination/rel)!=h for rel,h in manifest['files'].items()):
        raise ValueError('Restored hash mismatch')
