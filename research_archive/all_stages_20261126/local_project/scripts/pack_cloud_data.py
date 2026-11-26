"""Prepare Train/Val-only ZIPs locally; never copy hidden Test into the cloud pack."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from cxr.data.archive import archive_path
from cxr.utils.config import load_config


SPLITS = ("train", "valid")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pack_split(original: zipfile.ZipFile, items: list[zipfile.ZipInfo], path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as packed:
        for item in items:
            with original.open(item) as src, packed.open(item.filename, "w", force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    args = parser.parse_args()
    source_path = archive_path(load_config(args.config))
    if not source_path.is_file():
        parser.error("Packaging requires the original combined ZIP")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path) as original:
        groups = {split: [item for item in original.infolist()
                          if not item.is_dir() and (item.filename.startswith(f"cxr_first/{split}/") or
                          item.filename == f"cxr_first/{split}.csv")]
                  for split in SPLITS}
        expected = {"train": 30345, "valid": 1423}
        if {split: len(items) for split, items in groups.items()} != expected:
            raise ValueError("Original Train/Val file count differs from audited archive")
        estimated = sum(item.file_size for items in groups.values() for item in items)
        if shutil.disk_usage(output.parent).free < estimated + int(args.reserve_gb * 1_000_000_000):
            raise OSError("Insufficient local free space for split ZIP packaging plus reserve")
        output.mkdir()
        files = {}
        for split in SPLITS:
            path = output / f"{split}.zip"
            pack_split(original, groups[split], path)
            with zipfile.ZipFile(path) as check:
                names = {item.filename for item in check.infolist() if not item.is_dir()}
            if names != {item.filename for item in groups[split]}:
                raise ValueError(f"Repacked {split} archive has missing/extra members")
            files[f"{split}.zip"] = {"bytes": path.stat().st_size, "sha256": sha256(path),
                                     "members": len(groups[split])}
    manifest = {"format": "cxr_split_zip_v1", "source_basename": source_path.name,
                "excluded": "hidden Test images and test.csv", "files": files}
    (output / "pack_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
