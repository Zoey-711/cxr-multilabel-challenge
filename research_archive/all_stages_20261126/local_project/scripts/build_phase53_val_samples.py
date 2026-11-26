"""Extract 100+5 unmodified official Val JPEGs for private T4 reference timing."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxr.data.archive import image_member, read_split_rows
from cxr.data.dataset import preprocess_image

ARCHIVE = Path("F:/") / "\u6570\u636e\u96c6" / "cxr_first.zip"
OLD = ROOT / "outputs/analysis/phase29_deployment/benchmark_images"
OUT = ROOT / "outputs/deployment/phase53_val_t4_samples"
TAR = OUT.with_suffix(".tar.gz")
SIDE = TAR.with_name(TAR.name + ".sha256.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def checked_names(folder: Path, count: int) -> list[str]:
    paths = list(folder.iterdir())
    if len(paths) != count or any(not p.is_file() or p.suffix.lower() != ".jpg" for p in paths):
        raise ValueError(f"Expected exactly {count} reference JPEG names: {folder}")
    names = sorted(p.name for p in paths)
    if len(set(names)) != count:
        raise ValueError("Duplicate image names")
    return names


def hash_archive(path: Path) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    next_report = 8 * 1024**3
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            h.update(block)
            done += len(block)
            if done >= next_report:
                print(f"source archive SHA-256 progress {done/1024**3:.1f}/{total/1024**3:.1f} GiB", flush=True)
                next_report += 8 * 1024**3
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or TAR.exists() or SIDE.exists():
        raise FileExistsError("Phase53 Val sample output already exists; will not overwrite")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    rows = read_split_rows(ARCHIVE, "val")
    by_id = {row["Dicom_id"] + ".jpg": row for row in rows}
    studies = Counter((row["Subject_id"], row["Study_id"]) for row in rows)
    if len(rows) != 1422 or len(by_id) != len(rows) or len(studies) != 859:
        raise ValueError("Unexpected official Val metadata coverage")
    chosen = {"pictures_100": checked_names(OLD / "pictures_100", 100),
              "pictures_5": checked_names(OLD / "warmup_5", 5)}
    if set(chosen["pictures_100"]) & set(chosen["pictures_5"]):
        raise ValueError("Timed and warmup image lists overlap")
    if any(name not in by_id for names in chosen.values() for name in names):
        raise ValueError("Chosen image absent from official Val metadata")
    if any(studies[(by_id[name]["Subject_id"], by_id[name]["Study_id"])] != 1
           for names in chosen.values() for name in names):
        raise ValueError("Selection is not made of complete single-image Studies")

    OUT.mkdir(parents=True)
    files = {}
    old_byte_matches = 0
    with zipfile.ZipFile(ARCHIVE) as source:
        metadata_bytes = source.read("cxr_first/valid.csv")
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
        for group, names in chosen.items():
            target_dir = OUT / group
            target_dir.mkdir()
            for name in names:
                row = by_id[name]
                member = image_member(row, "val")
                raw = source.read(member)
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    tensor = preprocess_image(image, 384, preprocessing="black_pad_v1")
                if tuple(tensor.shape) != (3, 384, 384) or not torch.isfinite(tensor).all():
                    raise ValueError(f"Phase53 preprocessing failed: {name}")
                destination = target_dir / name
                with destination.open("xb") as stream:
                    stream.write(raw)
                byte_sha = hashlib.sha256(raw).hexdigest()
                if digest(destination) != byte_sha:
                    raise ValueError(f"Extracted bytes changed: {name}")
                prior = OLD / ("warmup_5" if group == "pictures_5" else "pictures_100") / name
                old_byte_matches += int(prior.read_bytes() == raw)
                files[f"{group}/{name}"] = {
                    "bytes": len(raw), "sha256": byte_sha,
                    "source_member": member, "source_member_sha256": byte_sha,
                    "Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                    "study_image_count": studies[(row["Subject_id"], row["Study_id"])]}
        print(f"extracted and decoded {len(files)} original Val JPEGs", flush=True)
    index = OUT / "timed_image_index.csv"
    with index.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "Subject_id", "Study_id", "study_image_count"])
        for name in chosen["pictures_100"]:
            row = by_id[name]
            writer.writerow([name, row["Subject_id"], row["Study_id"],
                             studies[(row["Subject_id"], row["Study_id"])]] )
    files["timed_image_index.csv"] = {"bytes": index.stat().st_size, "sha256": digest(index)}
    print("hashing original 55.2 GB source archive", flush=True)
    archive_sha = hash_archive(ARCHIVE)
    manifest = {"purpose": "private T4 reference timing, not organizer-fixed Test timing",
                "source_split": "official Val only", "hidden_test_accessed": False,
                "labels_exported": False, "pre_resize_or_recompression": False,
                "selection": "frozen Phase29 deterministic 105 single-image Val Studies, rechecked against full Val metadata",
                "source_archive_name": ARCHIVE.name, "source_archive_bytes": ARCHIVE.stat().st_size,
                "source_archive_sha256": archive_sha,
                "source_metadata_member": "cxr_first/valid.csv",
                "source_metadata_sha256": metadata_sha,
                "full_val_images": len(rows), "full_val_studies": len(studies),
                "timed_images": 100, "warmup_images": 5,
                "timed_complete_studies": 100, "old_reference_raw_byte_matches": old_byte_matches,
                "files": files}
    manifest_path = OUT / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with tarfile.open(TAR, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(Path(OUT.name) / path.relative_to(OUT)).replace("\\", "/"), recursive=False)
    side = {"archive": TAR.name, "archive_bytes": TAR.stat().st_size,
            "archive_sha256": digest(TAR), "manifest_sha256": digest(manifest_path)}
    SIDE.write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(json.dumps(side, indent=2), flush=True)


if __name__ == "__main__":
    main()
