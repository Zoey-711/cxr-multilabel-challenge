"""Map the official 100-image timing sample to a full unlabeled Test image manifest.

This never reads Test labels.  It only permits a Study-level comparison when
every image belonging to a Study appears in the measured sample.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from predict import HERE, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    parser.add_argument("full_test_manifest", type=Path,
                        help="CSV: Subject_id,Study_id,image_path for every Test image")
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    load_config(args.config)
    if args.output.exists():
        parser.error("Refusing to overwrite existing index")
    with args.full_test_manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Subject_id", "Study_id", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Full Test image manifest needs Subject_id,Study_id,image_path")
    by_name = {}
    counts = Counter()
    for row in rows:
        name = Path(row["image_path"]).name
        if name in by_name or not all(row[k] for k in required):
            raise ValueError("Duplicate image basename or missing manifest ID/path")
        by_name[name] = row
        counts[(row["Subject_id"], row["Study_id"])] += 1
    if len(counts) != 1414:
        raise ValueError(f"Expected 1414 Test Studies, found {len(counts)}")
    images = sorted(p for p in args.pictures_100.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 100 or len({p.name for p in images}) != 100:
        raise ValueError("Expected exactly 100 distinct timing images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["image", "Subject_id", "Study_id", "study_image_count"])
        writer.writeheader()
        for path in images:
            row = by_name.get(path.name)
            if row is None:
                raise ValueError(f"Timing image absent from full Test manifest: {path.name}")
            key = (row["Subject_id"], row["Study_id"])
            writer.writerow({"image": str(path.relative_to(args.pictures_100)),
                             "Subject_id": key[0], "Study_id": key[1],
                             "study_image_count": counts[key]})


if __name__ == "__main__":
    main()
