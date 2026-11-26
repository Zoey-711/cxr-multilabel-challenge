"""Export only the official unlabeled Test image-to-Study index from its ZIP."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from predict import HERE, load_config
from cxr.data.archive import read_split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_config(args.config)
    if args.output.exists():
        parser.error("Refusing to overwrite existing Test index")
    rows = read_split_rows(args.archive, "test")
    keys = {(r["Subject_id"], r["Study_id"]) for r in rows}
    if len(rows) != 2218 or len(keys) != 1414:
        raise ValueError("Official Test index coverage mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id", "image_path"])
        for row in rows:
            writer.writerow([row["Subject_id"], row["Study_id"], f"{row['Dicom_id']}.jpg"])


if __name__ == "__main__":
    main()
