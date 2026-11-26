"""Audit a manifest after its actual field names have been inspected."""

import argparse
import json
from pathlib import Path

from cxr.data.audit import ManifestSchema, audit_csv
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    columns = data["columns"]
    required = ("subject_id", "study_id", "image_path")
    if any(not columns.get(key) for key in required):
        parser.error("Set actual data.columns subject_id, study_id, and image_path after inspecting metadata")
    manifest = data.get(f"{args.split}_manifest")
    if not manifest:
        parser.error(f"Set data.{args.split}_manifest to the supplied CSV path")
    if not data.get("image_root"):
        parser.error("Set data.image_root to the supplied image directory")
    labels = columns.get("labels") or []
    if not isinstance(labels, list):
        parser.error("data.columns.labels must be a list of actual column names")
    if args.split != "test" and not labels:
        parser.error("Set data.columns.labels to the actual label columns for train/val")
    schema = ManifestSchema(columns["subject_id"], columns["study_id"], columns["image_path"], tuple(labels))
    report = audit_csv(manifest, schema, image_root=data.get("image_root"))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
