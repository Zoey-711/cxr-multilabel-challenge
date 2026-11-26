"""Issue a new read-only E02 version after the empty-Study rule is confirmed."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.aggregate import StudyPrediction
from cxr.inference.versioned_submission import sha256, verify_candidate, write_bundle
from cxr.utils.config import load_config


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    parser.add_argument("--official-rule-reference", required=True,
                        help="Organizer notice or rule page explicitly allowing omission of zero-prediction Studies")
    args = parser.parse_args()
    if not args.official_rule_reference.strip():
        parser.error("An explicit official rule reference is required")
    root = args.output_root.resolve()
    source = root / args.source_version
    target = root / args.new_version
    if source == target or target.exists():
        parser.error("New version must differ and must not already exist")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != args.source_version:
        raise ValueError("Source version does not match manifest")
    for name, entry in manifest["files"].items():
        path = source / name
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Source bundle file changed: {name}")
    config = load_config(args.config)
    rows = read_split_rows(archive_path(config), "test")
    expected_keys = {(row["Subject_id"], row["Study_id"]) for row in rows}
    verified = verify_candidate(source / "probability_matrix.csv", source / "submission.csv",
                                expected_keys, threshold=0.5)
    if len(rows) != 2218 or verified["matrix_studies"] != 1414:
        raise ValueError("Source bundle does not match the complete test set")
    fusion_config = json.loads((source / "fusion_config.json").read_text(encoding="utf-8"))
    fusion_config["source_bundle_version"] = args.source_version
    fusion_config["empty_study_rule_confirmation"] = (
        args.official_rule_reference.strip())
    image_rows_by_model, study_rows_by_model = {}, {}
    for name in ("B01", "B02", "B03", "M01"):
        image_rows_by_model[name] = read_csv(source / f"{name.lower()}_image_probabilities.csv")
        study_rows_by_model[name] = read_csv(source / f"{name.lower()}_study_probabilities.csv")
    counts = Counter((row["Subject_id"], row["Study_id"]) for row in image_rows_by_model["B01"])
    matrix = read_csv(source / "probability_matrix.csv")
    studies = [StudyPrediction(row["Subject_id"], row["Study_id"],
                               counts[(row["Subject_id"], row["Study_id"])],
                               tuple(float(row[f"probability_{i}"]) for i in range(1, 11)))
               for row in matrix]
    directory, checked = write_bundle(root, args.new_version, studies, expected_keys,
                                      fusion_config, image_rows_by_model, study_rows_by_model,
                                      empty_study_policy_confirmed=True)
    if checked["csv_sha256"] != verified["csv_sha256"]:
        raise ValueError("Reissued CSV differs from the source predictions")
    print(json.dumps({"new_directory": str(directory), "source_version": args.source_version,
                      "csv_sha256_unchanged": True, "format_status": checked["format_status"],
                      "candidate_rows": checked["candidate_rows"],
                      "covered_studies": checked["candidate_covered_studies"],
                      "zero_prediction_studies": checked["zero_prediction_studies"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
