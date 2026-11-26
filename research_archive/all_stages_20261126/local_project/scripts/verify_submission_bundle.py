"""Read-only verification of a versioned submission candidate and its history."""

import argparse
import csv
import json
import math
import stat
from collections import defaultdict
from pathlib import Path

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.versioned_submission import sha256, verify_candidate
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output_root.resolve()
    directory = root / args.version
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != args.version:
        raise ValueError("Manifest version differs from directory")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Bundle files differ from manifest")
    for name, entry in manifest["files"].items():
        path = directory / name
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Bundle file changed after manifest: {name}")
        if path.stat().st_mode & stat.S_IWRITE:
            raise ValueError(f"Bundle file is not marked read-only: {name}")
    if (directory / "manifest.json").stat().st_mode & stat.S_IWRITE:
        raise ValueError("Manifest is not marked read-only")
    rows = read_split_rows(archive_path(config), "test")
    keys = {(row["Subject_id"], row["Study_id"]) for row in rows}
    policy = manifest.get("submission_policy", "threshold")
    checked = verify_candidate(directory / "probability_matrix.csv", directory / "submission.csv",
                               keys, threshold=0.5, policy=policy)
    if len(rows) != 2218 or len(keys) != 1414 or \
            checked["csv_sha256"] != manifest["csv_sha256"]:
        raise ValueError("Test coverage or submission digest differs")
    expected_image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in rows}
    model_studies = {}
    for model in manifest["model_versions"]:
        image_path = directory / f"{model.lower()}_image_probabilities.csv"
        study_path = directory / f"{model.lower()}_study_probabilities.csv"
        with image_path.open("r", encoding="utf-8", newline="") as handle:
            image_rows = list(csv.DictReader(handle))
        with study_path.open("r", encoding="utf-8", newline="") as handle:
            study_rows = list(csv.DictReader(handle))
        image_keys = {(row["Subject_id"], row["Study_id"], row["Image_id"]) for row in image_rows}
        study_keys = {(row["Subject_id"], row["Study_id"]) for row in study_rows}
        if len(image_rows) != 2218 or image_keys != expected_image_keys or \
                len(study_rows) != 1414 or study_keys != keys:
            raise ValueError(f"Per-model probability coverage differs: {model}")
        grouped = defaultdict(list)
        for row in image_rows:
            grouped[(row["Subject_id"], row["Study_id"])].append(
                [float(row[f"probability_{i}"]) for i in range(1, 11)])
        study_lookup = {}
        for row in study_rows:
            key = (row["Subject_id"], row["Study_id"])
            values = [float(row[f"probability_{i}"]) for i in range(1, 11)]
            expected = [sum(vector[j] for vector in grouped[key]) / len(grouped[key]) for j in range(10)]
            if not all(math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
                       for left, right in zip(values, expected)):
                raise ValueError(f"Per-model image-to-Study Mean differs: {model} {key}")
            study_lookup[key] = values
        model_studies[model] = study_lookup
    with (directory / "probability_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    if args.version.startswith("E01_"):
        required_models = {"B01", "B02", "B03"}
    elif args.version.startswith("E02_"):
        required_models = {"B01", "B02", "B03", "M01"}
    else:
        raise ValueError("Unknown fusion scheme")
    if set(model_studies) != required_models:
        raise ValueError("Model set differs from fusion scheme")
    for row in matrix_rows:
        key = (row["Subject_id"], row["Study_id"])
        for j in range(10):
            expected = sum(model_studies[name][key][j] for name in ("B01", "B02", "B03")) / 3
            if "M01" in required_models:
                expected = 0.5 * expected + 0.5 * model_studies["M01"][key][j]
            actual = float(row[f"probability_{j + 1}"])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Fusion formula differs: {key}, class {j + 1}")
    with (root / "submission_history.csv").open("r", encoding="utf-8", newline="") as handle:
        history = [row for row in csv.DictReader(handle) if row["version"] == args.version]
    if len(history) != 1 or history[0]["csv_sha256"] != checked["csv_sha256"] or \
            not history[0]["upload_status"]:
        raise ValueError("Submission history is missing or inconsistent")
    report = json.loads((directory / "prediction_check.json").read_text(encoding="utf-8"))
    if report["candidate_rows"] != checked["candidate_rows"] or \
            report["zero_prediction_studies"] != checked["zero_prediction_studies"]:
        raise ValueError("Stored prediction check differs")
    expected_status = ("ready_for_manual_upload" if policy == "all_classes" or
                       manifest.get("empty_study_omission_confirmed") or
                       checked["zero_prediction_studies"] == 0 else
                       "blocked_pending_official_empty_study_rule")
    if report["format_status"] != expected_status:
        raise ValueError("Stored format status disagrees with the confirmed empty-Study rule")
    effective_status = history[0]["format_status"]
    if effective_status != report["format_status"]:
        event_path = root / "submission_status_events.csv"
        if not event_path.exists():
            raise ValueError("Format status changed without an audit event")
        with event_path.open("r", encoding="utf-8", newline="") as handle:
            events = [event for event in csv.DictReader(handle)
                      if event["version"] == args.version]
        if not events or events[-1]["new_format_status"] != effective_status:
            raise ValueError("Format status history lacks the latest correction")
    print(json.dumps({"version": args.version, "images_per_model": 2218,
                      "model_count": len(manifest["model_versions"]),
                      "studies": checked["matrix_studies"],
                      "csv_rows": checked["candidate_rows"],
                      "covered_studies": checked["candidate_covered_studies"],
                      "zero_prediction_studies": checked["zero_prediction_studies"],
                      "csv_sha256": checked["csv_sha256"],
                      "sealed_format_status": report["format_status"],
                      "effective_format_status": effective_status,
                      "history_upload_status": history[0]["upload_status"],
                      "image_mean_and_fusion_verified": True,
                      "all_bundle_hashes_match": True, "all_bundle_files_read_only": True},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
