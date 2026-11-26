"""Read-only independent validation of E07's sealed all-class submission bundle."""

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


FIELDS = tuple(f"probability_{i}" for i in range(1, 11))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def vector(row: dict) -> tuple[float, ...]:
    return tuple(float(row[name]) for name in FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--version", default="E07_20261018")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    directory = args.output_root / args.version
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    fusion = json.loads((directory / "fusion_config.json").read_text(encoding="utf-8"))
    if (manifest["version"] != args.version or
            manifest["model_versions"] != ["B01", "B02", "B03", "M02", "T01"] or
            manifest["upload_status"] != "generated" or manifest["submission_policy"] != "all_classes" or
            abs(manifest["validation_macro_auc"] - 0.8115193037842005) > 1e-12):
        raise ValueError("E07 manifest identity or status differs")
    if set(manifest["files"]) | {"manifest.json"} != {p.name for p in directory.iterdir() if p.is_file()}:
        raise ValueError("E07 file set differs from manifest")
    for name, record in manifest["files"].items():
        path = directory / name
        if sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"] or \
                path.stat().st_mode & stat.S_IWRITE:
            raise ValueError(f"Changed or unsealed E07 artifact: {name}")
    if (directory / "manifest.json").stat().st_mode & stat.S_IWRITE:
        raise ValueError("E07 manifest is not read-only")
    e03 = args.output_root / "E03_20261017"
    e03_manifest = json.loads((e03 / "manifest.json").read_text(encoding="utf-8"))
    if (sha256(e03 / "manifest.json") != fusion["source_E03_manifest_sha256"] or
            sha256(e03 / "probability_matrix.csv") != fusion["source_E03_probability_matrix_sha256"] or
            sha256(e03 / "submission.csv") != e03_manifest["csv_sha256"]):
        raise ValueError("E03 source changed")
    t01_run = Path("outputs/experiments/t01_swin_tiny_384_20261018")
    t01_frozen = json.loads((t01_run / "frozen_manifest.json").read_text(encoding="utf-8"))
    if (sha256(t01_run / "best_model.pt") != t01_frozen["best_model.pt"]["sha256"] or
            manifest["checkpoint_sha256"]["T01"] != t01_frozen["best_model.pt"]["sha256"]):
        raise ValueError("T01 checkpoint hash mismatch")
    metadata = read_split_rows(archive_path(load_config(args.config)), "test")
    expected = {(row["Subject_id"], row["Study_id"]) for row in metadata}
    image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in metadata}
    checked = verify_candidate(directory / "probability_matrix.csv", directory / "submission.csv",
                               expected, policy="all_classes")
    if (len(metadata), len(expected), len(image_keys), checked["candidate_rows"],
            checked["candidate_covered_studies"]) != (2218, 1414, 2218, 14140, 1414):
        raise ValueError("E07 image, Study or CSV coverage differs")
    if checked["csv_sha256"] != manifest["csv_sha256"]:
        raise ValueError("E07 CSV SHA mismatch")
    images = rows(directory / "t01_image_probabilities.csv")
    grouped = defaultdict(list)
    for row in images:
        grouped[(row["Subject_id"], row["Study_id"])].append(vector(row))
    if len(images) != 2218 or \
            {(row["Subject_id"], row["Study_id"], row["Image_id"]) for row in images} != image_keys:
        raise ValueError("T01 image-level probability records differ from test metadata")
    e03_lookup = {(row["Subject_id"], row["Study_id"]): vector(row)
                  for row in rows(e03 / "probability_matrix.csv")}
    t01_lookup = {(row["Subject_id"], row["Study_id"]): vector(row)
                  for row in rows(directory / "t01_study_probabilities.csv")}
    fused = {(row["Subject_id"], row["Study_id"]): vector(row)
             for row in rows(directory / "probability_matrix.csv")}
    if not (set(e03_lookup) == set(t01_lookup) == set(fused) == expected):
        raise ValueError("E03/T01/E07 Study IDs differ")
    for key in expected:
        for j in range(10):
            mean = sum(item[j] for item in grouped[key]) / len(grouped[key])
            expected_fused = 0.5 * e03_lookup[key][j] + 0.5 * t01_lookup[key][j]
            if (not math.isclose(t01_lookup[key][j], mean, rel_tol=0, abs_tol=1e-12) or
                    not math.isclose(fused[key][j], expected_fused, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"T01 Mean or E07 fusion differs: {key}, class {j + 1}")
    submission = rows(directory / "submission.csv")
    for index, key in enumerate(sorted(expected)):
        group = submission[index * 10:(index + 1) * 10]
        if [(row["Subject_id"], row["Study_id"]) for row in group] != [key] * 10 or \
                [int(row["Predict_class"]) for row in group] != list(range(1, 11)):
            raise ValueError(f"E07 Study/class ordering differs: {key}")
    history = [row for row in rows(args.output_root / "submission_history.csv")
               if row["version"] == args.version]
    if len(history) != 1 or history[0]["upload_status"] != "generated" or \
            history[0]["csv_sha256"] != checked["csv_sha256"] or history[0]["official_score"]:
        raise ValueError("E07 history differs or falsely marks upload")
    print(json.dumps({"status": "verified_ready_for_manual_upload", "images": 2218,
                      "studies": 1414, "data_rows": 14140,
                      "csv_sha256": checked["csv_sha256"], "csv_bytes": checked["csv_bytes"],
                      "zero_predicted_studies_at_0_5": checked["zero_prediction_studies_at_0_5"],
                      "source_e03_sha256": fusion["source_E03_probability_matrix_sha256"],
                      "t01_checkpoint_sha256": manifest["checkpoint_sha256"]["T01"],
                      "upload_status": history[0]["upload_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
