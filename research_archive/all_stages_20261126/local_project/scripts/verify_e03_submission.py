"""Read-only independent verification of the sealed E03 candidate bundle."""

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


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def vector(row):
    return [float(row[f"probability_{i}"]) for i in range(1, 11)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--version", default="E03_20261017")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    directory = args.output_root / args.version
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    fusion = json.loads((directory / "fusion_config.json").read_text(encoding="utf-8"))
    if manifest["version"] != args.version or manifest["model_versions"] != ["B01", "B02", "B03", "M02"] or \
            manifest["upload_status"] != "generated" or manifest["submission_policy"] != "all_classes" or \
            manifest["validation_macro_auc"] != 0.8099309507263751:
        raise ValueError("E03 manifest identity, model list, validation AUC, or upload state differs")
    if set(manifest["files"]) | {"manifest.json"} != {p.name for p in directory.iterdir() if p.is_file()}:
        raise ValueError("Bundle file list differs from manifest")
    for name, record in manifest["files"].items():
        path = directory / name
        if sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"] or \
                path.stat().st_mode & stat.S_IWRITE:
            raise ValueError(f"Unsealed or changed file: {name}")
    if (directory / "manifest.json").stat().st_mode & stat.S_IWRITE:
        raise ValueError("Manifest is not read-only")
    source = args.output_root / "E01_20261016"
    e01_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if sha256(source / "manifest.json") != fusion["source_E01_manifest_sha256"] or \
            sha256(source / "probability_matrix.csv") != fusion["source_E01_probability_matrix_sha256"]:
        raise ValueError("E01 source probability provenance differs")
    frozen = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    m02 = json.loads(Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/frozen_manifest.json").read_text(encoding="utf-8"))
    for model in manifest["model_versions"]:
        expected = m02 if model == "M02" else frozen[model]
        if manifest["checkpoint_sha256"][model] != expected["best_model.pt"]["sha256"]:
            raise ValueError(f"Checkpoint provenance differs: {model}")
        if model != "M02" and e01_manifest["checkpoint_sha256"][model] != manifest["checkpoint_sha256"][model]:
            raise ValueError(f"Reused E01 model provenance differs: {model}")
    metadata = read_split_rows(archive_path(load_config(args.config)), "test")
    expected_keys = {(row["Subject_id"], row["Study_id"]) for row in metadata}
    expected_images = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in metadata}
    checked = verify_candidate(directory / "probability_matrix.csv", directory / "submission.csv",
                               expected_keys, policy="all_classes")
    if len(metadata) != 2218 or len(expected_keys) != 1414 or checked["candidate_rows"] != 14140 or \
            checked["csv_sha256"] != manifest["csv_sha256"]:
        raise ValueError("E03 CSV or test metadata coverage differs")
    submission_rows = read_rows(directory / "submission.csv")
    for study_index, key in enumerate(sorted(expected_keys)):
        study_rows = submission_rows[study_index * 10:(study_index + 1) * 10]
        if [(row["Subject_id"], row["Study_id"]) for row in study_rows] != [key] * 10 or \
                [int(row["Predict_class"]) for row in study_rows] != list(range(1, 11)):
            raise ValueError(f"E03 Study or class row order differs: {key}")
    e01 = {(row["Subject_id"], row["Study_id"]): vector(row)
           for row in read_rows(source / "probability_matrix.csv")}
    e03 = {(row["Subject_id"], row["Study_id"]): vector(row)
           for row in read_rows(directory / "probability_matrix.csv")}
    m02_studies = read_rows(directory / "m02_study_probabilities.csv")
    m02_images = read_rows(directory / "m02_image_probabilities.csv")
    if not (len(e01) == len(e03) == len(m02_studies) == 1414):
        raise ValueError("Mismatched Study counts")
    if len(m02_images) != 2218 or \
            {(row["Subject_id"], row["Study_id"], row["Image_id"]) for row in m02_images} != expected_images:
        raise ValueError("M02 image probability coverage differs")
    grouped = defaultdict(list)
    for row in m02_images:
        grouped[(row["Subject_id"], row["Study_id"])].append(vector(row))
    m02_lookup = {(row["Subject_id"], row["Study_id"]): vector(row) for row in m02_studies}
    if set(e01) != set(e03) or set(e03) != set(m02_lookup) or set(e03) != expected_keys:
        raise ValueError("Subject/Study alignment differs")
    for key in expected_keys:
        source_vectors = grouped[key]
        for j in range(10):
            mean = sum(values[j] for values in source_vectors) / len(source_vectors)
            fused = 0.5 * e01[key][j] + 0.5 * m02_lookup[key][j]
            if not math.isclose(m02_lookup[key][j], mean, rel_tol=0, abs_tol=1e-12) or \
                    not math.isclose(e03[key][j], fused, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"M02 Mean or E03 fusion mismatch: {key}, class {j+1}")
    history = [row for row in read_rows(args.output_root / "submission_history.csv")
               if row["version"] == args.version]
    if len(history) != 1 or history[0]["upload_status"] != "generated" or \
            history[0]["csv_sha256"] != checked["csv_sha256"] or history[0]["official_score"]:
        raise ValueError("E03 history is absent or indicates upload")
    print(json.dumps({"status": "verified_ready_for_manual_upload", "image_count": 2218,
                      "study_count": 1414, "csv_rows": 14140,
                      "m02_image_to_study_mean": "verified", "e03_formula": "verified",
                      "frozen_checkpoint_hashes": "verified", "source_e01_hashes": "verified",
                      "csv_sha256": checked["csv_sha256"], "csv_bytes": checked["csv_bytes"],
                      "upload_status": "generated"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
