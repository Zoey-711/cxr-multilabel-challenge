"""Reissue verified frozen E02 Study probabilities as ten rows per test Study."""

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.versioned_submission import (
    CSV_FIELDS, MATRIX_FIELDS, append_history, seal_bundle, sha256,
    verify_candidate, write_csv,
)
from cxr.utils.config import load_config


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M01": Path("outputs/experiments/m01_convnext_tiny_384_20261016"),
}


def csv_rows(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"Unexpected header in {path}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-version", default="E02_20261016")
    parser.add_argument("--version", default="E02_20261016_full10")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    source = args.output_root / args.source_version
    target = args.output_root / args.version
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing version: {target}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != args.source_version or manifest["model_versions"] != list(RUNS):
        raise ValueError("Source is not the frozen four-model E02")
    if manifest["fusion_weights"] != {"B01": 1 / 6, "B02": 1 / 6, "B03": 1 / 6, "M01": 0.5}:
        raise ValueError("Source fusion weights differ")
    if {p.name for p in source.iterdir() if p.is_file()} != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Source file listing differs from sealed manifest")
    for name, entry in manifest["files"].items():
        path = source / name
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Source probability bundle SHA/size mismatch: {name}")
    frozen_trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_m01 = json.loads((RUNS["M01"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    checkpoint_sources = {}
    for name, run in RUNS.items():
        path = run / "best_model.pt"
        digest = sha256(path)
        expected = (frozen_m01 if name == "M01" else frozen_trio[name])["best_model.pt"]["sha256"]
        if digest != expected or digest != manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Frozen best checkpoint SHA mismatch: {name}")
        checkpoint_sources[name] = {"path": str(path), "sha256": digest,
                                    "size_bytes": path.stat().st_size}
    test_rows = read_split_rows(archive_path(load_config(args.config)), "test")
    keys = {(row["Subject_id"], row["Study_id"]) for row in test_rows}
    study_ids = {row["Study_id"] for row in test_rows}
    if len(test_rows) != 2218 or len(keys) != 1414 or len(study_ids) != 1414:
        raise ValueError("Test metadata does not have 2,218 images and 1,414 unique Study IDs")
    matrix_path = source / "probability_matrix.csv"
    matrix_rows = csv_rows(matrix_path, MATRIX_FIELDS)
    matrix = {}
    for row in matrix_rows:
        key = (row["Subject_id"], row["Study_id"])
        if key in matrix:
            raise ValueError(f"Duplicate Study in source matrix: {key}")
        values = [float(row[f"probability_{i}"]) for i in range(1, 11)]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError(f"Nonfinite or out-of-range probability: {key}")
        matrix[key] = values
    if len(matrix_rows) != 1414 or set(matrix) != keys:
        raise ValueError("Source probability matrix Study coverage differs")
    per_model = {}
    probability_sources = {"probability_matrix.csv": sha256(matrix_path)}
    for name in RUNS:
        path = source / f"{name.lower()}_study_probabilities.csv"
        rows = csv_rows(path, MATRIX_FIELDS)
        lookup = {(row["Subject_id"], row["Study_id"]):
                  [float(row[f"probability_{i}"]) for i in range(1, 11)] for row in rows}
        if len(rows) != 1414 or len(lookup) != 1414 or set(lookup) != keys:
            raise ValueError(f"Missing/duplicate Study in {name} probability file")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for values in lookup.values() for p in values):
            raise ValueError(f"Invalid {name} probabilities")
        per_model[name] = lookup
        probability_sources[path.name] = sha256(path)
        probability_sources[f"{name.lower()}_image_probabilities.csv"] = sha256(
            source / f"{name.lower()}_image_probabilities.csv")
    for key, values in matrix.items():
        for index, actual in enumerate(values):
            expected = 0.5 * sum(per_model[name][key][index] for name in ("B01", "B02", "B03")) / 3
            expected += 0.5 * per_model["M01"][key][index]
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Source matrix differs from E02 fusion at {key}, class {index + 1}")
    candidate = [{"Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                  "Predict_class": class_id, "Probability": row[f"probability_{class_id}"]}
                 for row in matrix_rows for class_id in range(1, 11)]
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(matrix_path, target / "probability_matrix.csv")
    write_csv(target / "submission.csv", CSV_FIELDS, candidate)
    check = verify_candidate(target / "probability_matrix.csv", target / "submission.csv",
                             keys, policy="all_classes")
    if check["candidate_rows"] != 14140 or check["candidate_covered_studies"] != 1414 or \
            check["zero_prediction_studies"] != 0:
        raise ValueError("Full-ten submission failed row/coverage validation")
    check.update({"all_ten_classes_per_study": True, "unique_study_class_pairs": True,
                  "finite_probabilities": True, "no_gate_column": True,
                  "source_fusion_verified": True, "source_bundle_sha256_verified": True,
                  "frozen_checkpoint_sha256_verified": True,
                  "format_status": "ready_for_manual_upload"})
    (target / "validation_report.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    result_manifest = {"version": args.version, "generated_at": timestamp,
                       "source_version": args.source_version,
                       "source_manifest_sha256": sha256(source / "manifest.json"),
                       "source_probability_files_sha256": probability_sources,
                       "model_versions": list(RUNS), "checkpoints": checkpoint_sources,
                       "fusion_formula": "0.5*((B01+B02+B03)/3)+0.5*M01",
                       "fusion_weights": manifest["fusion_weights"],
                       "validation_macro_auc": manifest["validation_macro_auc"],
                       "submission_policy": "all ten class probabilities per Study; default 0.5; no Gate",
                       "csv_sha256": check["csv_sha256"],
                       "files": {path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                                 for path in target.iterdir() if path.is_file()},
                       "upload_status": "generated", "official_score": None}
    (target / "manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    append_history(args.output_root, {"version": args.version, "generated_at": timestamp,
                "candidate_csv": str(target / "submission.csv"), "csv_sha256": check["csv_sha256"],
                "validation_macro_auc": manifest["validation_macro_auc"],
                "row_count": check["candidate_rows"], "study_count": check["matrix_studies"],
                "zero_prediction_studies": 0, "format_status": "ready_for_manual_upload",
                "upload_status": "generated", "uploaded_at": "", "official_score": ""})
    seal_bundle(target)
    print(json.dumps({"directory": str(target.resolve()), "validation": check},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
