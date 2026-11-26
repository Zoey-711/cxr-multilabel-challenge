"""CPU-only official Test ID, probability and SHA audit for the H17 batch=1 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cost import validate_submission, validate_wide
from predict import HERE, load_config, sha256
from cxr.data.archive import read_split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_split_rows(args.archive, "test")
    expected = {(row["Subject_id"], row["Study_id"]) for row in rows}
    if len(rows) != 2218 or len(expected) != 1414:
        raise ValueError("Official unlabeled Test metadata size mismatch")
    submission = args.results / "submission.csv"
    probabilities = args.results / "test_predictions.csv"
    manifest = json.loads((args.results / "prediction_manifest.json").read_text(encoding="utf-8"))
    matrix = validate_submission(submission)
    if set(matrix) != expected:
        raise ValueError("Candidate Study IDs do not exactly match official Test metadata")
    wide = validate_wide(probabilities, matrix)
    if wide["status"] != "PASS" or wide["max_abs_diff"] != 0:
        raise ValueError("Wide probabilities differ from Submission")
    if sha256(submission) != manifest["submission_sha256"] or sha256(probabilities) != manifest["test_predictions_sha256"]:
        raise ValueError("Prediction SHA mismatch")
    weight = HERE / config["model_file"]
    if sha256(weight) != config["model_sha256"] or sha256(args.config) != manifest["config_sha256"]:
        raise ValueError("Model/config SHA mismatch")
    report = {"status": "PASS", "images": len(rows), "studies": len(expected),
              "submission_rows": len(matrix) * 10, "classes_per_study": 10,
              "columns": ["Subject_id", "Study_id", "Predict_class", "Probability"],
              "missing_studies": 0, "extra_studies": 0, "duplicate_study_class": 0,
              "probabilities_finite_and_in_range": True,
              "wide_submission_max_abs_diff": wide["max_abs_diff"],
              "test_auc": None, "hidden_test_labels_used": False,
              "submission_bytes": submission.stat().st_size,
              "submission_sha256": sha256(submission),
              "weight_sha256": sha256(weight), "config_sha256": sha256(args.config),
              "latency_status": "T4_UNVERIFIED"}
    output = args.results / "validation_report.json"
    if output.exists():
        parser.error("Refusing to overwrite validation report")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
