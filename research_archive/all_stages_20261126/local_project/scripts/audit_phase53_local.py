"""Compare 100 one-image Val Studies from packaged inference with frozen Val probabilities."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/deployment/phase53_t4_benchmark"
INDEX = ROOT / "outputs/analysis/phase29_deployment/benchmark_images/sample_image_index.csv"
REFERENCES = {
    "E08_ONNX_FP32": ROOT / "outputs/submissions/E08_ONNX_FP32_20261028/final_entry_val_predictions.csv",
    "H17_BATCH1_FP16": ROOT / "outputs/analysis/phase49_h17_batch1_acceleration/val_torch_frozen.csv",
    "E08_PYTORCH_FP16": ROOT / "outputs/submissions/E08_EMA_20261020/val_predictions.csv",
    "H17_ONNX_FP32": ROOT / "outputs/analysis/phase49_h17_batch1_acceleration/val_onnx_fp32.csv",
}

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def wide(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    prefix = "Prob_" if "Prob_1" in rows[0] else "probability_"
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        if key in result:
            raise ValueError(f"Duplicate reference Study: {key}")
        result[key] = np.asarray([float(row[f"{prefix}{i}"]) for i in range(1, 11)])
    return result

def main():
    with INDEX.open(encoding="utf-8-sig", newline="") as stream:
        index = {row["image"]: row for row in csv.DictReader(stream)}
    if len(index) != 100 or any(int(row["study_image_count"]) != 1 for row in index.values()):
        raise ValueError("Local sample must have 100 distinct one-image Val Studies")
    overview = {}
    for candidate, reference_path in REFERENCES.items():
        prediction_path = PACKAGE / "reports" / f"local4060_{candidate}" / "predictions_300.csv"
        report_path = prediction_path.parent / "local_val_audit.json"
        reference = wide(reference_path)
        with prediction_path.open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != 300:
            raise ValueError("Timing run lacks 300 predictions")
        differences = []
        repeat = {}
        gate = 0
        rank = 0
        for row in predictions:
            name = row["image"]
            if name not in index:
                raise ValueError(f"Image absent from Val index: {name}")
            key = index[name]["Subject_id"], index[name]["Study_id"]
            value = np.asarray([float(row[f"Prob_{i}"]) for i in range(1, 11)])
            truth = reference[key]
            differences.extend(np.abs(value - truth))
            repeat.setdefault(name, []).append(value)
            gate += int(np.any((value >= .5) != (truth >= .5)))
            rank += int(not np.array_equal(np.argsort(value, kind="stable"), np.argsort(truth, kind="stable")))
        if set(repeat) != set(index) or any(len(values) != 3 for values in repeat.values()):
            raise ValueError("Missing or repeated image round")
        repeated_max = max(float(np.max(np.abs(values[0] - other)))
                           for values in repeat.values() for other in values[1:])
        diff = np.asarray(differences)
        report = {"candidate": candidate, "source": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
                  "source_sha256": sha(reference_path), "sample_source": "authorized Val images",
                  "val_studies": 100, "rounds": 3, "probabilities_checked": len(diff),
                  "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                  "p99_abs_diff": float(np.percentile(diff, 99)),
                  "gate_mismatch_records": gate, "rank_mismatch_records": rank,
                  "repeat_max_abs_diff": repeated_max,
                  "reference_status": "PASS" if float(diff.max()) <= 1e-3 and gate == 0 and rank == 0 else "DIFFERENT_VERSION_OR_NUMERIC_OUTPUT",
                  "submission_consistency": "UNVERIFIED; official Test sample and mapping absent"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        overview[candidate] = report
    (PACKAGE / "reports/local_val_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(json.dumps({name:{"max_abs_diff": item["max_abs_diff"], "status": item["reference_status"]}
                      for name,item in overview.items()}))

if __name__ == "__main__":
    main()
