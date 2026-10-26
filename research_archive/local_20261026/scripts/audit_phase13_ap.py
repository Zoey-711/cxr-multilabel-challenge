"""Recompute tie-aware ten-class AP from frozen validation probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cxr.engine.b01_analysis import read_study_predictions
from train_study_heads import Studies, ap as earlier_ap
from train_v03_residual import load_previous
from analyze_phase9_errors import average_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache = root / cfg["source"]["feature_cache"]
    source = root / cfg["source"]["m02_run"]
    prior = root / cfg["source"]["phase12_run"]
    v03 = root / "outputs/experiments/v03_m02_multiview_residual"
    output = root / "outputs/analysis/phase13_ap_tieaware"
    if output.exists():
        raise FileExistsError(output)
    val = Studies(cache, "val")
    original, original_y, subjects, studies = read_study_predictions(source / "validation_analysis/study_predictions.csv")
    lookup = {key: i for i, key in enumerate(zip(subjects, studies))}
    order = [lookup[key] for key in val.keys]
    if not np.array_equal(original_y[order], val.labels):
        raise ValueError("M02 labels differ")
    predictions = {"M02": original[order]}
    for kind in ("v00", "v01", "v02"):
        predictions[kind] = load_previous(prior / kind / "study_predictions.csv", val.keys, val.labels)
    predictions["V03"] = load_previous(v03 / "study_predictions.csv", val.keys, val.labels)
    result = {}
    largest_change = 0.0
    for name, probs in predictions.items():
        rows = []
        for j in range(10):
            value = average_precision(val.labels[:, j], probs[:, j])
            former = earlier_ap(val.labels[:, j], probs[:, j])
            largest_change = max(largest_change, abs(value - former))
            rows.append({"class_id": j + 1, "positive_studies": int(val.labels[:, j].sum()),
                         "ap_tie_aware": value, "prior_ap": former})
        result[name] = {"macro_ap_tie_aware": float(np.mean([row["ap_tie_aware"] for row in rows])),
                        "classes": rows}
    output.mkdir(parents=True)
    document = {"method": "Noninterpolated average precision; equal probability scores grouped at the same threshold",
                "validation_studies": len(val), "max_absolute_difference_from_earlier_ap": largest_change,
                "scores": result}
    (output / "metrics.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Tie-aware AP audit for Phase 13", "",
             "Recomputed from the frozen, Study-aligned probabilities of the same 859 validation Studies. Equal scores are evaluated as one threshold group. Earlier AUC and bootstrap results are unaffected; earlier AP values can shift slightly when scores tie.",
             "", f"Maximum absolute difference from the earlier AP implementation: {largest_change:.6f}.",
             "", "| Class | M02 AP | V00 AP | V01 AP | V02 AP | V03 AP |", "|---|---:|---:|---:|---:|---:|"]
    for j in range(10):
        lines.append(f"| {j+1} | " + " | ".join(f"{result[name]['classes'][j]['ap_tie_aware']:.5f}" for name in ("M02", "v00", "v01", "v02", "V03")) + " |")
    lines += ["", "Macro-AP: " + ", ".join(f"{name} {result[name]['macro_ap_tie_aware']:.5f}" for name in ("M02", "v00", "v01", "v02", "V03")) + "."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"max_ap_correction": largest_change,
                      "v03_macro_ap_tie_aware": result["V03"]["macro_ap_tie_aware"]}))


if __name__ == "__main__":
    main()
