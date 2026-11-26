"""Paired patient bootstrap comparison of two completed best-checkpoint validation runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def aligned_predictions(b01_file: Path, b03_file: Path):
    p1, t1, subjects1, studies1 = read_study_predictions(b01_file)
    p2, t2, subjects2, studies2 = read_study_predictions(b03_file)
    b03_index = {(subject, study): index for index, (subject, study) in enumerate(zip(subjects2, studies2))}
    keys1 = list(zip(subjects1, studies1))
    if set(keys1) != set(b03_index):
        raise ValueError("B01 and B03 validation Study IDs differ")
    order = [b03_index[key] for key in keys1]
    p2, t2 = p2[order], t2[order]
    if not np.array_equal(t1, t2):
        raise ValueError("B01 and B03 validation labels differ")
    return p1, p2, t1, subjects1


def paired_patient_bootstrap(p1, p2, targets, subjects, *, replicates=2000, seed=42):
    groups = {}
    for index, subject in enumerate(subjects):
        groups.setdefault(subject, []).append(index)
    patient_indices = [np.asarray(values, dtype=np.int64) for values in groups.values()]
    rng = np.random.default_rng(seed)
    differences, invalid = [], 0
    for _ in range(replicates):
        sample = rng.integers(0, len(patient_indices), size=len(patient_indices))
        indices = np.concatenate([patient_indices[index] for index in sample])
        auc1 = [binary_auc(targets[indices, j], p1[indices, j]) for j in range(10)]
        auc2 = [binary_auc(targets[indices, j], p2[indices, j]) for j in range(10)]
        if any(value is None for value in auc1 + auc2):
            invalid += 1
            continue
        differences.append(float(np.mean(np.asarray(auc2) - np.asarray(auc1))))
    if len(differences) < 100:
        raise ValueError("Too few valid paired bootstrap replicates")
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {"method": "paired patient cluster bootstrap; draw each patient once per replicate and use the same sampled studies for B01 and B03",
            "seed": seed, "patient_count": len(patient_indices), "requested_replicates": replicates,
            "valid_replicates": len(differences), "invalid_replicates_excluded": invalid,
            "invalid_rule": "Exclude a replicate if any of ten classes lacks positive or negative studies; do not impute AUC",
            "difference_ci_95_lower": float(lower), "difference_ci_95_upper": float(upper)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="B03 config")
    parser.add_argument("--b01-run-dir", required=True, type=Path)
    parser.add_argument("--b03-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    b01, b03, output = args.b01_run_dir.resolve(), args.b03_run_dir.resolve(), args.output_dir.resolve()
    if b01 == b03 or output == b01 or b01 in output.parents:
        parser.error("Comparison output must not overwrite B01")
    if load_config(args.config) != load_config(b03 / "config.yaml"):
        parser.error("B03 config differs from its frozen run copy")
    for run in (b01, b03):
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            parser.error(f"Run is not complete: {run}")
    p1, p2, targets, subjects = aligned_predictions(
        b01 / "validation_analysis" / "study_predictions.csv",
        b03 / "validation_analysis" / "study_predictions.csv")
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        auc1 = binary_auc(targets[:, index], p1[:, index])
        auc2 = binary_auc(targets[:, index], p2[:, index])
        if auc1 is None or auc2 is None:
            raise ValueError(f"Original validation data lack both labels for class {index+1}")
        rows.append({"class_id": index + 1, "class_name": name, "positive_studies": int(targets[:, index].sum()),
                     "negative_studies": int((targets[:, index] == 0).sum()),
                     "b01_auc": auc1, "b03_auc": auc2, "delta_b03_minus_b01": auc2 - auc1})
    bootstrap = paired_patient_bootstrap(p1, p2, targets, subjects,
                                         replicates=args.bootstrap_replicates, seed=args.seed)
    b01_auc = float(np.mean([row["b01_auc"] for row in rows]))
    b03_auc = float(np.mean([row["b03_auc"] for row in rows]))
    best1 = json.loads((b01 / "best_metrics.json").read_text(encoding="utf-8"))
    best2 = json.loads((b03 / "best_metrics.json").read_text(encoding="utf-8"))
    if abs(b01_auc - best1["val"]["macro_auc"]) > 1e-8 or abs(b03_auc - best2["val"]["macro_auc"]) > 1e-8:
        raise ValueError("Independent best-checkpoint AUC does not match training records")
    result = {"study_count": len(targets), "b01_best_epoch": best1["epoch"], "b03_best_epoch": best2["epoch"],
              "b01_macro_auc": b01_auc, "b03_macro_auc": b03_auc,
              "macro_auc_delta_b03_minus_b01": b03_auc - b01_auc,
              "b01_macro_f1_at_0_5": best1["val"]["macro_f1"],
              "b03_macro_f1_at_0_5": best2["val"]["macro_f1"], "paired_bootstrap": bootstrap}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_class_auc_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "paired_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# B01/B03 paired validation comparison", "",
              f"All {len(targets)} validation Studies aligned by Subject_id and Study_id with identical targets.",
              f"B01 best epoch {best1['epoch']}: Macro-AUC {b01_auc:.5f}, Macro-F1@0.5 {best1['val']['macro_f1']:.5f}.",
              f"B03 best epoch {best2['epoch']}: Macro-AUC {b03_auc:.5f}, Macro-F1@0.5 {best2['val']['macro_f1']:.5f}.",
              f"B03 minus B01 Macro-AUC: {b03_auc-b01_auc:+.5f}; paired patient-bootstrap 95% CI "
              f"[{bootstrap['difference_ci_95_lower']:+.5f}, {bootstrap['difference_ci_95_upper']:+.5f}].",
              f"Paired bootstrap valid/attempted: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']}; "
              f"invalid excluded: {bootstrap['invalid_replicates_excluded']}.", "",
              "| ID | Class | B01 AUC | B03 AUC | Delta |", "|---:|---|---:|---:|---:|"]
    report += [f"| {row['class_id']} | {row['class_name']} | {row['b01_auc']:.5f} | "
               f"{row['b03_auc']:.5f} | {row['delta_b03_minus_b01']:+.5f} |" for row in rows]
    (output / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
