"""Compare independently validated M01/M02 Study probabilities with patient bootstrap."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def measures(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    aucs, f1s, details = [], [], []
    for index in range(10):
        y = targets[:, index] == 1
        prediction = probabilities[:, index] >= 0.5
        auc = binary_auc(targets[:, index], probabilities[:, index])
        if auc is None:
            raise ValueError(f"Class {index + 1} lacks positive/negative Studies")
        tp = int((y & prediction).sum())
        fp = int((~y & prediction).sum())
        fn = int((y & ~prediction).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        aucs.append(auc)
        f1s.append(f1)
        details.append({"class_id": index + 1, "class_name": CLASS_NAMES[index],
                        "positive_studies": int(y.sum()), "negative_studies": int((~y).sum()),
                        "auc": auc, "f1_at_0_5": f1, "tp": tp, "fp": fp, "fn": fn})
    return {"macro_auc": float(np.mean(aucs)), "macro_f1_at_0_5": float(np.mean(f1s)),
            "per_class": details}


def patient_bootstrap(m01: np.ndarray, m02: np.ndarray, targets: np.ndarray, subjects: list[str],
                      *, replicates: int, seed: int) -> dict:
    groups = defaultdict(list)
    for index, subject in enumerate(subjects):
        groups[subject].append(index)
    patients = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    deltas, class_deltas, invalid = [], [[] for _ in range(10)], 0
    for _ in range(replicates):
        draw = rng.integers(0, len(patients), size=len(patients))
        indices = np.concatenate([patients[index] for index in draw])
        y = targets[indices]
        if any(y[:, j].min() == y[:, j].max() for j in range(10)):
            invalid += 1
            continue
        auc1 = [binary_auc(y[:, j], m01[indices, j]) for j in range(10)]
        auc2 = [binary_auc(y[:, j], m02[indices, j]) for j in range(10)]
        differences = [second - first for first, second in zip(auc1, auc2)]
        deltas.append(float(np.mean(differences)))
        for j, difference in enumerate(differences):
            class_deltas[j].append(difference)
    if len(deltas) < 100:
        raise ValueError("Too few valid patient bootstrap draws")
    return {"method": "paired patient-cluster percentile bootstrap; same resampled patients and all their Studies for M01 and M02",
            "patient_count": len(patients), "seed": seed, "requested_replicates": replicates,
            "valid_replicates": len(deltas), "invalid_replicates_excluded": invalid,
            "invalid_rule": "Exclude draw if any of ten classes lacks positive or negative Studies",
            "ci_95_lower": float(np.quantile(deltas, 0.025)),
            "ci_95_upper": float(np.quantile(deltas, 0.975)),
            "per_class_difference_ci_95": [
                {"class_id": j + 1, "lower": float(np.quantile(values, 0.025)),
                 "upper": float(np.quantile(values, 0.975))}
                for j, values in enumerate(class_deltas)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="M02 config")
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--m02-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    m01_run, m02_run, output = (path.resolve() for path in
                                (args.m01_run_dir, args.m02_run_dir, args.output_dir))
    if output == m01_run or m01_run in output.parents:
        parser.error("Comparison must not write into frozen M01")
    if load_config(args.config) != load_config(m02_run / "config.yaml"):
        parser.error("M02 config differs from its runtime copy")
    for run in (m01_run, m02_run):
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            parser.error(f"Incomplete run: {run}")
    m01, y1, subjects1, studies1 = read_study_predictions(
        m01_run / "validation_analysis" / "study_predictions.csv")
    m02, y2, subjects2, studies2 = read_study_predictions(
        m02_run / "validation_analysis" / "study_predictions.csv")
    keys1, keys2 = list(zip(subjects1, studies1)), list(zip(subjects2, studies2))
    if len(keys1) != 859 or len(keys2) != 859 or set(keys1) != set(keys2):
        raise ValueError("M01/M02 Study keys are not the same 859 Studies")
    order = [dict((key, i) for i, key in enumerate(keys2))[key] for key in keys1]
    m02, y2 = m02[order], y2[order]
    if not np.array_equal(y1, y2):
        raise ValueError("M01/M02 Study targets differ")
    scores = {"M01": measures(m01, y1), "M02": measures(m02, y1)}
    for name, run in (("M01", m01_run), ("M02", m02_run)):
        best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
        if abs(scores[name]["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
            raise ValueError(f"{name} Study probabilities disagree with best checkpoint metrics")
    bootstrap = patient_bootstrap(m01, m02, y1, subjects1,
                                  replicates=args.bootstrap_replicates, seed=args.seed)
    difference = scores["M02"]["macro_auc"] - scores["M01"]["macro_auc"]
    bootstrap["point_difference_m02_minus_m01"] = difference
    records = {}
    for name, run in (("M01", m01_run), ("M02", m02_run)):
        records[name] = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    trend = {}
    for name, rows in records.items():
        best_epoch = max(rows, key=lambda row: row["val"]["macro_auc"])["epoch"]
        stop_epoch = rows[-1]["epoch"]
        trend[name] = {"best_epoch": best_epoch, "stop_epoch": stop_epoch,
                       "best_auc": rows[best_epoch - 1]["val"]["macro_auc"],
                       "final_auc": rows[-1]["val"]["macro_auc"],
                       "auc_drop_best_to_final": rows[best_epoch - 1]["val"]["macro_auc"] - rows[-1]["val"]["macro_auc"],
                       "best_train_loss": rows[best_epoch - 1]["train_loss"],
                       "final_train_loss": rows[-1]["train_loss"],
                       "total_epoch_hours": sum(row["elapsed_seconds"] for row in rows) / 3600,
                       "peak_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in rows),
                       "peak_gpu_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in rows)}
    result = {"study_count": 859, "patient_count": len(set(subjects1)), "scores": scores,
              "paired_bootstrap": bootstrap, "epoch_trend": trend,
              "interpretation_scope": "M02 changes the entire conservative augmentation strategy; individual operators are not isolated"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_class_auc.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["class_id", "class_name", "positive_studies", "negative_studies",
                  "m01_auc", "m02_auc", "auc_difference", "m01_f1_at_0_5", "m02_f1_at_0_5"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row1, row2 in zip(scores["M01"]["per_class"], scores["M02"]["per_class"]):
            writer.writerow({"class_id": row1["class_id"], "class_name": row1["class_name"],
                             "positive_studies": row1["positive_studies"],
                             "negative_studies": row1["negative_studies"],
                             "m01_auc": row1["auc"], "m02_auc": row2["auc"],
                             "auc_difference": row2["auc"] - row1["auc"],
                             "m01_f1_at_0_5": row1["f1_at_0_5"], "m02_f1_at_0_5": row2["f1_at_0_5"]})
    lines = ["# M01 versus M02", "",
             "All 859 Study IDs and all ten targets match exactly. Scores were independently recomputed from the best checkpoints' saved Study probabilities.",
             f"M01 Macro-AUC {scores['M01']['macro_auc']:.5f}; M02 Macro-AUC {scores['M02']['macro_auc']:.5f}; "
             f"M02−M01 {difference:+.5f} (paired patient 95% CI "
             f"[{bootstrap['ci_95_lower']:+.5f}, {bootstrap['ci_95_upper']:+.5f}]).",
             f"M01 Macro-F1@0.5 {scores['M01']['macro_f1_at_0_5']:.5f}; M02 {scores['M02']['macro_f1_at_0_5']:.5f}.",
             f"Bootstrap: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']} valid draws from {bootstrap['patient_count']} patients.",
             "", "| Class | M01 AUC | M02 AUC | Difference |", "|---|---:|---:|---:|"]
    for one, two in zip(scores["M01"]["per_class"], scores["M02"]["per_class"]):
        lines.append(f"| {one['class_id']} {one['class_name']} | {one['auc']:.5f} | "
                     f"{two['auc']:.5f} | {two['auc'] - one['auc']:+.5f} |")
    lines += ["", "Focused class 1/4/8 paired patient confidence intervals for M02−M01:"]
    for j in (0, 3, 7):
        interval = bootstrap["per_class_difference_ci_95"][j]
        lines.append(f"- {j + 1} {CLASS_NAMES[j]}: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]")
    lines += ["", f"M01 best/stop epoch {trend['M01']['best_epoch']}/{trend['M01']['stop_epoch']}; "
              f"M02 {trend['M02']['best_epoch']}/{trend['M02']['stop_epoch']}.",
              "Overfitting trend, training time and GPU peaks are in `comparison.json`.",
              "The treatment is the combined augmentation strategy; this comparison cannot attribute a change to one operator."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"m01_auc": scores["M01"]["macro_auc"], "m02_auc": scores["M02"]["macro_auc"],
                      "difference": difference, "ci": [bootstrap["ci_95_lower"], bootstrap["ci_95_upper"]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
