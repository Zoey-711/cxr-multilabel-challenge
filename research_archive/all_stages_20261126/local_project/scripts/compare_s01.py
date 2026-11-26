"""Compare frozen M02 and completed S01, plus fixed E03/E04 validation fusion."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compare_m01_m02 import measures, patient_bootstrap
from cxr.data.archive import archive_path, read_split_rows
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}


def stratum_scores(probabilities, targets, mask):
    p, y = probabilities[mask], targets[mask]
    aucs, f1s, per_class = [], [], []
    for j in range(10):
        auc = binary_auc(y[:, j], p[:, j])
        true = y[:, j] == 1
        pred = p[:, j] >= 0.5
        tp, fp, fn = int((true & pred).sum()), int((~true & pred).sum()), int((true & ~pred).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if auc is not None:
            aucs.append(auc)
        f1s.append(f1)
        per_class.append({"class_id": j + 1, "class_name": CLASS_NAMES[j],
                          "positive_studies": int(true.sum()), "negative_studies": int((~true).sum()),
                          "auc": auc, "f1_at_0_5": f1})
    return {"study_count": int(mask.sum()), "macro_auc": float(np.mean(aucs)) if len(aucs) == 10 else None,
            "classes_with_defined_auc": len(aucs), "macro_f1_at_0_5": float(np.mean(f1s)),
            "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Exact S01 runtime config")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("S01 runtime config differs")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("S01 training incomplete")
    frozen = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    m02_frozen = json.loads((RUNS["M02"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    arrays, reference_keys, targets, subjects = {}, None, None, None
    for name, run in {**RUNS, "S01": args.run_dir}.items():
        if name != "S01":
            manifest = m02_frozen if name == "M02" else frozen[name]
            for artifact in ("best_model.pt", "validation_analysis/study_predictions.csv"):
                if sha256(run / artifact) != manifest[artifact]["sha256"]:
                    raise ValueError(f"Frozen {name} source hash differs: {artifact}")
        p, y, subject_ids, study_ids = read_study_predictions(run / "validation_analysis/study_predictions.csv")
        keys = list(zip(subject_ids, study_ids))
        if len(keys) != 859 or len(set(keys)) != 859:
            raise ValueError(f"Expected 859 distinct Studies for {name}")
        if reference_keys is None:
            reference_keys, targets, subjects = keys, y, subject_ids
            order = range(859)
        else:
            lookup = {key: index for index, key in enumerate(keys)}
            if set(lookup) != set(reference_keys):
                raise ValueError(f"Study/Subject alignment differs: {name}")
            order = [lookup[key] for key in reference_keys]
        p, y = p[list(order)], y[list(order)]
        if not np.array_equal(y, targets):
            raise ValueError(f"Labels differ: {name}")
        arrays[name] = p
    e01 = (arrays["B01"] + arrays["B02"] + arrays["B03"]) / 3
    arrays["E03"] = 0.5 * e01 + 0.5 * arrays["M02"]
    arrays["E04"] = 0.5 * e01 + 0.5 * arrays["S01"]
    scores = {name: measures(arrays[name], targets) for name in ("M02", "S01", "E03", "E04")}
    for name, expected in (("E03", 0.8099309507263751), ("M02", 0.7972442723187514)):
        if abs(scores[name]["macro_auc"] - expected) > 1e-10:
            raise ValueError(f"{name} differs from frozen validation AUC")
    best = json.loads((args.run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    recomputed = json.loads((args.run_dir / "validation_analysis" / "analysis.json").read_text(encoding="utf-8"))
    if abs(scores["S01"]["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(recomputed["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
        raise ValueError("Independent S01 validation differs from best checkpoint metrics")
    comparisons = {}
    for baseline, candidate in (("M02", "S01"), ("E03", "E04")):
        paired = patient_bootstrap(arrays[baseline], arrays[candidate], targets, subjects,
                                   replicates=args.bootstrap_replicates, seed=args.seed)
        paired["method"] = "paired patient-cluster percentile bootstrap; same sampled patients and all their Studies for baseline and candidate"
        paired["point_difference"] = scores[candidate]["macro_auc"] - scores[baseline]["macro_auc"]
        comparisons[f"{candidate}_minus_{baseline}"] = paired
    val_rows = read_split_rows(archive_path(load_config(args.config)), "val")
    image_counts = Counter((row["Subject_id"], row["Study_id"]) for row in val_rows)
    group_masks = {"single_image": np.array([image_counts[key] == 1 for key in reference_keys]),
                   "multi_image": np.array([image_counts[key] >= 2 for key in reference_keys])}
    if int(group_masks["single_image"].sum()) != 391 or int(group_masks["multi_image"].sum()) != 468:
        raise ValueError("Official validation image-count strata differ")
    strata = {name: {group: stratum_scores(arrays[name], targets, mask)
                     for group, mask in group_masks.items()} for name in ("M02", "S01")}
    group_gap = {name: strata[name]["multi_image"]["macro_auc"] - strata[name]["single_image"]["macro_auc"]
                 if all(strata[name][group]["macro_auc"] is not None for group in group_masks) else None
                 for name in strata}
    trends = {}
    for name, run in (("M02", RUNS["M02"]), ("S01", args.run_dir)):
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        best_epoch = max(records, key=lambda record: record["val"]["macro_auc"])["epoch"]
        best_record = next(record for record in records if record["epoch"] == best_epoch)
        trends[name] = {"best_epoch": best_epoch, "stop_epoch": records[-1]["epoch"],
                        "best_auc": best_record["val"]["macro_auc"],
                        "final_auc": records[-1]["val"]["macro_auc"],
                        "auc_drop_best_to_stop": best_record["val"]["macro_auc"] - records[-1]["val"]["macro_auc"],
                        "best_train_loss": best_record["train_loss"],
                        "final_train_loss": records[-1]["train_loss"],
                        "epoch_runtime_hours": sum(record["elapsed_seconds"] for record in records) / 3600,
                        "amp_skipped_steps": sum(record["amp_skipped_steps"] for record in records),
                        "peak_gpu_allocated_mib": max(record["gpu_peak_allocated_mib"] for record in records)}
    result = {"study_count": 859, "patient_count": len(set(subjects)), "scores": scores,
              "comparisons": comparisons, "strata": strata,
              "multi_minus_single_macro_auc": group_gap,
              "epoch_trends": trends,
              "interpretation": "Predefined weights only; subgroup class prevalences differ; internal patient holdout was not evaluated"}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_class_auc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", "m02_auc", "s01_auc", "e03_auc", "e04_auc"])
        for j in range(10):
            writer.writerow([j + 1, CLASS_NAMES[j], scores["S01"]["per_class"][j]["positive_studies"],
                             *(scores[name]["per_class"][j]["auc"] for name in ("M02", "S01", "E03", "E04"))])
    lines = ["# S01 versus M02 and fixed E04 versus E03", "",
             "All 859 Studies, 228 patients and ten targets align by Subject_id/Study_id. "
             "E03 = 0.5 E01 + 0.5 M02; E04 = 0.5 E01 + 0.5 S01. No weight search.", "",
             "| Model | Macro-AUC | Macro-F1@0.5 |", "|---|---:|---:|"]
    for name in ("M02", "S01", "E03", "E04"):
        lines.append(f"| {name} | {scores[name]['macro_auc']:.5f} | {scores[name]['macro_f1_at_0_5']:.5f} |")
    for name, paired in comparisons.items():
        lines.append(f"\n{name} Macro-AUC difference {paired['point_difference']:+.5f}; "
                     f"paired patient 95% CI [{paired['ci_95_lower']:+.5f}, {paired['ci_95_upper']:+.5f}] "
                     f"({paired['valid_replicates']}/{paired['requested_replicates']} valid draws).")
    lines.extend(["", f"M02 best/stop Epoch {trends['M02']['best_epoch']}/{trends['M02']['stop_epoch']}, "
                  f"AUC drop {trends['M02']['auc_drop_best_to_stop']:.5f}; "
                  f"S01 {trends['S01']['best_epoch']}/{trends['S01']['stop_epoch']}, "
                  f"drop {trends['S01']['auc_drop_best_to_stop']:.5f}.",
                  "Single- and multi-image Study metrics and ten-class composition are in `comparison.json`. "
                  "The groups differ in disease prevalence, so their raw Macro-AUC gap is descriptive, not a causal effect of view count."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"auc": {name: score["macro_auc"] for name, score in scores.items()},
                      "diff_ci": {name: [value["ci_95_lower"], value["ci_95_upper"]]
                                  for name, value in comparisons.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
