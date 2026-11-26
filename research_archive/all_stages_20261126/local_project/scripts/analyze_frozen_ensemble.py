"""Evaluate one prespecified equal-weight fusion of frozen Study predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def read_aligned(run_dirs):
    predictions, targets, subjects, studies = {}, None, None, None
    evidence = {}
    for name, run in run_dirs.items():
        source = run / "validation_analysis" / "study_predictions.csv"
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            fields = tuple(csv.DictReader(handle).fieldnames or ())
        if not {"Subject_id", "Study_id", "image_count", *[f"probability_{i}" for i in range(1, 11)]}.issubset(fields):
            raise ValueError(f"Missing Study-level columns: {source}")
        if "Image_id" in fields or "Dicom_id" in fields:
            raise ValueError(f"Unexpected image-level key in Study table: {source}")
        p, y, s, st = read_study_predictions(source)
        keys = list(zip(s, st))
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate validation Study keys in {name}")
        if targets is None:
            targets, subjects, studies = y, s, st
        else:
            reference = list(zip(subjects, studies))
            if set(keys) != set(reference):
                raise ValueError(f"Validation Study keys differ for {name}")
            index = {key: position for position, key in enumerate(keys)}
            order = [index[key] for key in reference]
            p, y = p[order], y[order]
            if not np.array_equal(y, targets):
                raise ValueError(f"Validation labels differ for {name}")
        predictions[name] = p
        evidence[name] = {"source": str(source), "rows": len(p), "one_row_per_study": len(set(keys)) == len(keys),
                          "has_image_identifier": False, "has_image_count_only": True}
    if len(targets) != 859:
        raise ValueError("Expected exactly 859 aligned validation Studies")
    return predictions, targets, subjects, evidence


def macro_and_class_auc(probabilities, targets):
    aucs = [binary_auc(targets[:, index], probabilities[:, index]) for index in range(10)]
    if any(value is None for value in aucs):
        raise ValueError("A class lacks positive or negative Studies")
    return float(np.mean(aucs)), aucs


def paired_bootstrap_all(predictions, targets, subjects, *, replicates=2000, seed=42):
    groups = {}
    for index, subject in enumerate(subjects):
        groups.setdefault(subject, []).append(index)
    patient_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in predictions if name != "B01"}
    invalid = 0
    for _ in range(replicates):
        sample = rng.integers(0, len(patient_indices), len(patient_indices))
        indices = np.concatenate([patient_indices[index] for index in sample])
        if any(targets[indices, class_index].min() == targets[indices, class_index].max()
               for class_index in range(10)):
            invalid += 1
            continue
        baseline = macro_and_class_auc(predictions["B01"][indices], targets[indices])[0]
        for name in draws:
            value = macro_and_class_auc(predictions[name][indices], targets[indices])[0]
            draws[name].append(value - baseline)
    if not draws or min(map(len, draws.values())) < 100:
        raise ValueError("Too few valid bootstrap resamples")
    results = {"B01": {"lower": 0.0, "upper": 0.0}}
    for name, values in draws.items():
        low, high = np.quantile(values, (0.025, 0.975))
        results[name] = {"lower": float(low), "upper": float(high)}
    return {"patient_count": len(patient_indices), "requested_replicates": replicates,
            "valid_replicates": replicates - invalid, "invalid_replicates_excluded": invalid,
            "invalid_rule": "Skip a resample if any class lacks positives or negatives; never impute AUC",
            "method": "paired patient-cluster bootstrap; use identical patient draws for every model",
            "seed": seed, "delta_ci_95": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="B01 runtime config")
    parser.add_argument("--b01-run-dir", required=True, type=Path)
    parser.add_argument("--b02-run-dir", required=True, type=Path)
    parser.add_argument("--b03-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = {"B01": args.b01_run_dir.resolve(), "B02": args.b02_run_dir.resolve(),
            "B03": args.b03_run_dir.resolve()}
    output = args.output_dir.resolve()
    if any(output == run or run in output.parents for run in runs.values()):
        parser.error("Ensemble analysis must be outside all frozen experiment directories")
    if load_config(args.config) != load_config(runs["B01"] / "config.yaml"):
        parser.error("B01 analysis config differs from frozen runtime config")
    for name, run in runs.items():
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            parser.error(f"Frozen run is not complete: {name}")
    predictions, targets, subjects, evidence = read_aligned(runs)
    predictions["B01_B02_B03_equal"] = sum(predictions.values()) / 3.0
    results, class_rows = {}, []
    for name, p in predictions.items():
        macro, aucs = macro_and_class_auc(p, targets)
        results[name] = {"macro_auc": macro, "per_class_auc": aucs}
        for index, auc in enumerate(aucs):
            class_rows.append({"scheme": name, "class_id": index + 1, "class_name": CLASS_NAMES[index],
                               "positive_studies": int(targets[:, index].sum()),
                               "negative_studies": int((targets[:, index] == 0).sum()), "auc": auc})
    for name in runs:
        recorded = json.loads((runs[name] / "best_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auc"]
        if abs(results[name]["macro_auc"] - recorded) > 1e-8:
            raise ValueError(f"Saved Study predictions differ from {name} best AUC")
    bootstrap = paired_bootstrap_all(predictions, targets, subjects,
                                     replicates=args.bootstrap_replicates, seed=args.seed)
    for name, result in results.items():
        result["delta_vs_b01"] = result["macro_auc"] - results["B01"]["macro_auc"]
        result["delta_ci_95"] = bootstrap["delta_ci_95"][name]
    correlations = {}
    for left, right in (("B01", "B02"), ("B01", "B03"), ("B02", "B03")):
        values = [float(np.corrcoef(predictions[left][:, j], predictions[right][:, j])[0, 1]) for j in range(10)]
        correlations[f"{left}_{right}"] = {"mean_per_class_pearson": float(np.mean(values)), "per_class_pearson": values}
    report = {"study_count": len(targets), "image_prediction_availability": evidence,
              "mean_max_image_fusion": "not_evaluable_no_saved_image_probabilities",
              "prespecified_ensemble": "equal arithmetic mean of three aligned Study-level probability vectors",
              "schemes": results, "paired_bootstrap": bootstrap, "prediction_correlations": correlations}
    output.mkdir(parents=True, exist_ok=True)
    frozen_manifest = {}
    for name, run in runs.items():
        frozen_manifest[name] = {}
        for relative in ("config.yaml", "metrics.jsonl", "best_model.pt", "best_metrics.json",
                         "validation_analysis/study_predictions.csv"):
            path = run / relative
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            frozen_manifest[name][relative] = {"bytes": path.stat().st_size, "sha256": digest}
    (output / "frozen_manifest.json").write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ensemble_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "class_auc.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    lines = ["# Frozen-model validation ensemble", "",
             "All schemes use exactly 859 validation Studies aligned by Subject_id and Study_id, with identical targets.",
             "The saved files contain one row per Study with image_count and ten Study probabilities, but no image identifiers or per-image probabilities. Image-level Mean versus Max cannot be tested from these artifacts; no image probabilities were inferred.",
             "Only one prespecified fusion was evaluated: the unweighted mean of B01, B02, and B03 Study probabilities. No weight search was performed.",
             "", "| Scheme | Macro-AUC | Delta vs B01 | Paired patient-bootstrap 95% CI |",
             "|---|---:|---:|---:|"]
    for name, value in results.items():
        ci = value["delta_ci_95"]
        lines.append(f"| {name} | {value['macro_auc']:.5f} | {value['delta_vs_b01']:+.5f} | "
                     f"[{ci['lower']:+.5f}, {ci['upper']:+.5f}] |")
    lines += ["", f"Bootstrap: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']} valid patient resamples; "
              f"{bootstrap['invalid_replicates_excluded']} excluded for absent class positives/negatives.",
              "", "| Class ID | Label | B01 | B02 | B03 | Equal fusion |",
              "|---:|---|---:|---:|---:|---:|"]
    for index, label in enumerate(CLASS_NAMES):
        lines.append(f"| {index + 1} | {label} | " + " | ".join(
            f"{results[name]['per_class_auc'][index]:.5f}"
            for name in ("B01", "B02", "B03", "B01_B02_B03_equal")) + " |")
    lines += ["", "Full-precision AUCs are in `class_auc.csv`. Pairwise class-wise probability correlations are in `ensemble_analysis.json`."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"schemes": {name: {key: value[key] for key in ("macro_auc", "delta_vs_b01", "delta_ci_95")}
                                  for name, value in results.items()},
                      "mean_max_image_fusion": report["mean_max_image_fusion"]}, indent=2))


if __name__ == "__main__":
    main()
