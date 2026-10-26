"""Compare three prespecified M01 schemes on the same 859 validation Studies."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def read_aligned(runs: dict[str, Path]):
    predictions = {}
    reference_keys = reference_targets = subjects = None
    for name, run in runs.items():
        p, y, s, st = read_study_predictions(run / "validation_analysis" / "study_predictions.csv")
        keys = list(zip(s, st))
        if reference_keys is None:
            reference_keys, reference_targets, subjects = keys, y, s
        else:
            if set(keys) != set(reference_keys):
                raise ValueError(f"Validation Study IDs differ for {name}")
            by_key = {key: index for index, key in enumerate(keys)}
            order = [by_key[key] for key in reference_keys]
            p, y = p[order], y[order]
            if not np.array_equal(y, reference_targets):
                raise ValueError(f"Validation labels differ for {name}")
        predictions[name] = p
    if len(reference_keys) != 859:
        raise ValueError(f"Expected 859 validation Studies; got {len(reference_keys)}")
    return predictions, reference_targets, subjects


def score(probabilities, targets):
    aucs = [binary_auc(targets[:, j], probabilities[:, j]) for j in range(10)]
    if any(value is None for value in aucs):
        raise ValueError("All ten classes need positive and negative validation Studies")
    positives = probabilities >= 0.5
    f1 = []
    for j in range(10):
        truth, prediction = targets[:, j] == 1, positives[:, j]
        tp = int((truth & prediction).sum())
        fp = int((~truth & prediction).sum())
        fn = int((truth & ~prediction).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"macro_auc": float(np.mean(aucs)), "per_class_auc": aucs,
            "macro_f1_at_0_5": float(np.mean(f1)),
            "zero_prediction_studies_at_0_5": int((positives.sum(axis=1) == 0).sum())}


def paired_patient_bootstrap(predictions, targets, subjects, *, replicates, seed):
    groups = {}
    for index, subject in enumerate(subjects):
        groups.setdefault(subject, []).append(index)
    patient_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    comparisons = {name: {reference: [] for reference in ("B01", "Frozen_equal")}
                   for name in ("M01", "M01_B01_equal", "M01_Frozen_equal_half")}
    rng = np.random.default_rng(seed)
    invalid = 0
    for _ in range(replicates):
        draw = rng.integers(0, len(patient_indices), size=len(patient_indices))
        indices = np.concatenate([patient_indices[index] for index in draw])
        sample_targets = targets[indices]
        if any(sample_targets[:, j].min() == sample_targets[:, j].max() for j in range(10)):
            invalid += 1
            continue
        auc = {name: float(np.mean([binary_auc(sample_targets[:, j], p[indices, j]) for j in range(10)]))
               for name, p in predictions.items()}
        for name, references in comparisons.items():
            for reference, values in references.items():
                values.append(auc[name] - auc[reference])
    if replicates - invalid < 100:
        raise ValueError("Too few valid patient bootstrap draws")
    intervals = {name: {reference: {"lower": float(np.quantile(values, 0.025)),
                                    "upper": float(np.quantile(values, 0.975))}
                        for reference, values in references.items()}
                 for name, references in comparisons.items()}
    return {"method": "paired patient-cluster bootstrap; identical draws for all schemes",
            "patient_count": len(patient_indices), "seed": seed,
            "requested_replicates": replicates, "valid_replicates": replicates - invalid,
            "invalid_replicates_excluded": invalid,
            "invalid_rule": "Exclude draws lacking positives or negatives for any class; never impute AUC",
            "delta_ci_95": intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="M01 runtime config")
    for name in ("b01", "b02", "b03", "m01"):
        parser.add_argument(f"--{name}-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = {name.upper(): getattr(args, f"{name}-run-dir".replace("-", "_")).resolve()
            for name in ("b01", "b02", "b03", "m01")}
    output = args.output_dir.resolve()
    if any(output == runs[name] or runs[name] in output.parents for name in ("B01", "B02", "B03")):
        parser.error("Comparison output cannot be inside a frozen B01/B02/B03 directory")
    if load_config(args.config) != load_config(runs["M01"] / "config.yaml"):
        parser.error("M01 analysis config differs from its runtime copy")
    for name, run in runs.items():
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            parser.error(f"Run is not complete: {name}")
    predictions, targets, subjects = read_aligned(runs)
    predictions["Frozen_equal"] = (predictions["B01"] + predictions["B02"] + predictions["B03"]) / 3
    predictions["M01_B01_equal"] = (predictions["M01"] + predictions["B01"]) / 2
    predictions["M01_Frozen_equal_half"] = (predictions["M01"] + predictions["Frozen_equal"]) / 2
    scores = {name: score(p, targets) for name, p in predictions.items()}
    for name, run in runs.items():
        expected = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auc"]
        if abs(scores[name]["macro_auc"] - expected) > 1e-8:
            raise ValueError(f"{name} saved predictions disagree with best checkpoint metrics")
    if abs(scores["Frozen_equal"]["macro_auc"] - 0.8026744715463039) > 1e-8:
        raise ValueError("Frozen three-model fusion differs from its prior result")
    bootstrap = paired_patient_bootstrap(predictions, targets, subjects,
                                         replicates=args.bootstrap_replicates, seed=args.seed)
    requested = ("M01", "M01_B01_equal", "M01_Frozen_equal_half")
    for name in requested:
        scores[name]["delta_vs_b01"] = scores[name]["macro_auc"] - scores["B01"]["macro_auc"]
        scores[name]["delta_vs_frozen_equal"] = scores[name]["macro_auc"] - scores["Frozen_equal"]["macro_auc"]
        scores[name]["delta_ci_95"] = bootstrap["delta_ci_95"][name]
    output.mkdir(parents=True, exist_ok=True)
    report = {"study_count": len(targets), "schemes": scores, "bootstrap": bootstrap,
              "scheme_definitions": {"M01": "M01 Study probabilities",
                                     "M01_B01_equal": "(M01 + B01) / 2",
                                     "M01_Frozen_equal_half": "(M01 + (B01 + B02 + B03) / 3) / 2"}}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_class_auc.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "positive_studies", "negative_studies",
                                                   *[f"{name}_auc" for name in ("B01", "Frozen_equal", *requested)]])
        writer.writeheader()
        for j, label in enumerate(CLASS_NAMES):
            writer.writerow({"class_id": j + 1, "class_name": label,
                             "positive_studies": int(targets[:, j].sum()),
                             "negative_studies": int((targets[:, j] == 0).sum()),
                             **{f"{name}_auc": scores[name]["per_class_auc"][j]
                                for name in ("B01", "Frozen_equal", *requested)}})
    lines = ["# M01 validation complementarity", "",
             "All schemes use the same 859 Study IDs and identical ten-class targets.",
             "Only the three prespecified M01 schemes were evaluated; no weight search was run.", "",
             "| Scheme | Macro-AUC | Macro-F1@0.5 | Delta vs B01 (95% CI) | Delta vs frozen equal (95% CI) |",
             "|---|---:|---:|---:|---:|"]
    for name in requested:
        result = scores[name]
        ci1, ci2 = result["delta_ci_95"]["B01"], result["delta_ci_95"]["Frozen_equal"]
        lines.append(f"| {name} | {result['macro_auc']:.5f} | {result['macro_f1_at_0_5']:.5f} | "
                     f"{result['delta_vs_b01']:+.5f} [{ci1['lower']:+.5f}, {ci1['upper']:+.5f}] | "
                     f"{result['delta_vs_frozen_equal']:+.5f} [{ci2['lower']:+.5f}, {ci2['upper']:+.5f}] |")
    lines += ["", f"Patient-cluster bootstrap: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']} valid draws; "
              f"{bootstrap['invalid_replicates_excluded']} excluded.",
              "Full ten-class AUCs are in `per_class_auc.csv`."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: scores[name][key] for key in ("macro_auc", "macro_f1_at_0_5",
                                                        "delta_vs_b01", "delta_vs_frozen_equal", "delta_ci_95")}
                      for name in requested}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
