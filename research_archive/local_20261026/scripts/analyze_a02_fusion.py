"""Evaluate the three prespecified validation-only A02 fusion schemes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_m01_m02 import measures, patient_bootstrap
from cxr.engine.b01_analysis import read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M01": Path("outputs/experiments/m01_convnext_tiny_384_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen M02 runtime config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/a02_fixed_fusions_20261017"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"A02 output already exists: {args.output_dir}")
    if load_config(args.config) != load_config(RUNS["M02"] / "config.yaml"):
        raise ValueError("Frozen M02 config mismatch")
    frozen_trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    source_hashes, predictions = {}, {}
    reference_keys, targets, subjects = None, None, None
    for name, run in RUNS.items():
        frozen = frozen_trio[name] if name in frozen_trio else json.loads(
            (run / "frozen_manifest.json").read_text(encoding="utf-8"))
        relative = "validation_analysis/study_predictions.csv"
        path = run / relative
        if sha256(path) != frozen[relative]["sha256"]:
            raise ValueError(f"Frozen validation prediction hash differs: {name}")
        if sha256(run / "best_model.pt") != frozen["best_model.pt"]["sha256"]:
            raise ValueError(f"Frozen best checkpoint hash differs: {name}")
        source_hashes[name] = {"study_predictions_sha256": sha256(path),
                               "best_checkpoint_sha256": sha256(run / "best_model.pt")}
        p, y, subject_ids, study_ids = read_study_predictions(path)
        keys = list(zip(subject_ids, study_ids))
        if len(keys) != 859 or len(set(keys)) != 859 or len(set(study_ids)) != 859:
            raise ValueError(f"Expected 859 unique Study IDs: {name}")
        if reference_keys is None:
            reference_keys, targets, subjects = keys, y, subject_ids
            order = list(range(859))
        else:
            lookup = {key: index for index, key in enumerate(keys)}
            if set(lookup) != set(reference_keys):
                raise ValueError(f"Subject/Study mapping differs: {name}")
            order = [lookup[key] for key in reference_keys]
        p, y = p[order], y[order]
        if not np.array_equal(y, targets):
            raise ValueError(f"Study labels differ: {name}")
        best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
        if abs(measures(p, y)["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
            raise ValueError(f"Frozen {name} validation metric mismatch")
        predictions[name] = p
    # No separate E01 validation CSV was frozen: reconstruct its exact specified
    # equal mean from hash-verified B01/B02/B03 Study probabilities.
    e01 = (predictions["B01"] + predictions["B02"] + predictions["B03"]) / 3
    schemes = {"E01": e01, "E01_M01_equal": 0.5 * e01 + 0.5 * predictions["M01"],
               "E01_M02_equal": 0.5 * e01 + 0.5 * predictions["M02"]}
    scores = {name: measures(p, targets) for name, p in schemes.items()}
    if abs(scores["E01"]["macro_auc"] - 0.8026744715463039) > 1e-12:
        raise ValueError("Reconstructed E01 differs from frozen validation result")
    frozen_e02 = json.loads((RUNS["M01"] / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    if abs(scores["E01_M01_equal"]["macro_auc"] -
           frozen_e02["schemes"]["M01_Frozen_equal_half"]["macro_auc"]) > 1e-12:
        raise ValueError("A02 E01/M01 differs from frozen E02 validation result")
    comparisons = {}
    for name in ("E01_M01_equal", "E01_M02_equal"):
        bootstrap = patient_bootstrap(e01, schemes[name], targets, subjects,
                                      replicates=args.bootstrap_replicates, seed=args.seed)
        bootstrap["method"] = "paired patient-cluster percentile bootstrap; same resampled patients and all their Studies for E01 and the candidate fusion"
        comparisons[name] = {"delta_macro_auc_vs_e01": scores[name]["macro_auc"] - scores["E01"]["macro_auc"],
                             "paired_patient_bootstrap": bootstrap}
    result = {"study_count": 859, "patient_count": len(set(subjects)),
              "source_probabilities": "frozen Study-level 10-class CSVs; E01 reconstructed as exact B01/B02/B03 mean",
              "source_sha256": source_hashes,
              "schemes": {"E01": "(B01+B02+B03)/3",
                          "E01_M01_equal": "0.5*E01+0.5*M01",
                          "E01_M02_equal": "0.5*E01+0.5*M02"},
              "scores": scores, "comparisons_vs_e01": comparisons,
              "analysis_scope": "prespecified validation-only schemes; no weight search or test labels"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "a02_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_class_auc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", "negative_studies",
                         "e01_auc", "m01_auc", "m02_auc", "e01_m01_equal_auc", "e01_m02_equal_auc"])
        m01_scores = measures(predictions["M01"], targets)["per_class"]
        m02_scores = measures(predictions["M02"], targets)["per_class"]
        for j in range(10):
            row = scores["E01"]["per_class"][j]
            writer.writerow([j + 1, row["class_name"], row["positive_studies"], row["negative_studies"],
                             row["auc"], m01_scores[j]["auc"], m02_scores[j]["auc"],
                             scores["E01_M01_equal"]["per_class"][j]["auc"],
                             scores["E01_M02_equal"]["per_class"][j]["auc"]])
    lines = ["# A02 fixed validation fusions", "",
             "All 859 Studies and 228 patients align by Subject_id and Study_id; all ten labels match.",
             "No standalone E01 validation probability file was frozen. E01 was reconstructed from the three SHA-verified frozen DenseNet Study probability files and reproduced its recorded Macro-AUC exactly.",
             "", "| Scheme | Macro-AUC | Macro-F1@0.5 | Delta vs E01 | Paired patient 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for name in schemes:
        score = scores[name]
        if name == "E01":
            delta, ci = 0.0, "[0, 0]"
        else:
            item = comparisons[name]
            delta = item["delta_macro_auc_vs_e01"]
            pair = item["paired_patient_bootstrap"]
            ci = f"[{pair['ci_95_lower']:+.5f}, {pair['ci_95_upper']:+.5f}]"
        lines.append(f"| {name} | {score['macro_auc']:.5f} | {score['macro_f1_at_0_5']:.5f} | {delta:+.5f} | {ci} |")
    lines += ["", "Detailed ten-class AUCs are in `per_class_auc.csv` and `a02_results.json`.",
              "These are validation-only fixed fusions; no test-ranking-based selection or weight search was performed."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"macro_auc": {name: scores[name]["macro_auc"] for name in schemes},
                      "macro_f1": {name: scores[name]["macro_f1_at_0_5"] for name in schemes},
                      "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
