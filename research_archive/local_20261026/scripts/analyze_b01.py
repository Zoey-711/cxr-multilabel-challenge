00
+"""Freeze B01 diagnostics using saved validation predictions only; never load a model."""

import argparse
import json
from pathlib import Path

from cxr.engine.b01_analysis import (epoch_curves_svg, patient_bootstrap_auc, read_epoch_records,
                                     read_study_predictions, threshold_analysis, write_csv)
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output == run_dir or run_dir in output.parents:
        parser.error("Analysis output must be outside the frozen B01 run directory")
    if load_config(args.config) != load_config(run_dir / "config.yaml"):
        parser.error("Config does not match the frozen B01 run")
    if json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        parser.error("B01 is not complete")
    probabilities, targets, subjects, studies = read_study_predictions(
        run_dir / "validation_analysis" / "study_predictions.csv")
    summaries, class_rows = threshold_analysis(probabilities, targets)
    epochs = read_epoch_records(run_dir / "metrics.jsonl")
    best = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    bootstrap = patient_bootstrap_auc(probabilities, targets, subjects,
                                      replicates=args.bootstrap_replicates, seed=args.seed)
    if len(studies) != best["val"]["study_count"] or abs(bootstrap["point_macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
        raise ValueError("Saved predictions do not reproduce B01 best Study-level AUC")
    if summaries[-1]["zero_predicted_class_studies"] != 230:
        raise ValueError("0.5 zero-prediction count differs from frozen B01 analysis")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "threshold_summary.csv", summaries)
    write_csv(output / "threshold_by_class.csv", class_rows)
    write_csv(output / "epoch_metrics.csv", epochs)
    epoch_curves_svg(epochs, output / "epoch_curves.svg")
    (output / "bootstrap_auc.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8")
    peak = max(epochs, key=lambda row: row["auc"])
    report = ["# B01 frozen validation analysis", "",
              "Source: existing validation `study_predictions.csv` and `metrics.jsonl`; no model inference or training.",
              f"Study count: {len(studies)}; patient count: {bootstrap['patient_count']}.",
              f"Best epoch: {best['epoch']}; Study-level Macro-AUC: {bootstrap['point_macro_auc']:.5f}; "
              f"patient-bootstrap 95% CI: [{bootstrap['ci_95_percentile_lower']:.5f}, {bootstrap['ci_95_percentile_upper']:.5f}].",
              f"Bootstrap: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']} valid replicates; "
              f"{bootstrap['invalid_replicates_excluded']} excluded for a class missing positives or negatives.",
              "All studies of a sampled patient stay together, including repeat studies; duplicates from resampling are retained.",
              "", "## Threshold sensitivity", "",
              "| Threshold | Macro-F1 | Macro-Precision | Macro-Recall | Zero-prediction Studies |",
              "|---:|---:|---:|---:|---:|"]
    for row in summaries:
        report.append(f"| {row['threshold']:.1f} | {row['macro_f1']:.5f} | {row['macro_precision']:.5f} | "
                      f"{row['macro_recall']:.5f} | {row['zero_predicted_class_studies']}/{row['study_count']} |")
    report.extend(["", "Precision and Recall are unweighted means of ten per-class values; undefined precision from zero predictions is set to 0.",
                   "Each class's positive/negative count, threshold metrics, and probability p0/p10/p25/p50/p75/p90/p100 are in `threshold_by_class.csv`.",
                   "", "## Epoch trajectory", "",
                   f"AUC peaks at epoch {peak['epoch']} ({peak['auc']:.5f}); the first lower AUC is epoch "
                   f"{peak['epoch']+1} ({epochs[peak['epoch']]['auc']:.5f}).",
                   f"By epoch 7 AUC falls to {epochs[6]['auc']:.5f} while training loss falls from "
                   f"{peak['loss']:.5f} to {epochs[6]['loss']:.5f}; this is the first clear sustained overfitting signal.",
                   f"At epoch {epochs[-1]['epoch']}: AUC {epochs[-1]['auc']:.5f}, loss {epochs[-1]['loss']:.5f}.",
                   "See `epoch_curves.svg` and `epoch_metrics.csv` for every epoch.", "",
                   "Threshold results describe the validation set only. The competition's primary AUC does not depend on the threshold."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best["epoch"], "bootstrap": bootstrap,
                      "thresholds": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
