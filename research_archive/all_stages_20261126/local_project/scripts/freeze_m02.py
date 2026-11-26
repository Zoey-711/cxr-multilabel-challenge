"""Write M02 completion report and hash its validated experiment artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if load_config(args.config) != load_config(run / "config.yaml"):
        parser.error("M02 config differs from its runtime copy")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        parser.error("M02 training has not completed")
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((run / "validation_analysis" / "analysis.json").read_text(encoding="utf-8"))
    compared = json.loads((run / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["epoch"] for row in records] != list(range(1, len(records) + 1)) or \
            records[-1]["epoch"] != status["epoch"] or \
            status["best_epoch"] != best["epoch"] or \
            analysis["study_count"] != 859 or \
            abs(analysis["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(compared["scores"]["M02"]["macro_auc"] - analysis["macro_auc"]) > 1e-8:
        raise ValueError("M02 completed artifacts fail validation cross-check")
    with (run / "gpu_monitor.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        monitor = list(csv.DictReader(handle))
    board_peak = max(int(row["gpu_board_used_mib"]) for row in monitor if row["gpu_board_used_mib"])
    runtime_hours = sum(row["elapsed_seconds"] for row in records) / 3600
    peak_allocated = max(row["gpu_peak_allocated_mib"] for row in records)
    peak_reserved = max(row["gpu_peak_reserved_mib"] for row in records)
    amp_skips = sum(row["amp_skipped_steps"] for row in records)
    trend = compared["epoch_trend"]
    delta = compared["paired_bootstrap"]["point_difference_m02_minus_m01"]
    ci = compared["paired_bootstrap"]
    lines = ["# M02 completed training and independent validation", "",
             f"Best epoch {best['epoch']}; stop epoch {status['stop_epoch']} ({status['stop_reason']}); patience {status['early_stopping_patience']}.",
             f"Best Study-level Macro-AUC {analysis['macro_auc']:.5f}; Macro-F1@0.5 {analysis['macro_f1']:.5f}; 859 Studies.",
             f"M02−M01 Macro-AUC {delta:+.5f}; paired patient bootstrap 95% CI "
             f"[{ci['ci_95_lower']:+.5f}, {ci['ci_95_upper']:+.5f}].",
             f"M01 best/stop epoch {trend['M01']['best_epoch']}/{trend['M01']['stop_epoch']}; "
             f"M02 {trend['M02']['best_epoch']}/{trend['M02']['stop_epoch']}.",
             f"M01 AUC decline best to stop {trend['M01']['auc_drop_best_to_final']:.5f}; "
             f"M02 {trend['M02']['auc_drop_best_to_final']:.5f}.",
             f"Epoch runtime sum {runtime_hours:.2f} h; peak GPU allocated/reserved {peak_allocated}/{peak_reserved} MiB; "
             f"board peak {board_peak} MiB; AMP skipped steps {amp_skips}.",
             "Independent validation reloaded best_model.pt and reproduced the logged 859-Study AUC. "
             "Image-level and Study-level probabilities are saved under validation_analysis/.",
             "M02 changes the combined train augmentation strategy; attribution to a single operator is not supported.",
             "", "| ID | Class | M01 AUC | M02 AUC | Difference |", "|---:|---|---:|---:|---:|"]
    for row1, row2 in zip(compared["scores"]["M01"]["per_class"], compared["scores"]["M02"]["per_class"]):
        lines.append(f"| {row1['class_id']} | {row1['class_name']} | {row1['auc']:.5f} | "
                     f"{row2['auc']:.5f} | {row2['auc'] - row1['auc']:+.5f} |")
    (run / "m02_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = ("config.yaml", "seed.txt", "environment.json", "pretrained_source.json",
             "train_stdout.log", "train_stderr.log", "gpu_monitor.csv", "monitor_stdout.log",
             "monitor_stderr.log", "metrics.jsonl", "best_metrics.json", "best_model.pt",
             "last_training.pt", "status.json", "validation_analysis/analysis.json",
             "validation_analysis/class_metrics.csv", "validation_analysis/study_predictions.csv",
             "validation_analysis/image_predictions.csv", "comparison/comparison.json",
             "comparison/per_class_auc.csv", "comparison/report.md", "m02_final_report.md")
    manifest = {}
    for relative in files:
        path = run / relative
        with path.open("rb") as handle:
            manifest[relative] = {"size_bytes": path.stat().st_size,
                                  "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    (run / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best["epoch"], "stop_epoch": status["stop_epoch"],
                      "macro_auc": analysis["macro_auc"], "macro_f1": analysis["macro_f1"],
                      "delta_vs_m01": delta, "ci_95": [ci["ci_95_lower"], ci["ci_95_upper"]],
                      "runtime_hours": runtime_hours, "gpu_board_peak_mib": board_peak}, ensure_ascii=False))


if __name__ == "__main__":
    main()
