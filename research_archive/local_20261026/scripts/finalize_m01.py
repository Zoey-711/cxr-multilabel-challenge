"""Wait for M01 completion, then independently validate, compare, and freeze it."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def run_script(script, args, log):
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, str(script), *args], stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    for name in ("b01", "b02", "b03", "m01"):
        parser.add_argument(f"--{name}-run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    runs = {name.upper(): getattr(args, f"{name}_run_dir").resolve()
            for name in ("b01", "b02", "b03", "m01")}
    m01 = runs["M01"]
    if load_config(args.config) != load_config(m01 / "config.yaml"):
        parser.error("M01 runtime config differs from requested config")
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    while True:
        status = json.loads((m01 / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"M01 training failed: {status.get('error')}")
        time.sleep(args.poll_seconds)

    scripts = Path(__file__).resolve().parent
    run_script(scripts / "analyze_validation.py",
               ["--config", str(args.config), "--run-dir", str(m01)],
               m01 / "validation_analysis_stdout.log")
    comparison = m01 / "comparison"
    run_script(scripts / "compare_m01.py",
               ["--config", str(args.config),
                "--b01-run-dir", str(runs["B01"]), "--b02-run-dir", str(runs["B02"]),
                "--b03-run-dir", str(runs["B03"]), "--m01-run-dir", str(m01),
                "--output-dir", str(comparison), "--bootstrap-replicates", "2000", "--seed", "42"],
               m01 / "comparison_stdout.log")

    records = [json.loads(line) for line in (m01 / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    best = json.loads((m01 / "best_metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((m01 / "validation_analysis" / "analysis.json").read_text(encoding="utf-8"))
    compared = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    if not records or records[-1]["epoch"] != status["epoch"] or best["epoch"] > status["epoch"]:
        raise ValueError("M01 epoch history is incomplete")
    if abs(analysis["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
        raise ValueError("M01 independent validation differs from best checkpoint log")
    if analysis["study_count"] != 859:
        raise ValueError("M01 independent validation did not cover 859 Studies")
    files = ("config.yaml", "seed.txt", "environment.json", "pretrained_source.json",
             "train_stdout.log", "train_stderr.log", "gpu_monitor.csv",
             "metrics.jsonl", "best_metrics.json", "best_model.pt", "last_training.pt",
             "validation_analysis/analysis.json", "validation_analysis/study_predictions.csv",
             "validation_analysis/image_predictions.csv", "comparison/comparison.json",
             "comparison/per_class_auc.csv")
    manifest = {}
    for relative in files:
        path = m01 / relative
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest[relative] = {"size_bytes": path.stat().st_size, "sha256": sha}
    (m01 / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    hours = sum(record["elapsed_seconds"] for record in records) / 3600
    peak = max(record["gpu_peak_allocated_mib"] for record in records)
    skipped = sum(record["amp_skipped_steps"] for record in records)
    lines = ["# M01 completed training and validation", "",
             f"Best epoch: {best['epoch']}; stopped after epoch {status['epoch']}.",
             f"Best Study-level Macro-AUC: {analysis['macro_auc']:.5f}; Macro-F1@0.5: {analysis['macro_f1']:.5f}.",
             f"Best-epoch training BCE: {best['train_loss']:.5f}.",
             f"Recorded epoch runtime: {hours:.2f} hours; peak allocated GPU memory: {peak} MiB; AMP skipped optimizer steps: {skipped}.",
             f"Best checkpoint size: {manifest['best_model.pt']['size_bytes']} bytes.",
             "Independent validation reloaded best_model.pt, covered all 859 Studies, and matched training AUC.",
             "Training configuration, checkpoint, predictions, and comparison results are hashed in frozen_manifest.json.",
             "The first post-training analysis used inference batch 16 and failed its exact AUC cross-check under FP16. "
             "Rerunning independent inference with the training validation batch 8 matched the recorded AUC exactly; "
             "the checkpoint and training run were unchanged.",
             "", "| ID | Class | M01 AUC |", "|---:|---|---:|"]
    lines.extend(f"| {j + 1} | {name} | {analysis['per_class'][j]['auc']:.5f} |"
                 for j, name in enumerate(CLASS_NAMES))
    lines += ["", "See comparison/report.md for the three prespecified complementarity schemes and paired bootstrap intervals."]
    (m01 / "m01_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_path = m01 / "m01_final_report.md"
    with report_path.open("rb") as handle:
        manifest["m01_final_report.md"] = {"size_bytes": report_path.stat().st_size,
                                           "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    (m01 / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best["epoch"], "macro_auc": analysis["macro_auc"],
                      "macro_f1": analysis["macro_f1"], "epoch_count": len(records),
                      "runtime_hours": hours, "peak_allocated_mib": peak,
                      "m01_b01_auc": compared["schemes"]["M01_B01_equal"]["macro_auc"],
                      "m01_frozen_auc": compared["schemes"]["M01_Frozen_equal_half"]["macro_auc"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
