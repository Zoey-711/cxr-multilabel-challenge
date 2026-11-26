"""Validate and freeze N01 raw/EMA checkpoints, metrics and comparison files."""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from cxr.utils.config import load_config


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "frozen_manifest.json").exists():
        raise FileExistsError("N01 already frozen")
    config = load_config(args.config)
    if config != load_config(run / "config.yaml") or config["model"]["name"] != "efficientnet_v2_s" or \
            config["training"]["early_stopping_patience"] != 6:
        raise ValueError("N01 runtime configuration differs")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if status["state"] != "complete" or len(rows) != status["stop_epoch"] or len(rows) > 30 or \
            [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("N01 training state or epoch logs incomplete")
    expected_groups = 1897
    if any(row["optimizer_steps"] + row["amp_skipped_steps"] != expected_groups or
           row["study_sampling"] is not None for row in rows):
        raise ValueError("N01 optimizer groups or sampling differ from M02")
    if rows[-1]["ema_updates"] != sum(row["optimizer_steps"] for row in rows):
        raise ValueError("EMA was not updated exactly once per successful optimizer step")
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    best_ema = json.loads((run / "best_ema_metrics.json").read_text(encoding="utf-8"))
    raw = json.loads((run / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    ema = json.loads((run / "ema_validation_analysis/analysis.json").read_text(encoding="utf-8"))
    comparison = json.loads((run / "comparison/comparison.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run / "inference_benchmark.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "pretrained_source.json").read_text(encoding="utf-8"))
    if (status["best_epoch"] != best["epoch"] or
            best_ema["ema_best_epoch"] != best_ema["epoch"] or
            raw["study_count"] != 859 or ema["study_count"] != 859 or
            abs(raw["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or
            abs(ema["macro_auc"] - best_ema["ema_val"]["macro_auc"]) > 1e-8 or
            abs(comparison["scores"]["N01"]["macro_auc"] - raw["macro_auc"]) > 1e-8 or
            abs(comparison["scores"]["N01_EMA"]["macro_auc"] - ema["macro_auc"]) > 1e-8 or
            comparison["fusion"]["E08"] != "0.5*E03+0.5*N01" or
            benchmark["checkpoint_total_bytes_e08"] > 500_000_000 or
            provenance["sha256"] != config["model"]["pretrained_sha256"]):
        raise ValueError("N01 frozen-artifact cross-check failed")
    scores = comparison["scores"]
    intervals = comparison["bootstrap"]["differences"]
    lines = ["# N01 EfficientNetV2-S: independent validation and fixed fusion", "",
             "All scores below use the same 859 official validation Studies and 228 patients. "
             "The Phase 15 internal Holdout was not read or used.", "",
             "| Scheme | Study Macro-AUC | Macro-AP | Macro-F1@0.5 |",
             "|---|---:|---:|---:|"]
    for name in ("N01", "N01_EMA", "E03", "E07", "E08", "E08_EMA"):
        item = scores[name]
        lines.append(f"| {name} | {item['macro_auc']:.5f} | {item['macro_ap']:.5f} | "
                     f"{item['macro_f1_at_0_5']:.5f} |")
    lines.extend(["", "## Paired patient bootstrap", "",
                  f"{comparison['bootstrap']['valid_replicates']} valid draws; "
                  f"{comparison['bootstrap']['invalid_resamples']} invalid draws excluded. "
                  "The same sampled patients and all their Studies were used for each paired difference.", ""])
    for name, item in intervals.items():
        ci = item["ci95"]
        lines.append(f"- {name}: {item['point']:+.5f}; 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}].")
    lines.extend(["", "Raw and EMA variants share a training run and validation set. Their multiple "
                  "comparisons are correlated, not independent significance tests.", "",
                  "## Training and deployment", "",
                  f"Raw best epoch {best['epoch']}; EMA best epoch {best_ema['epoch']}; "
                  f"stop epoch {status['stop_epoch']} ({status['stop_reason']}). "
                  f"EMA successful-step updates {rows[-1]['ema_updates']}; "
                  f"total AMP skips {sum(row['amp_skipped_steps'] for row in rows)}.",
                  f"Training epoch time {sum(row['elapsed_seconds'] for row in rows)/3600:.2f} GPU hours; "
                  f"peak PyTorch allocated memory {max(row['gpu_peak_allocated_mib'] for row in rows)} MiB.",
                  f"E08 five inference checkpoints total {benchmark['checkpoint_total_bytes_e08']:,} bytes "
                  f"(< {benchmark['checkpoint_limit_bytes']:,} bytes).",
                  f"Local RTX 4060 Laptop N01 single-image mean latency "
                  f"{benchmark['results']['N01_raw']['latency_ms_mean']:.1f} ms; "
                  f"five-model E08 mean latency {benchmark['results']['E08_all_five_models']['latency_ms_mean']:.1f} ms. "
                  "Neither is an official T4 measurement.", "",
                  "Ten-class AUC/AP and focused classes 1, 4 and 8 are in `comparison/per_class.csv`; "
                  "error residual correlations are in `comparison/comparison.json`.", "",
                  "E03's leaderboard display 76.45 is user-reported; the exact uploaded file remains "
                  "unverified locally. No hidden test inference or leaderboard upload was performed.", ""])
    (run / "n01_report.md").write_text("\n".join(lines), encoding="utf-8")
    required = {"config.yaml", "best_model.pt", "best_ema_model.pt", "best_metrics.json",
                "best_ema_metrics.json", "metrics.jsonl", "train_stdout.log", "train_stderr.log",
                "validation_analysis/study_predictions.csv", "ema_validation_analysis/study_predictions.csv",
                "comparison/comparison.json", "inference_benchmark.json", "n01_report.md"}
    files = {path.relative_to(run).as_posix(): {"size_bytes": path.stat().st_size, "sha256": digest(path)}
             for path in run.rglob("*") if path.is_file() and path.name != "frozen_manifest.json"}
    if not required.issubset(files):
        raise ValueError(f"Missing frozen N01 files: {sorted(required - set(files))}")
    (run / "frozen_manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in run.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    print(json.dumps({"best_epoch": best["epoch"], "raw_auc": raw["macro_auc"],
                      "ema_best_epoch": best_ema["epoch"], "ema_auc": ema["macro_auc"],
                      "files_frozen": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
