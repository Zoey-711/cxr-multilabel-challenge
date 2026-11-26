"""Seal T01 config, logs, best model, validation, benchmark and fixed E07 comparison."""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "frozen_manifest.json").exists():
        raise FileExistsError("T01 is already frozen")
    config = load_config(args.config)
    if config != load_config(run / "config.yaml") or config["data"]["input_size"] != 384 or \
            config["model"]["name"] != "swin_t" or config["training"]["batch_size"] != 8 or \
            config["training"]["gradient_accumulation_steps"] != 2:
        raise ValueError("T01 runtime config differs from approved Swin-Tiny 384/8x2 control")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    independent = json.loads((run / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    comparison = json.loads((run / "comparison/comparison.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run / "inference_benchmark.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "pretrained_source.json").read_text(encoding="utf-8"))
    epochs = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if status["state"] != "complete" or not epochs or len(epochs) > 30 or \
            [item["epoch"] for item in epochs] != list(range(1, len(epochs)+1)) or \
            status["stop_epoch"] != epochs[-1]["epoch"] or status["best_epoch"] != best["epoch"] or \
            status["early_stopping_patience"] != 6 or independent["study_count"] != 859 or \
            comparison["patient_count"] != 228 or benchmark["results"]["T01"]["input_size"] != 384 or \
            provenance["sha256"] != "704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b" or \
            comparison["fusion"].get("E07") != "0.5*E03+0.5*T01" or \
            abs(independent["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(comparison["scores"]["T01"]["macro_auc"] - independent["macro_auc"]) > 1e-8:
        raise ValueError("T01 training, independent validation, benchmark or comparison is inconsistent")
    if any(item["optimizer_steps"] + item["amp_skipped_steps"] != 1897 or item["study_sampling"] is not None
           for item in epochs):
        raise ValueError("T01 optimizer groups or sampling differ from M02")
    frozen_names = ("b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated",
                    "b03_densenet121_lr3e5_20261016", "m02_convnext_tiny_384_aug_20261016_elevated")
    total_model_bytes = (run / "best_model.pt").stat().st_size + sum(
        (Path("outputs/experiments") / name / "best_model.pt").stat().st_size for name in frozen_names)
    if total_model_bytes > 500_000_000:
        raise ValueError("E07 inference model files exceed 500 MB")
    manifest = {}
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.name == "frozen_manifest.json":
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest[path.relative_to(run).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": digest}
    required = {"config.yaml", "best_model.pt", "metrics.jsonl", "train_stdout.log", "train_stderr.log",
                "validation_analysis/study_predictions.csv", "validation_analysis/image_predictions.csv",
                "comparison/comparison.json", "inference_benchmark.json"}
    if not required.issubset(manifest):
        raise ValueError(f"Missing T01 artifacts: {sorted(required-set(manifest))}")
    (run / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in run.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    print(json.dumps({"run_dir": str(run), "best_epoch": best["epoch"],
                      "macro_auc": independent["macro_auc"], "files_frozen": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
