"""Seal the completed L01 training run and its independent validation results."""

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
        raise FileExistsError("L01 is already frozen")
    if load_config(args.config) != load_config(run / "config.yaml"):
        raise ValueError("L01 runtime config mismatch")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    independent = json.loads((run / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    comparison = json.loads((run / "comparison/comparison.json").read_text(encoding="utf-8"))
    epochs = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if status["state"] != "complete" or not epochs or len(epochs) > 30 or \
            [item["epoch"] for item in epochs] != list(range(1, len(epochs)+1)) or \
            status["stop_epoch"] != epochs[-1]["epoch"] or status["best_epoch"] != best["epoch"] or \
            status["early_stopping_patience"] != 6 or independent["study_count"] != 859 or \
            comparison["patient_count"] != 228 or \
            abs(independent["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(comparison["scores"]["L01"]["macro_auc"] - independent["macro_auc"]) > 1e-8:
        raise ValueError("L01 training, validation, or comparison is inconsistent")
    if any(item["optimizer_steps"] + item["amp_skipped_steps"] != 1897 or item["study_sampling"] is not None
           for item in epochs):
        raise ValueError("L01 optimizer accumulation groups or sampling differ from M02")
    manifest = {}
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.name == "frozen_manifest.json":
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest[path.relative_to(run).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": digest}
    if not {"config.yaml", "best_model.pt", "metrics.jsonl", "train_stdout.log", "train_stderr.log",
            "validation_analysis/study_predictions.csv", "validation_analysis/image_predictions.csv",
            "comparison/comparison.json"}.issubset(manifest):
        raise ValueError("L01 required artifacts are missing")
    (run / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in run.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    print(json.dumps({"run_dir": str(run), "best_epoch": best["epoch"],
                      "macro_auc": independent["macro_auc"], "files_frozen": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
