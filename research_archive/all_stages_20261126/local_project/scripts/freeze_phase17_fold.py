"""Verify and hash-lock one completed Phase17 Fold 1 experiment."""

import argparse
import json
import os
import stat
from pathlib import Path

from cxr.data.patient_folds import sha256
from cxr.utils.config import load_config


REQUIRED = {"config.yaml", "best_model.pt", "best_metrics.json", "last_training.pt",
            "metrics.jsonl", "train_stdout.log", "train_stderr.log",
            "validation_analysis/analysis.json", "validation_analysis/study_predictions.csv",
            "oof_analysis/analysis.json", "oof_analysis/study_predictions.csv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    run = args.run_dir.resolve()
    if load_config(run / "config.yaml") != config:
        raise ValueError("Training config copy differs")
    if (run / "frozen_manifest.json").exists():
        raise FileExistsError("Fold already frozen")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete":
        raise ValueError("Cannot freeze an incomplete Fold")
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    val = json.loads((run / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    oof = json.loads((run / "oof_analysis/analysis.json").read_text(encoding="utf-8"))
    if abs(val["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            best["epoch"] != status["best_epoch"] or \
            oof["fold"] != config["data"]["train_subset"]["fold"] or \
            oof["checkpoint_sha256"] != sha256(run / "best_model.pt"):
        raise ValueError("Best checkpoint, validation or OOF mismatch")
    files = {path.relative_to(run).as_posix(): {"size_bytes": path.stat().st_size,
                                                "sha256": sha256(path)}
             for path in run.rglob("*") if path.is_file() and path.name != "frozen_manifest.json"}
    if not REQUIRED.issubset(files):
        raise ValueError(f"Missing required frozen files: {sorted(REQUIRED - set(files))}")
    manifest = {"model": config["model"]["name"], "fold": oof["fold"],
                "best_epoch": best["epoch"], "official_val_macro_auc": val["macro_auc"],
                "oof_macro_auc": oof["macro_auc"],
                "best_checkpoint_sha256": files["best_model.pt"]["sha256"],
                "fold_index_sha256": config["data"]["train_subset"]["sha256"],
                "files": files}
    (run / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in run.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    print(json.dumps({key: manifest[key] for key in ("model", "fold", "best_epoch",
                                                      "official_val_macro_auc", "oof_macro_auc",
                                                      "best_checkpoint_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
