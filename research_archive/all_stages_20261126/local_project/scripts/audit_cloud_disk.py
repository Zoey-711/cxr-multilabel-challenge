"""Read-only archive and local-artifact capacity audit for Phase17 Cloud."""

import argparse
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from cxr.data.archive import archive_path
from cxr.utils.config import load_config


def directory_usage(path: Path) -> dict:
    count = size = 0
    if not path.exists():
        return {"files": 0, "bytes": 0}
    for base, _, files in os.walk(path):
        for name in files:
            try:
                size += (Path(base) / name).stat().st_size
                count += 1
            except (OSError, PermissionError):
                continue
    return {"files": count, "bytes": size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    archive = archive_path(config)
    if not archive.is_file():
        parser.error("Local audit expects the original combined ZIP")
    groups = defaultdict(lambda: {"files": 0, "uncompressed_bytes": 0, "compressed_bytes": 0})
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if item.is_dir():
                continue
            parts = item.filename.split("/")
            group = parts[1] if len(parts) > 1 else "other"
            entry = groups[group]
            entry["files"] += 1
            entry["uncompressed_bytes"] += item.file_size
            entry["compressed_bytes"] += item.compress_size
    project = Path(__file__).resolve().parents[1]
    runs = project / "outputs" / "experiments"
    controls = {
        "D": "b01_densenet121_384_20261016",
        "C": "m02_convnext_tiny_384_aug_20261016_elevated",
        "S": "t01_swin_tiny_384_20261018",
    }
    models = {}
    for name, run_id in controls.items():
        run = runs / run_id
        models[name] = {"best_checkpoint_bytes": (run / "best_model.pt").stat().st_size,
                        "last_training_bytes": (run / "last_training.pt").stat().st_size}
    checkpoint_15 = 5 * sum(item["best_checkpoint_bytes"] for item in models.values())
    resume_15 = 5 * sum(item["last_training_bytes"] for item in models.values())
    estimate = {
        "one_fold_three_models_best_plus_resume_bytes": sum(item["best_checkpoint_bytes"] +
                                                               item["last_training_bytes"] for item in models.values()),
        "fifteen_best_checkpoints_bytes": checkpoint_15,
        "fifteen_last_training_snapshots_bytes": resume_15,
        "fifteen_best_plus_resume_bytes": checkpoint_15 + resume_15,
    }
    train = groups["train"]["compressed_bytes"] + groups["train.csv"]["compressed_bytes"]
    valid = groups["valid"]["compressed_bytes"] + groups["valid.csv"]["compressed_bytes"]
    report = {
        "archive_path": str(archive), "archive_bytes": archive.stat().st_size,
        "uncompressed_file_bytes": sum(group["uncompressed_bytes"] for group in groups.values()),
        "file_count": sum(group["files"] for group in groups.values()),
        "archive_groups": dict(groups),
        "train_only_zip_estimated_compressed_member_bytes": train,
        "train_plus_valid_zip_estimated_compressed_member_bytes": train + valid,
        "pretrained_cache": directory_usage(project / ".wheels" / "torch_cache" / "hub" / "checkpoints"),
        "all_experiment_outputs": directory_usage(runs),
        "analysis_outputs": directory_usage(project / "outputs" / "analysis"),
        "python_environment": directory_usage(project / ".venv"),
        "control_artifact_bytes": models,
        "training_output_estimate": estimate,
        "disk_50gb_decimal_bytes": 50_000_000_000,
        "disk_50gib_bytes": 50 * 1024**3,
        "capacity_conclusion": "Original combined ZIP cannot fit 50 GB or 50 GiB. Train-only ZIP may fit, "
                               "but exact free bytes and safety margin must be checked before upload; "
                               "Val and outputs should be on separate storage."}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Audit output already exists: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("archive_bytes", "uncompressed_file_bytes", "file_count",
                                                   "train_only_zip_estimated_compressed_member_bytes",
                                                   "train_plus_valid_zip_estimated_compressed_member_bytes",
                                                   "training_output_estimate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
