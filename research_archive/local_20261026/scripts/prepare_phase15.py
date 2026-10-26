"""Audit the frozen patient split and two no-training Phase 15 control configs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from cxr.data.archive import archive_path, parse_positive_classes, read_split_rows
from cxr.data.internal_split import select_internal_train_rows
from cxr.models.factory import imagenet_weight_info
from cxr.utils.config import load_config


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def changes(before: dict, after: dict) -> list[str]:
    return [f"{section}.{key}" for section in before for key in set(before[section]) | set(after[section])
            if before[section].get(key) != after[section].get(key)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or root / protocol["audit_directory"]
    if output.exists():
        raise FileExistsError(output)
    if protocol["phase"] != 15 or set(protocol["runs"]) != {"A", "B"} or \
            protocol["fixed_fusion"].get("p15a_densenet121") != 0.5 or \
            protocol["fixed_fusion"].get("p15b_convnext_tiny") != 0.5:
        raise ValueError("Phase 15 fixed two-model protocol differs")
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 4060 GPU is required for the planned retrains")

    runs = {}
    common_archive = None
    split_root = root / protocol["split_directory"]
    split_report = json.loads((split_root / "split_report.json").read_text(encoding="utf-8"))
    if split_report["patient_overlap"] != 0 or split_report["official_validation_patient_overlap"] != 0:
        raise ValueError("Frozen split reports patient leakage")
    for label, plan in protocol["runs"].items():
        control_file, config_file = root / plan["control_config"], root / plan["config"]
        if sha256(control_file) != plan["control_config_sha256"]:
            raise ValueError(f"Frozen {label} control config SHA-256 changed")
        control, config = load_config(control_file), load_config(config_file)
        differences = sorted(changes(control, config))
        allowed = ["data.train_subset", "experiment.name", "model.pretrained_sha256"]
        if differences != allowed:
            raise ValueError(f"{label} has unapproved changes from its runtime control: {differences}")
        if config["model"]["pretrained"] != "imagenet" or \
                config["model"]["pretrained_sha256"] != plan["pretrained_sha256"] or \
                config["validation"]["study_fusion"] != "mean_probability" or \
                config["validation"]["primary_metric"] != "macro_auc":
            raise ValueError(f"{label} weight or validation protocol differs")
        source, url, weight_name = imagenet_weight_info(config["model"]["name"])
        cache = root / ".wheels/torch_cache/hub/checkpoints" / weight_name
        original = json.loads((control_file.parent / "pretrained_source.json").read_text(encoding="utf-8"))
        if sha256(cache) != plan["pretrained_sha256"] or original["sha256"] != plan["pretrained_sha256"]:
            raise ValueError(f"{label} ImageNet cache differs from the original run")
        archive = archive_path(config)
        if common_archive is None:
            common_archive = archive
        if archive != common_archive or archive != archive_path(control):
            raise ValueError("Control and retrain archives differ")
        original_rows = read_split_rows(archive, "train")
        val_rows = read_split_rows(archive, "val")
        train_rows = select_internal_train_rows(original_rows, config["data"]["train_subset"])
        train_patients = {row["Subject_id"] for row in train_rows}
        all_patients = {row["Subject_id"] for row in original_rows}
        val_patients = {row["Subject_id"] for row in val_rows}
        if (len(val_rows), len({(row["Subject_id"], row["Study_id"]) for row in val_rows}),
                len(val_patients)) != (1422, 859, 228):
            raise ValueError("Official validation coverage differs")
        if train_patients & val_patients or all_patients & val_patients:
            raise ValueError("Official validation patient leakage")
        study_rows = {}
        for row in train_rows:
            key = (row["Subject_id"], row["Study_id"])
            labels = parse_positive_classes(row["Predict_class"])
            if key in study_rows and study_rows[key] != labels:
                raise ValueError("Inconsistent Study label")
            study_rows[key] = labels
        run_dir = root / "outputs/experiments" / plan["run_id"]
        if run_dir.exists():
            raise FileExistsError(f"Planned formal run directory already exists: {run_dir}")
        batch = int(config["training"]["batch_size"])
        accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
        updates_per_epoch = math.ceil(math.ceil(len(train_rows) / batch) / accumulation)
        records = [json.loads(line) for line in (control_file.parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        historical_mean_epoch_seconds = sum(row["elapsed_seconds"] for row in records) / len(records)
        lower_seconds = historical_mean_epoch_seconds * len(train_rows) / len(original_rows)
        positive_counts = {str(class_id): sum(class_id in labels for labels in study_rows.values())
                           for class_id in range(1, 11)}
        if any(count <= 0 or count >= len(study_rows) for count in positive_counts.values()):
            raise ValueError("Internal Train lacks positive/negative Study labels")
        runs[label] = {"control_config_sha256": sha256(control_file), "retrain_config_sha256": sha256(config_file),
                       "runtime_config_changes": differences, "run_id": plan["run_id"],
                       "planned_output": str(run_dir), "model": config["model"]["name"],
                       "pretrained_source": source, "pretrained_url": url,
                       "pretrained_sha256": sha256(cache), "weight_cache_bytes": cache.stat().st_size,
                       "patients": len(train_patients), "studies": len(study_rows), "images": len(train_rows),
                       "positive_studies_by_class_id": positive_counts,
                       "micro_batch": batch, "accumulation": accumulation,
                       "effective_batch": batch * accumulation, "optimizer_groups_per_epoch": updates_per_epoch,
                       "historical_epochs": len(records), "historical_mean_epoch_seconds": historical_mean_epoch_seconds,
                       "projected_epoch_seconds_lower": lower_seconds,
                       "projected_epoch_seconds_upper": historical_mean_epoch_seconds,
                       "original_gpu_peak_allocated_mib": max((row.get("gpu_peak_allocated_mib", 0) for row in records)) or None}
    holdout = split_report["counts"]["internal_val"]
    if split_report["counts"]["internal_train"] != {"patients": 9949, "studies": 16667, "images": 27301} or \
            holdout != {"patients": 1106, "studies": 1846, "images": 3043}:
        raise ValueError("Frozen patient split size differs")
    with (split_root / "study_split.csv").open(encoding="utf-8-sig", newline="") as handle:
        frozen_studies = list(csv.DictReader(handle))
    holdout_studies = [row for row in frozen_studies if row["split"] == "internal_val"]
    if len(holdout_studies) != holdout["studies"]:
        raise ValueError("Frozen Holdout Study count differs")
    holdout_support = {str(class_id): {
        "positive_studies": sum(class_id in parse_positive_classes(row["Predict_class"]) for row in holdout_studies),
        "negative_studies": sum(class_id not in parse_positive_classes(row["Predict_class"]) for row in holdout_studies)}
        for class_id in range(1, 11)}
    if any(not item["positive_studies"] or not item["negative_studies"] for item in holdout_support.values()):
        raise ValueError("Internal Holdout lacks positive/negative Study labels")
    total_lower = sum(row["projected_epoch_seconds_lower"] * row["historical_epochs"] for row in runs.values())
    total_upper = sum(row["projected_epoch_seconds_upper"] * row["historical_epochs"] for row in runs.values())
    max_lower = sum(row["projected_epoch_seconds_lower"] * 30 for row in runs.values())
    max_upper = sum(row["projected_epoch_seconds_upper"] * 30 for row in runs.values())
    result = {"phase": 15, "status": "prepared_not_trained", "source_archive": str(common_archive),
              "split_directory": str(split_root), "split_sha256": {
                  name: sha256(split_root / name) for name in ("patient_split.csv", "study_split.csv", "image_split.csv")},
              "train": split_report["counts"]["internal_train"], "holdout_class_support": holdout_support,
              "patient_overlap": 0, "official_validation_patients": split_report["official_validation_patients"],
              "runs": runs, "gpu": {"name": torch.cuda.get_device_name(0),
                                     "total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20)},
              "estimated_two_model_hours_at_historical_stop_lower_upper": [total_lower / 3600, total_upper / 3600],
              "estimated_two_model_hours_at_30_epochs_lower_upper": [max_lower / 3600, max_upper / 3600],
              "selection": "official validation Study Macro-AUC only, per model",
              "holdout": {**holdout, "role": "single final evaluation after both checkpoints and equal fusion are frozen"},
              "fusion": "0.5*P15A_StudyMean+0.5*P15B_StudyMean",
              "bootstrap": protocol["bootstrap"]}
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "gpu": result["gpu"],
                      "estimated_hours": result["estimated_two_model_hours_at_historical_stop_lower_upper"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
