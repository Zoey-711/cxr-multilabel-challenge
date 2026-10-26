"""Validate a friend's direct Phase17 result folder before isolated local import.

No GPU, training, Test labels, or network access is used. The source remains read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROB_COLUMNS = ["Subject_id", "Study_id", *(f"Prob_{i}" for i in range(1, 11))]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes result root: {relative}")
    return path


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def label_keys(path: Path, fold: int | None = None) -> set[tuple[str, str]]:
    items = rows(path)
    keys = [(r["Subject_id"], r["Study_id"]) for r in items
            if fold is None or int(r["fold"]) == fold]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate fixed label keys: {path}")
    return set(keys)


def prediction_keys(path: Path, expected: set[tuple[str, str]]) -> set[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PROB_COLUMNS:
            raise ValueError(f"Wrong probability columns/order: {path}")
        seen = set()
        for row in reader:
            key = (row["Subject_id"], row["Study_id"])
            if not all(key) or key in seen:
                raise ValueError(f"Duplicate/blank Study key: {path}: {key}")
            seen.add(key)
            values = [float(row[f"Prob_{i}"]) for i in range(1, 11)]
            if any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
                raise ValueError(f"Invalid probability: {path}: {key}")
    if seen != expected:
        raise ValueError(f"Prediction Study coverage mismatch: {path}; got {len(seen)}, expected {len(expected)}")
    return seen


def verify(source: Path, plan: dict, expected_commit: str) -> dict:
    source = source.resolve()
    manifest_path = source / "manifests/delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "phase17_direct_folder_v1" or manifest.get("git_commit") != expected_commit:
        raise ValueError("Delivery format or source Git commit mismatch")
    if manifest.get("data_sha256") != plan["data_sha256"] or manifest.get("fold_hashes") != plan["fold_hashes"]:
        raise ValueError("Official archive or frozen patient-fold version mismatch")
    if manifest.get("code_sha256") != plan["artifacts"]:
        raise ValueError("Frozen code artifact hashes differ from the delivery plan")
    for name, sha in plan["fold_hashes"].items():
        if digest(ROOT / "outputs/analysis/phase17_patient_folds" / name) != sha:
            raise ValueError(f"Local frozen fold changed: {name}")
        if digest(source / "manifests" / name) != sha:
            raise ValueError(f"Delivered frozen fold changed: {name}")
    for name, info in manifest["important_files"].items():
        path = safe_path(source, name)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Delivery file hash mismatch: {name}")
    for name in ("train_labels.csv", "val_labels.csv"):
        if digest(source / "manifests" / name) != digest(ROOT / "batch_assets" / name):
            raise ValueError(f"Evaluation label version mismatch: {name}")
    train_keys = label_keys(source / "manifests/train_labels.csv")
    val_keys = label_keys(source / "manifests/val_labels.csv")
    image_rows = rows(source / "manifests/image_folds.csv")
    image_fields = set(image_rows[0]) if image_rows else set()
    if not {"Subject_id", "Study_id", "Dicom_id", "archive_member", "fold"} <= image_fields:
        raise ValueError("Frozen image index columns are incomplete")
    image_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"], r["archive_member"])
                  for r in image_rows}
    if len(image_rows) != 30344 or len(image_keys) != len(image_rows):
        raise ValueError("Frozen image index coverage/uniqueness differs")
    test_keys = None
    if len(train_keys) != 18513 or len(val_keys) != 859:
        raise ValueError("Official Train/Val Study count mismatch")
    task_map = {task["id"]: task for task in plan["tasks"]}
    extension = source / "manifests/extension_selection.json"
    if extension.is_file():
        extra = json.loads(extension.read_text(encoding="utf-8"))
        if extra.get("source_fold_hashes") != plan["fold_hashes"]:
            raise ValueError("G/H selection fold version mismatch")
        task_map.update({task["id"]: task for task in extra["tasks"]})
    completed = manifest["completed_tasks"]
    if len(completed) != len(set(completed)) or not set(completed) <= set(task_map):
        raise ValueError("Duplicate or unknown completed task ID")
    branch_oof: dict[str, set[tuple[str, str]]] = {}
    verified = []
    for task_id in completed:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("Unsafe task ID")
        task = task_map[task_id]
        run = source / "experiments" / task_id
        done = json.loads((run / "DONE.json").read_text(encoding="utf-8"))
        record = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        if done.get("state") != "complete" or done.get("model_id") != task_id:
            raise ValueError(f"Incomplete or mismatched DONE record: {task_id}")
        if record.get("model_id") != task_id or record.get("git_commit") != expected_commit:
            raise ValueError(f"Incorrect source model/commit: {task_id}")
        if record.get("fold_hashes") not in (plan["fold_hashes"], {}):
            raise ValueError(f"Incorrect fold version: {task_id}")
        if digest(run / "resolved_config.yaml") != task["config_sha256"]:
            raise ValueError(f"Training config changed: {task_id}")
        for name, sha in done["verified_sha256"].items():
            if digest(safe_path(run, name)) != sha:
                raise ValueError(f"Frozen task artifact changed: {task_id}/{name}")
        if digest(run / "training_image_index.csv") != record["training_image_index_sha256"]:
            raise ValueError(f"Training image index changed: {task_id}")
        run_image_rows = rows(run / "training_image_index.csv")
        actual_image_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"], r["archive_member"])
                             for r in run_image_rows}
        fold = task.get("fold")
        fold = int(fold) if fold is not None else None
        expected_image_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"], r["archive_member"])
                               for r in image_rows if fold is None or int(r["fold"]) != fold}
        if len(run_image_rows) != len(actual_image_keys) or actual_image_keys != expected_image_keys:
            raise ValueError(f"Training image index includes a held-out Fold or misses a Train image: {task_id}")
        if digest(run / "best_model.pth") != (run / "best_model.sha256").read_text(encoding="ascii").split()[0]:
            raise ValueError(f"Best model SHA file mismatch: {task_id}")
        prediction_keys(run / "val_predictions.csv", val_keys)
        if not (run / "test_predictions.csv").is_file():
            raise ValueError(f"Completed task lacks hidden Test probabilities: {task_id}")
        with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            test_rows = list(csv.DictReader(handle))
        current = {(r["Subject_id"], r["Study_id"]) for r in test_rows}
        if len(test_rows) != 1414 or len(current) != 1414:
            raise ValueError(f"Hidden Test Study coverage mismatch: {task_id}")
        if test_keys is None:
            test_keys = current
        prediction_keys(run / "test_predictions.csv", test_keys)
        if fold is not None:
            expected_oof = label_keys(source / "manifests/train_labels.csv", fold)
            actual_oof = prediction_keys(run / "oof_predictions.csv", expected_oof)
            trained_patients = {r["Subject_id"] for r in rows(run / "training_image_index.csv")}
            if trained_patients & {subject for subject, _ in actual_oof}:
                raise ValueError(f"OOF patient leakage: {task_id}")
            branch = task_id.split("_F")[0]
            if branch_oof.setdefault(branch, set()) & actual_oof:
                raise ValueError(f"Duplicate OOF Study across folds: {branch}")
            branch_oof[branch].update(actual_oof)
        elif (run / "oof_predictions.csv").exists():
            raise ValueError(f"Full-Train model has a false OOF file: {task_id}")
        if task.get("ema"):
            if digest(run / "ema_best_model.pth") != (run / "ema_best_model.sha256").read_text(encoding="ascii").split()[0]:
                raise ValueError(f"EMA model SHA file mismatch: {task_id}")
            prediction_keys(run / "ema_val_predictions.csv", val_keys)
            if fold is not None:
                prediction_keys(run / "ema_oof_predictions.csv", expected_oof)
            if test_keys is not None:
                prediction_keys(run / "ema_test_predictions.csv", test_keys)
        verified.append(task_id)
    full_oof = {}
    for branch in ("D17", "C17", "S17", "N17"):
        all_folds = {f"{branch}_F{i:02d}" for i in range(1, 6)}
        if all_folds <= set(completed):
            if branch_oof.get(branch) != train_keys:
                raise ValueError(f"Complete five-fold OOF missing Study: {branch}")
            if len({subject for subject, _ in branch_oof[branch]}) != 11055:
                raise ValueError(f"Complete five-fold OOF missing patient: {branch}")
            prediction_keys(source / "oof" / f"{branch}_oof_all.csv", train_keys)
            full_oof[branch] = "PASS"
        else:
            full_oof[branch] = "INCOMPLETE"
    return {"state": "VERIFIED", "source_commit": expected_commit,
            "completed_tasks": verified, "full_oof": full_oof,
            "val_studies": len(val_keys), "test_studies": len(test_keys) if test_keys is not None else None,
            "hidden_test_auc": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Friend's frozen 26-task batch_plan.yaml")
    parser.add_argument("--source", required=True, type=Path, help="Received phase17_batch directory")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--version", required=True, help="Unique import version, e.g. friend_v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.version):
        parser.error("Unsafe import version")
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = verify(args.source, plan, args.expected_commit)
    if args.verify_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    destination = ROOT / "outputs/phase17_friend_import" / args.version
    if destination.exists():
        raise FileExistsError(f"Import version already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source, destination)
    if verify(destination, plan, args.expected_commit) != report:
        raise ValueError("Copied result verification differs from source")
    (destination / "IMPORT_VERIFIED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
