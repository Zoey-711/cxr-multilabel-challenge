"""Frozen paired protocol and explicit server authorization for eight new folds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "configs/phase41_plan.yaml"
SOURCE_COUNTS = {"patients": 11055, "studies": 18513, "images": 30344}
REQUIRED = ("experiment_id", "architecture", "input_size", "seed", "fold", "micro_batch",
            "accumulation", "epochs", "patience", "ema_decay", "ema_use_buffers", "backbone_lr",
            "head_lr", "weight_decay", "warmup_epochs", "augmentation", "preprocessing",
            "selection_metric", "pretrained_sha256", "fold_sha256")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_plan() -> dict:
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    expected = {f"P41_{variant}_F{fold:02d}" for variant in ("BASE", "FUSION") for fold in range(2, 6)}
    tasks = plan["tasks"]
    if plan.get("schema") != "phase41_v1" or len(tasks) != 8 or {t["id"] for t in tasks} != expected:
        raise ValueError("Phase41 requires exactly eight new Fold2-5 tasks")
    if sum(plan["fold_counts"][fold][key] for fold in range(1, 6) for key in ("patients",)) != SOURCE_COUNTS["patients"]:
        raise ValueError("Frozen patient counts changed")
    for key in ("studies", "images"):
        if sum(plan["fold_counts"][fold][key] for fold in range(1, 6)) != SOURCE_COUNTS[key]:
            raise ValueError(f"Frozen {key} counts changed")
    for variant in ("BASE", "FUSION"):
        source = plan["fold1"][variant]
        if digest(ROOT / source["config"]) != source["config_sha256"]:
            raise ValueError("Frozen Fold1 configuration changed")
        base = yaml.safe_load((ROOT / source["config"]).read_text(encoding="utf-8"))
        if set(base) != set(REQUIRED) or base["fold"] != 1 or base["micro_batch"] != 4 or base["accumulation"] != 4:
            raise ValueError("Fold1 protocol or BatchNorm microbatch differs")
        if (base["pretrained_sha256"] != plan["pretrained_sha256"] or
                base["fold_sha256"] != plan["fold_sha256"] or base["ema_use_buffers"] is not True):
            raise ValueError("Fold1 initialization, split or EMA differs")
    for task in tasks:
        if task["id"] != f"P41_{task['variant']}_F{task['fold']:02d}" or task["seed"] != 42:
            raise ValueError("Task ID, fold or seed changed")
        if not 2 <= task["fold"] <= 5 or not 0 < task["timeout_hours"] <= 48 or task["max_retries"] != 1:
            raise ValueError("Task timeout/retry changed")
        config = resolved_config(plan, task)
        reference = yaml.safe_load((ROOT / plan["fold1"][task["variant"]]["config"]).read_text(encoding="utf-8"))
        for key in reference:
            if key not in ("experiment_id", "fold", "selection_metric") and config[key] != reference[key]:
                raise ValueError(f"Frozen Phase39 setting drifted: {key}")
    return plan


def resolved_config(plan: dict, task: dict) -> dict:
    source = plan["fold1"][task["variant"]]
    config = yaml.safe_load((ROOT / source["config"]).read_text(encoding="utf-8"))
    config.update(experiment_id=task["id"], fold=task["fold"], seed=task["seed"],
                  selection_metric="fold_study_macro_auc")
    config["fold_expected_counts"] = {
        "oof": plan["fold_counts"][task["fold"]],
        "train": {k: SOURCE_COUNTS[k] - plan["fold_counts"][task["fold"]][k] for k in SOURCE_COUNTS},
    }
    return config


def server_config(path: Path) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = ("data_root", "pretrained_root", "protocol_root", "fold1_results_root", "output_root",
                "gpu_ids", "max_concurrent", "gpu_hours", "minimum_free_disk_gib",
                "minimum_free_memory_gib_per_worker", "upload")
    if set(settings) not in (set(required), set(required) | {"recovery_source_root"}) or any(not Path(settings[k]).is_absolute() for k in
                                            ("data_root", "pretrained_root", "protocol_root", "fold1_results_root", "output_root")):
        raise ValueError("Complete absolute server paths and resource limits required")
    if "recovery_source_root" in settings:
        source = Path(settings["recovery_source_root"])
        target = Path(settings["output_root"])
        if not source.is_absolute() or source.resolve() == target.resolve():
            raise ValueError("Recovery requires a distinct absolute frozen source directory")
    ids = settings["gpu_ids"]
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or len(ids) > 8 or
            any(type(i) is not int or i < 0 for i in ids) or
            type(settings["max_concurrent"]) is not int or not 1 <= settings["max_concurrent"] <= len(ids)):
        raise ValueError("Authorize one to eight distinct physical GPU IDs")
    if type(settings["gpu_hours"]) not in (int, float) or not 0 < settings["gpu_hours"] <= 10000:
        raise ValueError("Friend must explicitly authorize a finite GPU-hour budget")
    for name in ("minimum_free_disk_gib", "minimum_free_memory_gib_per_worker"):
        if type(settings[name]) not in (int, float) or settings[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if settings["upload"].get("enabled"):
        upload = settings["upload"]
        expected = ("results/phase41r-recovery", "phase41r-recovery-v1") if "recovery_source_root" in settings else ("results/phase41-fivefold", "phase41-fivefold-v1")
        if (upload.get("repository") != "Zoey-711/cxr-multilabel-challenge" or
                (upload.get("branch"), upload.get("release_tag")) != expected or
                not Path(upload.get("checkout", "")).is_absolute()):
            raise ValueError("Phase41 upload requires the dedicated private result branch and Release")
    return settings
