"""Verify frozen local Phase17 files and export a portable, CPU-only OOF handoff."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_phase17_five_fold import ROOT, RUN_DATES, WEIGHTS, frozen_index, verified_prediction, verified_run
from cxr.data.patient_folds import sha256
from cxr.utils.config import load_config


BRANCH_PREFIX = {"D17": "f17_densenet121", "C17": "c17", "S17": "s17"}
PROBS = [f"Prob_{i}" for i in range(1, 11)]


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction_table(path: Path, *, has_fold: bool):
    fields = ["Subject_id", "Study_id"] + (["Fold"] if has_fold else []) + \
        [f"probability_{i}" for i in range(1, 11)] + [f"target_{i}" for i in range(1, 11)]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Study/class columns differ: {path}")
        rows = list(reader)
    keys = [(row["Subject_id"], row["Study_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Repeated Study: {path}")
    p = np.asarray([[float(row[f"probability_{i}"]) for i in range(1, 11)] for row in rows])
    y = np.asarray([[int(row[f"target_{i}"]) for i in range(1, 11)] for row in rows])
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"Invalid prediction or label: {path}")
    folds = [int(row["Fold"]) for row in rows] if has_fold else None
    return keys, p, y, folds


def verify_friend_plan(path: Path, fold_hashes: dict):
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if plan["fold_hashes"] != fold_hashes:
        raise ValueError("Friend plan uses a different patient-fold split")
    tasks = {task["id"]: task for task in plan["tasks"]}
    records = []
    for fold in range(1, 6):
        for branch in WEIGHTS:
            name = f"{branch}_F{fold:02d}"
            task = tasks[name]
            local = ROOT / "configs" / f"{BRANCH_PREFIX[branch]}_fold{fold}_384.yaml"
            remote = path.parents[1] / task["config"]
            if sha256(local) != task["source_config_sha256"] or \
                    sha256(remote) != task["config_sha256"] or task["fold_hashes"] != fold_hashes:
                raise ValueError(f"Friend task config/hash differs: {name}")
            local_config = load_config(local)
            friend_config = load_config(remote)
            local_config["data"]["archive_path"] = friend_config["data"]["archive_path"]
            ema = friend_config["training"].pop("ema", None)
            if local_config != friend_config:
                raise ValueError(f"Friend training protocol differs beyond archive path/EMA: {name}")
            if ema and (branch != "C17" or ema != {"enabled": True, "decay": 0.999,
                    "update_after_step": 1, "use_buffers": True}):
                raise ValueError(f"Unexpected EMA addition: {name}")
            records.append({"task_id": name, "source_config_sha256": task["source_config_sha256"],
                            "friend_config_sha256": task["config_sha256"],
                            "difference": "data path + C17 EMA" if ema else "data path only"})
    return {"state": "FOLD_AND_BASE_CONFIG_ALIGNED", "plan_path": str(path),
            "plan_sha256": sha256(path), "tasks": records,
            "note": "Friend models are independently trained versions; matching Fold/config does not equate their predictions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--friend-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "configs/f17_densenet121_fold5_384.yaml").resolve():
        parser.error("Use the frozen Fold5 D17 configuration")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Handoff directory already exists: {out}")
    patients, studies, fold_hashes = frozen_index()
    friend = verify_friend_plan(args.friend_plan.resolve(), fold_hashes)
    frozen = []
    for fold in range(1, 6):
        for branch in WEIGHTS:
            run, manifest, record = verified_run(branch, fold, fold_hashes)
            config = run / "config.yaml"
            entry = manifest["files"]["config.yaml"]
            if config.stat().st_size != entry["size_bytes"] or sha256(config) != entry["sha256"]:
                raise ValueError(f"Runtime config SHA differs: {run}")
            for split, expected in (("validation_analysis", 859),
                                    ("oof_analysis", sum(value == fold for value in studies.values()))):
                keys, labels, probability = verified_prediction(run, manifest, split)
                if len(keys) != expected or probability.shape != (expected, 10):
                    raise ValueError(f"Study coverage differs: {run}/{split}")
                if split == "oof_analysis" and any(patients[subject] != fold or
                        studies[(subject, study)] != fold for subject, study in keys):
                    raise ValueError(f"Patient/Study leakage: {run}")
            record.update({"model": manifest["model"], "runtime_config_sha256": sha256(config),
                           "manifest_sha256": sha256(run / "frozen_manifest.json"),
                           "best_epoch": manifest["best_epoch"],
                           "val_macro_auc": manifest["official_val_macro_auc"],
                           "oof_macro_auc": manifest["oof_macro_auc"],
                           "relative_run_path": run.relative_to(ROOT).as_posix(),
                           "artifact_paths": {
                               name: (run / name).relative_to(ROOT).as_posix()
                               for name in ("best_model.pt", "config.yaml", "frozen_manifest.json",
                                            "validation_analysis/study_predictions.csv",
                                            "oof_analysis/study_predictions.csv")}})
            frozen.append(record)
    analysis_dir = ROOT / "outputs/analysis/phase17_five_fold_final_20261023"
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    for name, digest in analysis["output_sha256"].items():
        if sha256(analysis_dir / name) != digest:
            raise ValueError(f"Complete five-fold output SHA differs: {name}")
    if len(frozen) != 15 or analysis["complete_oof"]["coverage_studies"] != 18513 or \
            analysis["complete_oof"]["coverage_patients"] != 11055:
        raise ValueError("The 15-model/full OOF analysis is incomplete")
    oof_keys, oof_p, oof_y, oof_folds = prediction_table(analysis_dir / "e17_oof_complete.csv", has_fold=True)
    if len(oof_keys) != 18513 or set(oof_keys) != set(studies) or len({x[0] for x in oof_keys}) != 11055 or \
            any(studies[key] != fold or patients[key[0]] != fold for key, fold in zip(oof_keys, oof_folds)):
        raise ValueError("Complete OOF is not exactly one withheld Fold per Study/patient")
    label_rows = read_rows(ROOT / "batch_assets/train_labels.csv")
    labels = {(row["Subject_id"], row["Study_id"]): row for row in label_rows}
    if len(labels) != 18513:
        raise ValueError("Official Train label coverage differs")
    if any(int(labels[key]["fold"]) != fold or
           any(int(labels[key][f"Target_{i}"]) != int(y[i - 1]) for i in range(1, 11))
           for key, fold, y in zip(oof_keys, oof_folds, oof_y)):
        raise ValueError("Complete OOF target/class order differs from frozen labels")
    branch_values = {}
    for branch in WEIGHTS:
        keys, probabilities, targets, folds = prediction_table(
            analysis_dir / f"{branch.lower()}_oof_complete.csv", has_fold=True)
        if set(keys) != set(oof_keys):
            raise ValueError(f"{branch} complete OOF differs in Study coverage")
        positions = {key: index for index, key in enumerate(keys)}
        order = [positions[key] for key in oof_keys]
        if not np.array_equal(targets[list(order)], oof_y) or \
                [folds[i] for i in order] != oof_folds:
            raise ValueError(f"{branch} OOF target/Fold differs")
        branch_values[branch] = probabilities[list(order)]
    recomputed = sum(WEIGHTS[name] * branch_values[name] for name in WEIGHTS)
    max_diff = float(np.max(np.abs(recomputed - oof_p)))
    if max_diff > 1e-12:
        raise ValueError(f"Complete E17 OOF fusion differs: {max_diff}")
    val_keys, val_p, val_y, _ = prediction_table(analysis_dir / "e17_val_fivefold.csv", has_fold=False)
    if len(val_keys) != 859 or len({key[0] for key in val_keys}) != 228:
        raise ValueError("Five-fold validation coverage differs")
    val_branches = {}
    for branch in WEIGHTS:
        keys, p, y, _ = prediction_table(analysis_dir / f"{branch.lower()}_val_fivefold.csv", has_fold=False)
        if keys != val_keys or not np.array_equal(y, val_y):
            raise ValueError(f"{branch} validation Study/label alignment differs")
        val_branches[branch] = p
    if np.max(np.abs(sum(WEIGHTS[name] * val_branches[name] for name in WEIGHTS) - val_p)) > 1e-12:
        raise ValueError("Five-fold Val fusion differs")
    out.mkdir(parents=True)
    with (out / "e17_oof_study_probabilities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id", *PROBS])
        writer.writerows([subject, study, *(repr(float(x)) for x in probability)]
                         for (subject, study), probability in zip(oof_keys, oof_p))
    with (out / "e17_oof_source_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id", "Fold", "D17_Model_ID", "C17_Model_ID", "S17_Model_ID"])
        writer.writerows([subject, study, fold, *(f"{branch}_F{fold:02d}" for branch in WEIGHTS)]
                         for (subject, study), fold in zip(oof_keys, oof_folds))
    audit = {"status": "VERIFIED", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "source_root": str(ROOT), "fold_hashes": fold_hashes,
             "counts": {"models": 15, "patients": 11055, "studies": 18513, "val_studies": 859},
             "oof_patient_leakage": False, "oof_duplicate_studies": 0,
             "oof_fixed_fusion_max_abs_diff": max_diff,
             "complete_oof_auc": analysis["complete_oof"]["E17"]["macro_auc"],
             "fivefold_val_auc": analysis["validation"]["E17"]["macro_auc"],
             "friend_plan_alignment": friend,
             "friend_n17_ema_comparison": "PENDING: no verified friend result import",
             "models": frozen,
             "source_outputs": {name: {"path": (analysis_dir / name).relative_to(ROOT).as_posix(),
                                       "sha256": digest}
                                for name, digest in analysis["output_sha256"].items()},
             "handoff_sha256": {name: sha256(out / name) for name in
                                ("e17_oof_study_probabilities.csv", "e17_oof_source_index.csv")}}
    (out / "freeze_manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Phase17 本地五折结果交接", "",
              "15/15 个最佳 checkpoint、运行配置、Val/OOF 概率和冻结 SHA-256 均已核验。",
              "18,513 个 Train Study / 11,055 名患者各有且仅有一个所属 Fold 的 OOF 来源，标签与10类顺序一致。",
              f"完整 E17 OOF AUC {audit['complete_oof_auc']:.6f}；五折平均 Val AUC {audit['fivefold_val_auc']:.6f}。",
              "统一概率文件为 e17_oof_study_probabilities.csv；对应 Fold 和三模型版本见 e17_oof_source_index.csv。",
              "朋友计划的折划分、D17/S17 原训练配置匹配；C17 额外维护 EMA，普通权重需与 EMA 明确区分。",
              "朋友模型独立训练，不能仅凭配置相同就混用其 OOF；必须先通过 import_phase17_friend_results.py 的 manifest/标签/索引检查。",
              "朋友 N17 EMA 正式结果未同步：PENDING；不得填写对比数值。",
              "官方 Val 曾重复用于早停和模型选择，Val 差异不能当作独立泛化证据。",
              "完整15模型超过500MB，且目标硬件延迟未验证；仅用于研究，不生成 Submission。"]
    (out / "handoff_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VERIFIED", "models": 15, "studies": 18513,
                      "patients": 11055, "friend_n17_ema": "PENDING", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
