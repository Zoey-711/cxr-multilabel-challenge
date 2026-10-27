"""Patient-isolated full OOF and fivefold Val summaries; no Test labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cxr.data.archive import read_split_rows
from cxr.data.dataset import study_targets_from_rows
from cxr.data.patient_folds import read_index
from cxr.engine.metrics import binary_auc
from cxr.phase39_support import write_csv, write_json
from cxr.phase41.protocol import digest


def matrix(path: Path, expected: set[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("Subject_id", "Study_id", *(f"Prob_{i}" for i in range(1, 11))):
            raise ValueError(f"Ten probability columns or ID order changed: {path}")
        rows = list(reader)
    table = {(r["Subject_id"], r["Study_id"]): np.asarray([float(r[f"Prob_{i}"]) for i in range(1, 11)]) for r in rows}
    if len(table) != len(rows) or set(table) != expected or any(not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() for p in table.values()):
        raise ValueError(f"Prediction coverage/value failure: {path}")
    return table


def auc_summary(truth, probabilities, keys):
    labels = np.asarray([truth[k] for k in keys], dtype=np.int8)
    values = np.asarray([probabilities[k] for k in keys], dtype=np.float64)
    per_class = [float(binary_auc(labels[:, j], values[:, j])) for j in range(10)]
    return dict(macro_auc=float(np.mean(per_class)), per_class_auc=per_class,
                positive_studies=[int(labels[:, j].sum()) for j in range(10)], studies=len(keys),
                patients=len({k[0] for k in keys}))


def bootstrap(truth, base, fusion, keys, *, seed=3901, draws=2000):
    patients = sorted({k[0] for k in keys})
    groups = {p: [k for k in keys if k[0] == p] for p in patients}
    rng = np.random.default_rng(seed)
    deltas = []
    invalid = 0
    for _ in range(draws):
        sample = [k for p in rng.choice(patients, size=len(patients), replace=True) for k in groups[p]]
        try:
            a = auc_summary(truth, base, sample)["macro_auc"]
            b = auc_summary(truth, fusion, sample)["macro_auc"]
            deltas.append(b - a)
        except ValueError:
            invalid += 1
    if len(deltas) < 0.9 * draws:
        raise ValueError("Too many invalid patient bootstrap draws")
    return dict(seed=seed, requested=draws, valid=len(deltas), invalid=invalid,
                delta=float(auc_summary(truth, fusion, keys)["macro_auc"] - auc_summary(truth, base, keys)["macro_auc"]),
                ci95=[float(x) for x in np.percentile(deltas, [2.5, 97.5])])


def analyze(root: Path, plan: dict, server: dict) -> dict:
    """Emit INCOMPLETE if any fold is missing; never fabricate full OOF."""
    root = Path(root)
    report = root / "reports"; report.mkdir(parents=True, exist_ok=True)
    archive = Path(server["data_root"])
    train_rows, val_rows = read_split_rows(archive, "train"), read_split_rows(archive, "val")
    train_truth = {(t.subject_id, t.study_id): t.labels for t in study_targets_from_rows(train_rows)}
    val_truth = {(t.subject_id, t.study_id): t.labels for t in study_targets_from_rows(val_rows)}
    fold_map = {(r["Subject_id"], r["Study_id"]): int(r["fold"]) for r in
                read_index(Path(server["protocol_root"]) / "study_folds.csv")}
    patient_map = {r["Subject_id"]: int(r["fold"]) for r in
                   read_index(Path(server["protocol_root"]) / "patient_folds.csv")}
    if set(fold_map) != set(train_truth) or len(patient_map) != 11055 or any(patient_map[s] != f for (s, _), f in fold_map.items()):
        raise ValueError("Frozen Study/patient fold source differs")
    result = {"state": "INCOMPLETE", "variants": {}, "missing": []}
    pooled = {}
    for variant in ("BASE", "FUSION"):
        oof = {}; val_parts = []
        for fold in range(1, 6):
            tid = plan["fold1"][variant]["model_id"] if fold == 1 else f"P41_{variant}_F{fold:02d}"
            folder = root / ("reused_fold1" if fold == 1 else "experiments") / tid
            receipt = folder / ("REUSED.json" if fold == 1 else "DONE.json")
            if not receipt.exists():
                result["missing"].append(tid); continue
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if any(digest(folder / name) != sha for name, sha in record["sha256"].items()):
                raise ValueError(f"Finished model altered: {tid}")
            keys = {k for k, f in fold_map.items() if f == fold}
            part = matrix(folder / "oof_predictions.csv", keys)
            if any(patient_map[subject] != fold for subject, _ in part):
                raise ValueError(f"OOF patient leakage: {tid}")
            if set(oof) & set(part):
                raise ValueError("Study predicted by more than one OOF fold")
            oof.update(part)
            val_parts.append(matrix(folder / "val_predictions.csv", set(val_truth)))
        info = dict(completed_folds=len(val_parts), oof_studies=len(oof), oof_patients=len({k[0] for k in oof}))
        if oof:
            keys = sorted(oof)
            info["partial_oof"] = auc_summary(train_truth, oof, keys)
            write_csv(root / "oof" / f"{variant}_partial_oof.csv", [dict(Subject_id=k[0], Study_id=k[1],
                          Source_Fold=fold_map[k], **{f"Prob_{j+1}": oof[k][j] for j in range(10)}) for k in keys])
        if len(val_parts) == 5 and set(oof) == set(train_truth) and len(oof) == 18513:
            info["state"] = "COMPLETE"
            info["full_oof"] = auc_summary(train_truth, oof, sorted(oof))
            val = {k: np.mean([part[k] for part in val_parts], axis=0) for k in val_truth}
            info["fivefold_val"] = auc_summary(val_truth, val, sorted(val))
            write_csv(root / "oof" / f"{variant}_oof_all.csv", [dict(Subject_id=k[0], Study_id=k[1],
                          **{f"Prob_{j+1}": oof[k][j] for j in range(10)}) for k in sorted(oof)])
            write_csv(root / "validation" / f"{variant}_val_fivefold.csv", [dict(Subject_id=k[0], Study_id=k[1],
                          **{f"Prob_{j+1}": val[k][j] for j in range(10)}) for k in sorted(val)])
            pooled[variant] = oof
        else:
            info["state"] = "PARTIAL_OOF" if oof else "PENDING"
        result["variants"][variant] = info
    if set(pooled) == {"BASE", "FUSION"}:
        keys = sorted(train_truth)
        later = [k for k in keys if fold_map[k] != 1]
        result["state"] = "COMPLETE"
        result["paired_full"] = bootstrap(train_truth, pooled["BASE"], pooled["FUSION"], keys)
        result["paired_folds2_5"] = bootstrap(train_truth, pooled["BASE"], pooled["FUSION"], later)
        write_csv(report / "per_class_auc.csv", [dict(class_id=i + 1,
                  base_auc=result["variants"]["BASE"]["full_oof"]["per_class_auc"][i],
                  fusion_auc=result["variants"]["FUSION"]["full_oof"]["per_class_auc"][i]) for i in range(10)])
    write_json(report / "fivefold_summary.json", result)
    return result
