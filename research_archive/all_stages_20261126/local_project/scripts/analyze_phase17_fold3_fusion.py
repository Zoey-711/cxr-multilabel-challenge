"""Score frozen Fold 3 fusion and the strictly partial Fold 1-3 OOF union."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_phase17_fold2_fusion import score
from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import read_study_predictions
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"D17": 0.25, "C17": 0.25, "S17": 0.50}
COUNTS = {1: 3687, 2: 3731, 3: 3701}


def load_fusion(fold: int, split: str):
    folder = "validation_analysis" if split == "val" else "oof_analysis"
    reference = None
    aligned = {}
    sources = {}
    for branch in WEIGHTS:
        run_id = f"{branch.lower()}_f{fold}_202610{19 if fold == 1 else 20 if fold == 2 else 21}"
        run = ROOT / "outputs/experiments" / run_id
        manifest = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
        if manifest["fold"] != fold or manifest["best_checkpoint_sha256"] != sha256(run / "best_model.pt"):
            raise ValueError(f"Changed or incomplete checkpoint: {run_id}")
        prediction_path = run / folder / "study_predictions.csv"
        if manifest["files"][f"{folder}/study_predictions.csv"]["sha256"] != sha256(prediction_path):
            raise ValueError(f"Changed prediction file: {run_id}")
        p, y, subjects, studies = read_study_predictions(prediction_path)
        keys = list(zip(subjects, studies))
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate Study: {run_id}")
        if reference is None:
            reference = (keys, y)
            aligned[branch] = p
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(reference[0]):
                raise ValueError(f"Different Study set: {run_id}")
            order = [lookup[key] for key in reference[0]]
            if not np.array_equal(y[order], reference[1]):
                raise ValueError(f"Different labels: {run_id}")
            aligned[branch] = p[order]
        sources[branch] = manifest["best_checkpoint_sha256"]
    keys, y = reference
    expected = 859 if split == "val" else COUNTS[fold]
    if len(keys) != expected:
        raise ValueError(f"Unexpected {split} coverage for Fold {fold}: {len(keys)}")
    p = sum(WEIGHTS[name] * aligned[name] for name in WEIGHTS)
    return keys, y, p, sources


def write_prediction(path: Path, keys, y, p):
    fields = ["Subject_id", "Study_id", *(f"probability_{i}" for i in range(1, 11)),
              *(f"target_{i}" for i in range(1, 11))]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows([subject, study, *probs, *targets]
                         for (subject, study), probs, targets in zip(keys, p, y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "configs/f17_densenet121_fold3_384.yaml").resolve():
        parser.error("Expected the frozen Fold 3 config")
    subset = load_config(args.config)["data"]["train_subset"]
    if subset["fold"] != 3:
        raise ValueError("Incorrect fold")
    out = ROOT / "outputs/analysis/phase17_fold3_fixed_fusion_20261021"
    if out.exists():
        raise FileExistsError(out)
    val_keys, val_y, val_p, val_sources = load_fusion(3, "val")
    oof_keys, oof_y, oof_p, oof_sources = load_fusion(3, "oof")
    if len({subject for subject, _ in oof_keys}) != 2211:
        raise ValueError("Fold 3 patient coverage differs")
    # Verify the OOF origin against the immutable patient-fold index.
    with (ROOT / "outputs/analysis/phase17_patient_folds/patient_folds.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        patient_fold = {row["Subject_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if any(patient_fold[subject] != 3 for subject, _ in oof_keys):
        raise ValueError("Fold 3 OOF patient leakage")
    union_keys, union_y, union_p = [], [], []
    for fold in (1, 2, 3):
        keys, y, p, _ = load_fusion(fold, "oof")
        if any(patient_fold[subject] != fold for subject, _ in keys):
            raise ValueError(f"Fold {fold} OOF patient leakage")
        union_keys.extend(keys)
        union_y.append(y)
        union_p.append(p)
    if len(set(union_keys)) != len(union_keys) or len({subject for subject, _ in union_keys}) != 6633:
        raise ValueError("Partial OOF duplicate or patient count mismatch")
    union_y = np.concatenate(union_y)
    union_p = np.concatenate(union_p)
    result = {"scope": "Fold 3 pilot and partial Fold 1-3 OOF only; not full five-fold OOF",
              "weights": WEIGHTS, "fold_index_sha256": subset["sha256"],
              "sources": val_sources,
              "fold3_val": score(val_p, val_y, [s for s, _ in val_keys]),
              "fold3_oof": score(oof_p, oof_y, [s for s, _ in oof_keys]),
              "partial_fold1_to_fold3_oof": score(union_p, union_y, [s for s, _ in union_keys])}
    out.mkdir(parents=True)
    write_prediction(out / "fold3_val_predictions.csv", val_keys, val_y, val_p)
    write_prediction(out / "fold3_oof_predictions.csv", oof_keys, oof_y, oof_p)
    write_prediction(out / "partial_fold1_to_fold3_oof.csv", union_keys, union_y, union_p)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key]["macro_auc"] for key in
                      ("fold3_val", "fold3_oof", "partial_fold1_to_fold3_oof")}))


if __name__ == "__main__":
    main()
