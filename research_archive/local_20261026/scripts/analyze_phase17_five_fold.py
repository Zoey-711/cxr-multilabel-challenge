"""Audit complete five-fold Phase17 OOF and fixed E17 validation fusion."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_phase17_fold2_fusion import score
from analyze_phase17_three_fold_ensemble import (
    PROB_FIELDS, TARGET_FIELDS, align, paired_bootstrap, read_e08, write_predictions,
)
from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import read_study_predictions
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"D17": 0.25, "C17": 0.25, "S17": 0.50}
RUN_DATES = {1: "20261019", 2: "20261020", 3: "20261021", 4: "20261022", 5: "20261023"}
CONFIGS = {"D17": "f17_densenet121", "C17": "c17", "S17": "s17"}
OUTPUT = ROOT / "outputs/analysis/phase17_five_fold_final_20261023"


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_index():
    directory = ROOT / "outputs/analysis/phase17_patient_folds"
    hashes = {name: sha256(directory / name) for name in
              ("patient_folds.csv", "study_folds.csv", "image_folds.csv")}
    patients = {row["Subject_id"]: int(row["fold"]) for row in rows(directory / "patient_folds.csv")}
    studies = {(row["Subject_id"], row["Study_id"]): int(row["fold"])
               for row in rows(directory / "study_folds.csv")}
    images = rows(directory / "image_folds.csv")
    if len(patients) != 11055 or len(studies) != 18513 or len(images) != 30344:
        raise ValueError("Frozen five-fold index coverage changed")
    if any(patients[subject] != fold for (subject, _), fold in studies.items()):
        raise ValueError("Patient/Study fold index conflict")
    if len({(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in images}) != len(images):
        raise ValueError("Repeated image in frozen fold index")
    if any(patients[row["Subject_id"]] != int(row["fold"]) or
           studies[(row["Subject_id"], row["Study_id"])] != int(row["fold"]) for row in images):
        raise ValueError("Image crosses its patient/Study fold")
    return patients, studies, hashes


def verified_prediction(run: Path, manifest: dict, split: str):
    rel = f"{split}/study_predictions.csv"
    path = run / rel
    recorded = manifest["files"][rel]
    if path.stat().st_size != recorded["size_bytes"] or sha256(path) != recorded["sha256"]:
        raise ValueError(f"Frozen prediction changed: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames or []
    if fields[:2] != ["Subject_id", "Study_id"] or fields[4:14] != PROB_FIELDS or fields[14:24] != TARGET_FIELDS:
        raise ValueError(f"Class order or Study columns changed: {path}")
    probability, labels, subjects, studies = read_study_predictions(path)
    keys = list(zip(subjects, studies))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate Study in {path}")
    return keys, labels, probability


def verified_run(branch: str, fold: int, hashes: dict):
    run_id = f"{branch.lower()}_f{fold}_{RUN_DATES[fold]}"
    run = ROOT / "outputs/experiments" / run_id
    manifest_path = run / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = ROOT / "configs" / f"{CONFIGS[branch]}_fold{fold}_384.yaml"
    if manifest["fold"] != fold or manifest["fold_index_sha256"] != hashes or \
            load_config(run / "config.yaml") != load_config(config_path):
        raise ValueError(f"Fold or runtime config differs: {run_id}")
    checkpoint = run / "best_model.pt"
    if checkpoint.stat().st_size != manifest["files"]["best_model.pt"]["size_bytes"] or \
            sha256(checkpoint) != manifest["best_checkpoint_sha256"]:
        raise ValueError(f"Frozen best checkpoint changed: {run_id}")
    source = {"branch": branch, "fold": fold, "run_id": run_id,
              "manifest_sha256": sha256(manifest_path),
              "checkpoint_sha256": manifest["best_checkpoint_sha256"],
              "checkpoint_bytes": checkpoint.stat().st_size,
              "best_epoch": manifest["best_epoch"],
              "val_prediction_sha256": manifest["files"]["validation_analysis/study_predictions.csv"]["sha256"],
              "oof_prediction_sha256": manifest["files"]["oof_analysis/study_predictions.csv"]["sha256"]}
    return run, manifest, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "configs/f17_densenet121_fold5_384.yaml").resolve():
        parser.error("Use the frozen Fold5 D17 configuration")
    if OUTPUT.exists():
        raise FileExistsError(f"Independent analysis directory already exists: {OUTPUT}")
    patients, studies, hashes = frozen_index()
    if load_config(args.config)["data"]["train_subset"]["sha256"] != hashes:
        raise ValueError("Frozen Fold5 index SHA-256 differs")
    sources, fold_metrics = [], {}
    val_keys = val_labels = None
    val_branch = {name: [] for name in WEIGHTS}
    oof_branch = {name: [] for name in WEIGHTS}
    oof_keys, oof_labels, oof_folds = [], [], []
    for fold in range(1, 6):
        fold_val_keys = fold_val_labels = None
        fold_oof_keys = fold_oof_labels = None
        fold_val, fold_oof = {}, {}
        for branch in WEIGHTS:
            run, manifest, source = verified_run(branch, fold, hashes)
            sources.append(source)
            keys, labels, probs = verified_prediction(run, manifest, "validation_analysis")
            if fold_val_keys is None:
                fold_val_keys, fold_val_labels = keys, labels
                fold_val[branch] = probs
            else:
                fold_val[branch] = align(fold_val_keys, fold_val_labels, keys, labels, probs)
            keys, labels, probs = verified_prediction(run, manifest, "oof_analysis")
            if fold_oof_keys is None:
                fold_oof_keys, fold_oof_labels = keys, labels
                fold_oof[branch] = probs
            else:
                fold_oof[branch] = align(fold_oof_keys, fold_oof_labels, keys, labels, probs)
        if len(fold_val_keys) != 859 or len({subject for subject, _ in fold_val_keys}) != 228:
            raise ValueError(f"Official Val coverage differs in Fold {fold}")
        if val_keys is None:
            val_keys, val_labels = fold_val_keys, fold_val_labels
        else:
            for branch in WEIGHTS:
                fold_val[branch] = align(val_keys, val_labels, fold_val_keys, fold_val_labels, fold_val[branch])
        expected = {key for key, owner in studies.items() if owner == fold}
        if set(fold_oof_keys) != expected or len(fold_oof_keys) != len(expected) or \
                any(patients[subject] != fold for subject, _ in fold_oof_keys):
            raise ValueError(f"OOF missing, duplicated or patient leakage in Fold {fold}")
        fold_val_fusion = sum(WEIGHTS[name] * fold_val[name] for name in WEIGHTS)
        fold_oof_fusion = sum(WEIGHTS[name] * fold_oof[name] for name in WEIGHTS)
        fold_metrics[str(fold)] = {
            "val": score(fold_val_fusion, val_labels, [subject for subject, _ in val_keys]),
            "oof": score(fold_oof_fusion, fold_oof_labels, [subject for subject, _ in fold_oof_keys]),
        }
        for branch in WEIGHTS:
            val_branch[branch].append(fold_val[branch])
            oof_branch[branch].append(fold_oof[branch])
        oof_keys.extend(fold_oof_keys)
        oof_labels.append(fold_oof_labels)
        oof_folds.extend([fold] * len(fold_oof_keys))
    if len(oof_keys) != 18513 or len(set(oof_keys)) != 18513 or set(oof_keys) != set(studies) or \
            len({subject for subject, _ in oof_keys}) != 11055:
        raise ValueError("Complete OOF Study/patient coverage failed")
    oof_y = np.concatenate(oof_labels)
    branch_oof = {name: np.concatenate(parts) for name, parts in oof_branch.items()}
    fusion_oof = sum(WEIGHTS[name] * branch_oof[name] for name in WEIGHTS)
    branch_oof_metrics = {name: score(value, oof_y, [subject for subject, _ in oof_keys])
                          for name, value in branch_oof.items()}
    fusion_oof_metrics = score(fusion_oof, oof_y, [subject for subject, _ in oof_keys])
    branch_val = {name: np.mean(np.stack(parts), axis=0) for name, parts in val_branch.items()}
    fusion_val = sum(WEIGHTS[name] * branch_val[name] for name in WEIGHTS)
    branch_val_metrics = {name: score(value, val_labels, [subject for subject, _ in val_keys])
                          for name, value in branch_val.items()}
    fusion_val_metrics = score(fusion_val, val_labels, [subject for subject, _ in val_keys])
    e08_dir = ROOT / "outputs/submissions/E08_EMA_20261020"
    e08_manifest = json.loads((e08_dir / "manifest.json").read_text(encoding="utf-8"))
    e08_path = e08_dir / "val_predictions.csv"
    if sha256(e08_path) != e08_manifest["files"]["val_predictions.csv"]["sha256"]:
        raise ValueError("Frozen E08 validation probabilities changed")
    e08_keys, e08_probs = read_e08(e08_path)
    e08 = align(val_keys, val_labels, e08_keys, None, e08_probs)
    e08_metrics = score(e08, val_labels, [subject for subject, _ in val_keys])
    if abs(e08_metrics["macro_auc"] - e08_manifest["validation_macro_auc"]) > 1e-10:
        raise ValueError("E08 validation AUC does not reproduce")
    paired = paired_bootstrap(fusion_val, e08, val_labels, val_keys)
    paired["delta_point"] = fusion_val_metrics["macro_auc"] - e08_metrics["macro_auc"]
    total_bytes = sum(source["checkpoint_bytes"] for source in sources)
    deployment = {"checkpoint_bytes": total_bytes, "limit_bytes": 500_000_000,
                  "size_status": "PASS" if total_bytes <= 500_000_000 else "SIZE_FAIL",
                  "latency_status": "LATENCY_UNVERIFIED", "official_target_timing": None,
                  "candidate_status": "RESEARCH_ONLY"}
    result = {"scope": "complete five-fold OOF and official validation, no Test labels or prediction",
              "weights": WEIGHTS, "fold_index_sha256": hashes, "fold_metrics": fold_metrics,
              "complete_oof": {"status": "COMPLETE_OOF", "coverage_studies": len(oof_keys),
                               "coverage_patients": len({subject for subject, _ in oof_keys}),
                               "branch": branch_oof_metrics, "E17": fusion_oof_metrics},
              "validation": {"branch_fivefold": branch_val_metrics, "E17": fusion_val_metrics,
                             "E08_EMA": e08_metrics, "paired_E17_minus_E08": paired},
              "deployment": deployment, "sources": sources,
              "e08_val_prediction_sha256": sha256(e08_path),
              "selection_bias_note": "Official Val was reused for early stopping and model selection; it is not a new independent test set."}
    OUTPUT.mkdir(parents=True)
    write_predictions(OUTPUT / "e17_val_fivefold.csv", val_keys, val_labels, fusion_val)
    write_predictions(OUTPUT / "e17_oof_complete.csv", oof_keys, oof_y, fusion_oof, oof_folds)
    for branch in WEIGHTS:
        write_predictions(OUTPUT / f"{branch.lower()}_oof_complete.csv", oof_keys, oof_y,
                          branch_oof[branch], oof_folds)
        write_predictions(OUTPUT / f"{branch.lower()}_val_fivefold.csv", val_keys, val_labels,
                          branch_val[branch])
    with (OUTPUT / "per_class_e17_vs_e08.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class_ID", "Class_Name", "Positive_Studies", "E17_Val_AUC", "E08_Val_AUC", "Delta"])
        for left, right in zip(fusion_val_metrics["per_class"], e08_metrics["per_class"]):
            writer.writerow([left["class_id"], left["class_name"], left["positive_studies"],
                             left["auc"], right["auc"], left["auc"] - right["auc"]])
    result["output_sha256"] = {path.name: sha256(path) for path in OUTPUT.glob("*.csv")}
    (OUTPUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Phase17 complete five-fold audit", "",
              "Within each Fold: 0.25 D17 + 0.25 C17 + 0.50 S17. Official Val: equal average of all five Fold probability matrices.",
              "OOF: each Study is predicted only by its own withheld Fold model; Fold OOF is concatenated, never averaged.", "",
              f"Fold5 E17 Val AUC: {fold_metrics['5']['val']['macro_auc']:.6f}; Fold5 OOF AUC: {fold_metrics['5']['oof']['macro_auc']:.6f}.",
              f"Complete E17 OOF AUC: {fusion_oof_metrics['macro_auc']:.6f} on 18,513 Studies / 11,055 patients.",
              f"Five-fold E17 Val AUC: {fusion_val_metrics['macro_auc']:.6f} on 859 Studies / 228 patients.",
              f"E08 Val AUC: {e08_metrics['macro_auc']:.6f}; paired E17-E08 delta {paired['delta_point']:+.6f}, 95% CI [{paired['delta_ci95'][0]:+.6f}, {paired['delta_ci95'][1]:+.6f}].",
              f"Inference weights: {total_bytes:,} bytes; {deployment['size_status']} against 500,000,000 bytes; target-device latency LATENCY_UNVERIFIED; RESEARCH_ONLY.",
              "Official Val was reused for early stopping and model selection; these comparisons are not independent-test evidence.",
              "No Test inference or submission was generated."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"fold5_val_auc": fold_metrics["5"]["val"]["macro_auc"],
                      "fold5_oof_auc": fold_metrics["5"]["oof"]["macro_auc"],
                      "complete_oof_auc": fusion_oof_metrics["macro_auc"],
                      "fivefold_val_auc": fusion_val_metrics["macro_auc"],
                      "paired_vs_e08": paired, "checkpoint_bytes": total_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
