"""Audit the frozen three-fold E17 ensemble without training or Test inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_phase17_fold2_fusion import score
from analyze_phase17_fold3_fusion import load_fusion
from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import binary_auc
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"D17": 0.25, "C17": 0.25, "S17": 0.5}
RUN_DATES = {1: "20261019", 2: "20261020", 3: "20261021"}
PROB_FIELDS = [f"probability_{index}" for index in range(1, 11)]
TARGET_FIELDS = [f"target_{index}" for index in range(1, 11)]


def read_index(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_sources():
    fold_dir = ROOT / "outputs/analysis/phase17_patient_folds"
    fold_hashes = {name: sha256(fold_dir / name) for name in
                   ("patient_folds.csv", "study_folds.csv", "image_folds.csv")}
    patients = {row["Subject_id"]: int(row["fold"]) for row in read_index(fold_dir / "patient_folds.csv")}
    studies = {(row["Subject_id"], row["Study_id"]): int(row["fold"])
               for row in read_index(fold_dir / "study_folds.csv")}
    if len(patients) != 11055 or len(studies) != 18513:
        raise ValueError("Frozen patient/Study index coverage changed")
    if any(patients[subject] != fold for (subject, _), fold in studies.items()):
        raise ValueError("Patient/Study fold index conflict")
    sources = []
    for fold in (1, 2, 3):
        for branch in WEIGHTS:
            run_id = f"{branch.lower()}_f{fold}_{RUN_DATES[fold]}"
            run = ROOT / "outputs/experiments" / run_id
            manifest_path = run / "frozen_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["fold"] != fold or manifest["fold_index_sha256"] != fold_hashes:
                raise ValueError(f"Fold manifest mismatch: {run_id}")
            files = ("best_model.pt", "config.yaml",
                     "validation_analysis/study_predictions.csv", "oof_analysis/study_predictions.csv")
            for name in files:
                path = run / name
                recorded = manifest["files"][name]
                if path.stat().st_size != recorded["size_bytes"] or sha256(path) != recorded["sha256"]:
                    raise ValueError(f"Frozen file changed: {run_id}/{name}")
            if manifest["best_checkpoint_sha256"] != manifest["files"]["best_model.pt"]["sha256"]:
                raise ValueError(f"Best checkpoint mismatch: {run_id}")
            for split in ("validation_analysis", "oof_analysis"):
                path = run / split / "study_predictions.csv"
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    fields = csv.DictReader(handle).fieldnames or []
                if fields[:2] != ["Subject_id", "Study_id"] or fields[4:14] != PROB_FIELDS or fields[14:24] != TARGET_FIELDS:
                    raise ValueError(f"10-class column order changed: {path}")
            sources.append({"model": branch, "fold": fold, "run_id": run_id,
                            "manifest_sha256": sha256(manifest_path),
                            "checkpoint_sha256": manifest["best_checkpoint_sha256"],
                            "checkpoint_bytes": (run / "best_model.pt").stat().st_size,
                            "val_prediction_sha256": manifest["files"]["validation_analysis/study_predictions.csv"]["sha256"],
                            "oof_prediction_sha256": manifest["files"]["oof_analysis/study_predictions.csv"]["sha256"]})
    return patients, studies, fold_hashes, sources


def align(reference_keys, reference_labels, keys, labels, values):
    if len(keys) != len(set(keys)) or set(keys) != set(reference_keys):
        raise ValueError("Missing, extra, or duplicated Study IDs")
    lookup = {key: index for index, key in enumerate(keys)}
    order = [lookup[key] for key in reference_keys]
    if labels is not None and not np.array_equal(reference_labels, labels[order]):
        raise ValueError("Study labels disagree after ID alignment")
    matrix = values[order]
    if matrix.shape != (len(reference_keys), 10) or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Invalid aligned 10-class probabilities")
    return matrix


def read_e08(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Subject_id", "Study_id", *PROB_FIELDS]:
            raise ValueError("E08 validation class order changed")
        rows = list(reader)
    keys = [(row["Subject_id"], row["Study_id"]) for row in rows]
    values = np.asarray([[float(row[field]) for field in PROB_FIELDS] for row in rows])
    return keys, values


def paired_bootstrap(candidate, baseline, labels, keys, *, seed=42, valid_samples=2000):
    groups = {}
    for index, (subject, _) in enumerate(keys):
        groups.setdefault(subject, []).append(index)
    if len(groups) != 228:
        raise ValueError("Expected 228 validation patients")
    groups = [np.asarray(value) for value in groups.values()]
    rng = np.random.default_rng(seed)
    deltas, invalid = [], 0
    attempts_limit = valid_samples * 5
    while len(deltas) < valid_samples and len(deltas) + invalid < attempts_limit:
        chosen = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        per_class = []
        for class_index in range(10):
            y = labels[indices, class_index]
            left = binary_auc(y, candidate[indices, class_index])
            right = binary_auc(y, baseline[indices, class_index])
            if left is None or right is None:
                break
            per_class.append(left - right)
        if len(per_class) != 10:
            invalid += 1
        else:
            deltas.append(float(np.mean(per_class)))
    if len(deltas) != valid_samples:
        raise ValueError(f"Only {len(deltas)} valid patient-cluster bootstrap samples")
    lower, upper = np.quantile(deltas, (0.025, 0.975))
    return {"method": "paired patient-cluster percentile bootstrap; all Studies of each sampled patient retained",
            "seed": seed, "patients": len(groups), "valid_samples": len(deltas), "invalid_samples": invalid,
            "invalid_rule": "Discard resample if any class lacks positive or negative Studies",
            "delta_ci95": [float(lower), float(upper)]}


def write_predictions(path, keys, labels, values, folds=None):
    fields = ["Subject_id", "Study_id"] + (["Fold"] if folds is not None else []) + PROB_FIELDS + TARGET_FIELDS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, ((subject, study), probs, truth) in enumerate(zip(keys, values, labels)):
            writer.writerow([subject, study] + ([folds[index]] if folds is not None else []) +
                            [repr(float(value)) for value in probs] + [int(value) for value in truth])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "configs/f17_densenet121_fold3_384.yaml").resolve():
        parser.error("Use the frozen Fold3 config")
    if load_config(args.config)["data"]["train_subset"]["fold"] != 3:
        raise ValueError("Frozen Fold3 config changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    patients, studies, fold_hashes, sources = audit_sources()
    val_ref_keys = val_ref_labels = None
    val_models = []
    oof_keys, oof_labels, oof_models, oof_folds = [], [], [], []
    fold_metrics = {}
    for fold in (1, 2, 3):
        keys, labels, values, _ = load_fusion(fold, "val")
        if val_ref_keys is None:
            val_ref_keys, val_ref_labels = keys, labels
            aligned = values
        else:
            aligned = align(val_ref_keys, val_ref_labels, keys, labels, values)
        val_models.append(aligned)
        keys, labels, values, _ = load_fusion(fold, "oof")
        expected = {key for key, assigned in studies.items() if assigned == fold}
        if set(keys) != expected or len(keys) != len(expected):
            raise ValueError(f"Fold {fold} OOF does not cover exactly its frozen Studies")
        if any(patients[subject] != fold for subject, _ in keys):
            raise ValueError(f"Fold {fold} OOF patient was seen in training")
        oof_keys.extend(keys)
        oof_labels.append(labels)
        oof_models.append(values)
        oof_folds.extend([fold] * len(keys))
        fold_metrics[str(fold)] = {"val": score(aligned, val_ref_labels, [s for s, _ in val_ref_keys]),
                                   "oof": score(values, labels, [s for s, _ in keys])}
    if len(val_ref_keys) != 859 or len({subject for subject, _ in val_ref_keys}) != 228:
        raise ValueError("Official Val coverage differs from 859 Studies/228 patients")
    val_threefold = np.mean(np.stack(val_models), axis=0)
    if len(set(oof_keys)) != len(oof_keys):
        raise ValueError("Fold 1-3 OOF Study appears more than once")
    oof_y = np.concatenate(oof_labels)
    oof_p = np.concatenate(oof_models)
    if len(oof_keys) != 11119 or len({subject for subject, _ in oof_keys}) != 6633:
        raise ValueError("PARTIAL_OOF coverage differs from frozen folds")
    e08_dir = ROOT / "outputs/submissions/E08_EMA_20261020"
    e08_manifest = json.loads((e08_dir / "manifest.json").read_text(encoding="utf-8"))
    e08_path = e08_dir / "val_predictions.csv"
    if sha256(e08_path) != e08_manifest["files"]["val_predictions.csv"]["sha256"]:
        raise ValueError("E08 frozen Val probabilities changed")
    e08_keys, e08_values = read_e08(e08_path)
    e08 = align(val_ref_keys, val_ref_labels, e08_keys, None, e08_values)
    e08_metrics = score(e08, val_ref_labels, [s for s, _ in val_ref_keys])
    if abs(e08_metrics["macro_auc"] - e08_manifest["validation_macro_auc"]) > 1e-10:
        raise ValueError("E08 validation AUC does not reproduce frozen value")
    val_metrics = score(val_threefold, val_ref_labels, [s for s, _ in val_ref_keys])
    oof_metrics = score(oof_p, oof_y, [s for s, _ in oof_keys])
    paired = paired_bootstrap(val_threefold, e08, val_ref_labels, val_ref_keys)
    paired["delta_point"] = val_metrics["macro_auc"] - e08_metrics["macro_auc"]
    total_bytes = sum(source["checkpoint_bytes"] for source in sources)
    if total_bytes <= 500_000_000:
        raise ValueError("Unexpected nine-checkpoint size; review deployment conclusion")
    result = {"candidate": "E17_THREE_FOLD_RESEARCH_ONLY", "val_fold_weights": [1/3] * 3,
              "within_fold_weights": WEIGHTS, "fold_index_sha256": fold_hashes,
              "fold_metrics": fold_metrics,
              "val": val_metrics, "partial_oof": {"status": "PARTIAL_OOF", **oof_metrics,
                  "coverage": "Fold 1-3 only; OOF Studies concatenated by frozen Fold, never averaged"},
              "e08_val": e08_metrics, "paired_vs_e08": paired,
              "deployment": {"checkpoint_bytes": total_bytes, "limit_bytes": 500_000_000,
                  "size_status": "SIZE_FAIL", "latency_status": "LATENCY_UNVERIFIED",
                  "candidate_status": "RESEARCH_ONLY", "official_target_timing": None},
              "sources": sources, "e08_val_prediction_sha256": sha256(e08_path)}
    output.mkdir(parents=True)
    write_predictions(output / "val_predictions.csv", val_ref_keys, val_ref_labels, val_threefold)
    write_predictions(output / "partial_oof_predictions.csv", oof_keys, oof_y, oof_p, oof_folds)
    with (output / "per_class_auc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class_ID", "Class_Name", "Positive_Studies", "E17_Three_Fold_AUC", "E08_AUC", "Delta"])
        for left, right in zip(val_metrics["per_class"], e08_metrics["per_class"]):
            writer.writerow([left["class_id"], left["class_name"], left["positive_studies"],
                             left["auc"], right["auc"], left["auc"] - right["auc"]])
    result["output_sha256"] = {name: sha256(output / name) for name in
                               ("val_predictions.csv", "partial_oof_predictions.csv", "per_class_auc.csv")}
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E17 three-fold fixed ensemble", "", "Three folds use 0.25 D17 + 0.25 C17 + 0.50 S17 each; official Val is the equal mean of the three Study probability matrices.",
             "", f"Fold Val AUC: F1 {fold_metrics['1']['val']['macro_auc']:.6f}; F2 {fold_metrics['2']['val']['macro_auc']:.6f}; F3 {fold_metrics['3']['val']['macro_auc']:.6f}.",
             f"Fold OOF AUC: F1 {fold_metrics['1']['oof']['macro_auc']:.6f}; F2 {fold_metrics['2']['oof']['macro_auc']:.6f}; F3 {fold_metrics['3']['oof']['macro_auc']:.6f}.",
             f"Val: {val_metrics['macro_auc']:.6f} on 859 Studies / 228 patients.",
             f"PARTIAL_OOF: {oof_metrics['macro_auc']:.6f} on {len(oof_keys)} Studies / {oof_metrics['patients']} patients. Fold 4-5 are absent.",
             f"E08 Val: {e08_metrics['macro_auc']:.6f}; paired delta {paired['delta_point']:+.6f}, 95% CI [{paired['delta_ci95'][0]:+.6f}, {paired['delta_ci95'][1]:+.6f}] ({paired['valid_samples']} valid; {paired['invalid_samples']} invalid).",
             f"Nine inference checkpoints: {total_bytes:,} bytes ({total_bytes/1e6:.2f} MB), SIZE_FAIL against 500,000,000 bytes.",
             "Official target-device end-to-end latency: LATENCY_UNVERIFIED. Candidate: RESEARCH_ONLY; no Test inference or submission produced.",
             "Official Val has been reused for early stopping/model selection; these comparisons are not new independent-test evidence.",
             "", "Sources and full per-class results are recorded in analysis.json and per_class_auc.csv."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"val_macro_auc": val_metrics["macro_auc"], "partial_oof_macro_auc": oof_metrics["macro_auc"],
                      "partial_oof_studies": len(oof_keys), "paired_vs_e08": paired,
                      "checkpoint_bytes": total_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
