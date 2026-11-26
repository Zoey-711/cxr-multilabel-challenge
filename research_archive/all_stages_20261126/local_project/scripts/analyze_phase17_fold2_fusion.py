"""Evaluate the predeclared D17/C17/S17 Fold 2 fusion, without refitting weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import read_study_predictions
from cxr.engine.metrics import binary_auc
from cxr.engine.average_precision import average_precision
from cxr.utils.labels import CLASS_NAMES


RUNS = {"D17-F2": "d17_f2_20261020", "C17-F2": "c17_f2_20261020",
        "S17-F2": "s17_f2_20261020"}
WEIGHTS = {"D17-F2": 0.25, "C17-F2": 0.25, "S17-F2": 0.50}


def score(probabilities: np.ndarray, labels: np.ndarray, subjects: list[str]) -> dict:
    if probabilities.shape != labels.shape or labels.shape[1] != 10 or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability or label matrix")
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        y = labels[:, index]
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError(f"Class {index + 1} has no positive/negative support")
        per_class.append({"class_id": index + 1, "class_name": name,
                          "positive_studies": int(np.sum(y == 1)),
                          "positive_patients": len({subject for subject, positive in zip(subjects, y) if positive}),
                          "auc": float(binary_auc(y, probabilities[:, index])),
                          "ap": float(average_precision(y, probabilities[:, index]))})
    pred = probabilities >= 0.5
    tp = np.sum((labels == 1) & pred, axis=0)
    fp = np.sum((labels == 0) & pred, axis=0)
    fn = np.sum((labels == 1) & ~pred, axis=0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(10), where=(2 * tp + fp + fn) > 0)
    return {"patients": len(set(subjects)), "studies": len(labels),
            "macro_auc": float(np.mean([row["auc"] for row in per_class])),
            "macro_ap": float(np.mean([row["ap"] for row in per_class])),
            "macro_f1_at_0_5": float(np.mean(f1)), "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve() != (root / "configs/f17_densenet121_fold2_384.yaml").resolve():
        parser.error("Expected the frozen D17 Fold 2 configuration")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"Fusion output already exists: {out}")
    result = {"weights": WEIGHTS, "scope": "Fold 2 pilot only; not complete five-fold OOF",
              "splits": {}, "sources": {}, "resources": {}}
    for model_id, run_id in RUNS.items():
        run = root / "outputs/experiments" / run_id
        manifest = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
        if manifest["fold"] != 2 or manifest["best_checkpoint_sha256"] != sha256(run / "best_model.pt"):
            raise ValueError(f"Unfrozen or changed checkpoint: {model_id}")
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        result["sources"][model_id] = manifest["best_checkpoint_sha256"]
        result["resources"][model_id] = {"best_epoch": status["best_epoch"],
                                         "stop_epoch": status["stop_epoch"],
                                         "stop_reason": status["stop_reason"],
                                         "training_hours": sum(row["elapsed_seconds"] for row in records) / 3600,
                                         "peak_gpu_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in records),
                                         "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in records)}
    for split, folder, expected_studies, expected_patients in (
            ("official_val", "validation_analysis", 859, 228),
            ("oof", "oof_analysis", 3731, 2211)):
        reference = None
        arrays = {}
        for model_id, run_id in RUNS.items():
            path = root / "outputs/experiments" / run_id / folder / "study_predictions.csv"
            probs, labels, subjects, studies = read_study_predictions(path)
            keys = list(zip(subjects, studies))
            if len(set(keys)) != len(keys):
                raise ValueError(f"Duplicate Study in {model_id} {split}")
            if reference is None:
                reference = (keys, labels, subjects)
                aligned = probs
            else:
                lookup = {key: i for i, key in enumerate(keys)}
                if set(lookup) != set(reference[0]):
                    raise ValueError(f"Study mismatch in {model_id} {split}")
                order = [lookup[key] for key in reference[0]]
                if not np.array_equal(labels[order], reference[1]):
                    raise ValueError(f"Label mismatch in {model_id} {split}")
                aligned = probs[order]
            arrays[model_id] = aligned
        keys, labels, subjects = reference
        if len(keys) != expected_studies or len(set(subjects)) != expected_patients:
            raise ValueError(f"Unexpected {split} coverage")
        arrays["E17-F2"] = sum(WEIGHTS[name] * arrays[name] for name in RUNS)
        result["splits"][split] = {name: score(matrix, labels, subjects) for name, matrix in arrays.items()}
    f1 = json.loads((root / "outputs/analysis/phase17_fold1_fixed_fusion_20261019/pilot_summary.json").read_text(encoding="utf-8"))
    result["fold1_reference"] = {
        "official_val_macro_auc": f1["splits"]["official_val"]["scores"]["E17-F1"]["macro_auc"],
        "fold1_oof_macro_auc": f1["splits"]["oof"]["scores"]["E17-F1"]["macro_auc"],
        "caution": "Fold 1 and Fold 2 OOF sets differ; do not average their AUCs as full OOF AUC"}
    out.mkdir(parents=True)
    (out / "fold2_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Phase 17 Fold 2 fixed fusion", "",
             "All results below use the predeclared 0.25 D17 + 0.25 C17 + 0.50 S17 weights.", "",
             "| Model | Best epoch | Stop epoch | Val Macro-AUC | Fold 2 OOF Macro-AUC | Training hours |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*RUNS, "E17-F2"):
        resource = result["resources"].get(name, {})
        val = result["splits"]["official_val"][name]["macro_auc"]
        oof = result["splits"]["oof"][name]["macro_auc"]
        lines.append(f"| {name} | {resource.get('best_epoch', '—')} | {resource.get('stop_epoch', '—')} | "
                     f"{val:.5f} | {oof:.5f} | {resource.get('training_hours', 0):.2f} |")
    lines += ["", f"Fold 1 fixed fusion: Val {result['fold1_reference']['official_val_macro_auc']:.5f}; "
              f"Fold 1 OOF {result['fold1_reference']['fold1_oof_macro_auc']:.5f}.",
              "Fold 1 and Fold 2 OOF have different patient sets. Their AUCs are not averaged as full OOF AUC.",
              "Official Val has been used repeatedly for checkpoint selection and is not an independent test set.", ""]
    (out / "fold2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({split: {name: scores["macro_auc"] for name, scores in items.items()}
                      for split, items in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
