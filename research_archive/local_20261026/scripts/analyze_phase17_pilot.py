"""Analyze the fixed three-backbone Fold 1 fusion on OOF and official Val."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import read_study_predictions
from cxr.engine.average_precision import average_precision
from cxr.engine.metrics import binary_auc
from cxr.data.patient_folds import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


NAMES = ("D17-F1", "C17-F1", "S17-F1")
WEIGHTS = {"D17-F1": 0.25, "C17-F1": 0.25, "S17-F1": 0.5}


def load_aligned(paths: dict[str, Path]):
    arrays, hashes = {}, {}
    reference_keys = reference_labels = reference_subjects = None
    for name in NAMES:
        path = paths[name]
        p, y, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate Study key in {name}")
        if reference_keys is None:
            reference_keys, reference_labels, reference_subjects = keys, y, subjects
            order = np.arange(len(keys))
        else:
            lookup = {key: index for index, key in enumerate(keys)}
            if set(lookup) != set(reference_keys):
                raise ValueError(f"Subject/Study keys differ: {name}")
            order = np.asarray([lookup[key] for key in reference_keys])
            if not np.array_equal(y[order], reference_labels):
                raise ValueError(f"Study labels differ: {name}")
        arrays[name] = p[order]
        hashes[name] = sha256(path)
    arrays["E17-F1"] = sum(WEIGHTS[name] * arrays[name] for name in NAMES)
    return arrays, reference_labels, reference_subjects, hashes


def scores(probabilities, labels, subjects):
    if not np.isfinite(probabilities).all() or probabilities.shape != labels.shape or labels.shape[1] != 10:
        raise ValueError("Invalid probability/target matrix")
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        positive = labels[:, index] == 1
        if not positive.any() or positive.all():
            raise ValueError(f"No positive/negative Studies in class {index + 1}")
        per_class.append({"class_id": index + 1, "class_name": class_name,
                          "positive_studies": int(positive.sum()),
                          "positive_patients": len({subject for subject, flag in zip(subjects, positive) if flag}),
                          "auc": float(binary_auc(labels[:, index], probabilities[:, index])),
                          "ap": average_precision(labels[:, index], probabilities[:, index])})
    predicted = probabilities >= 0.5
    tp = np.sum((labels == 1) & predicted, axis=0)
    fp = np.sum((labels == 0) & predicted, axis=0)
    fn = np.sum((labels == 1) & ~predicted, axis=0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(10, dtype=float), where=(2 * tp + fp + fn) > 0)
    return {"patient_count": len(set(subjects)), "study_count": len(labels),
            "macro_auc": float(np.mean([item["auc"] for item in per_class])),
            "macro_ap": float(np.mean([item["ap"] for item in per_class])),
            "macro_f1_at_0_5": float(np.mean(f1)),
            "per_class": per_class}


def patient_bootstrap(arrays, labels, subjects, *, draws=2000, seed=42):
    groups = defaultdict(list)
    for index, subject in enumerate(subjects):
        groups[subject].append(index)
    patients = [np.asarray(indices, dtype=np.int32) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    names = list(NAMES) + ["E17-F1"]
    replicates = {name: [] for name in names}
    invalid = 0
    while len(replicates[names[0]]) < draws:
        if invalid > draws * 20:
            raise RuntimeError("Too many invalid bootstrap draws")
        index = np.concatenate([patients[position] for position in rng.integers(0, len(patients), size=len(patients))])
        y = labels[index]
        if any(y[:, col].min() == y[:, col].max() for col in range(10)):
            invalid += 1
            continue
        for name in names:
            p = arrays[name][index]
            replicates[name].append(float(np.mean([binary_auc(y[:, col], p[:, col]) for col in range(10)])))
    return {"method": "patient cluster bootstrap; retain all Studies from each sampled patient",
            "valid_draws": draws, "invalid_draws": invalid,
            "auc_ci95": {name: [float(np.quantile(values, q)) for q in (0.025, 0.975)]
                         for name, values in replicates.items()},
            "fusion_minus_model_ci95": {name: [float(np.quantile(np.asarray(replicates["E17-F1"]) - values, q))
                                              for q in (0.025, 0.975)]
                                        for name, values in replicates.items() if name != "E17-F1"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--d-run-dir", required=True, type=Path)
    parser.add_argument("--c-run-dir", required=True, type=Path)
    parser.add_argument("--s-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    load_config(args.config)
    runs = dict(zip(NAMES, (args.d_run_dir, args.c_run_dir, args.s_run_dir)))
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"Pilot analysis directory already exists: {out}")
    result = {"fixed_weights": WEIGHTS, "scope": "Fold 1 pilot only; not a full five-fold OOF result",
              "sources": {}, "splits": {}, "resources": {}}
    for name, run in runs.items():
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status["state"] != "complete":
            raise ValueError(f"Incomplete run: {name}")
        result["sources"][name] = {"checkpoint_sha256": sha256(run / "best_model.pt"),
                                    "config_sha256": sha256(run / "config.yaml")}
        epoch_records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        result["resources"][name] = {
            "trained_epochs": len(epoch_records),
            "training_hours": sum(record["elapsed_seconds"] for record in epoch_records) / 3600,
            "peak_gpu_allocated_mib": max(record["gpu_peak_allocated_mib"] for record in epoch_records),
            "peak_gpu_reserved_mib": max(record["gpu_peak_reserved_mib"] for record in epoch_records),
            "optimizer_steps": sum(record["optimizer_steps"] for record in epoch_records),
            "amp_skipped_steps": sum(record["amp_skipped_steps"] for record in epoch_records),
            "best_checkpoint_bytes": (run / "best_model.pt").stat().st_size,
            "best_epoch": status["best_epoch"], "stop_epoch": status["stop_epoch"],
            "stop_reason": status["stop_reason"]}
    for split, folder, expected_studies, expected_patients in (("oof", "oof_analysis", 3687, 2211),
                                                                ("official_val", "validation_analysis", 859, 228)):
        paths = {name: runs[name] / folder / "study_predictions.csv" for name in NAMES}
        arrays, labels, subjects, hashes = load_aligned(paths)
        if len(labels) != expected_studies or len(set(subjects)) != expected_patients:
            raise ValueError(f"Unexpected {split} coverage")
        scores_by_name = {name: scores(probabilities, labels, subjects) for name, probabilities in arrays.items()}
        bootstrap = patient_bootstrap(arrays, labels, subjects) if split == "oof" else None
        result["splits"][split] = {"scores": scores_by_name, "bootstrap": bootstrap,
                                    "probability_sha256": hashes}
    sizes = {name: item["best_checkpoint_bytes"] for name, item in result["resources"].items()}
    d, c, s = (sizes[name] for name in NAMES)
    e07_ids = ("b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated",
               "b03_densenet121_lr3e5_20261016", "m02_convnext_tiny_384_aug_20261016_elevated",
               "t01_swin_tiny_384_20261018")
    e07_size = sum((Path("outputs/experiments") / name / "best_model.pt").stat().st_size for name in e07_ids)
    result["deployment_budget"] = {
        "limit_bytes": 500_000_000, "e07_actual_bytes": e07_size,
        "fifteen_models_projected_bytes": 5 * (d + c + s),
        "size_only_options_projected_bytes": {"D5_C1_S1": 5*d + c + s,
                                              "D5_C2_S1": 5*d + 2*c + s,
                                              "D5_C1_S2": 5*d + c + 2*s},
        "caution": "Fold 2-5 checkpoint bytes and official T4 latency are not yet measured"}
    out.mkdir(parents=True)
    (out / "pilot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {name: item["macro_auc"] for name, item in data["scores"].items()}
                      for split, data in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
