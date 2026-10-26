"""One final, locked Internal Holdout evaluation for Phase 15."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.data.internal_split import select_internal_train_rows
from cxr.engine.metrics import binary_auc
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-p, kind="mergesort")
    scores, labels = p[order], y[order]
    positive_cumulative = np.cumsum(labels)
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    previous_recall = 0.0
    ap = 0.0
    for end in ends:
        recall = positive_cumulative[end] / positives
        ap += (recall - previous_recall) * positive_cumulative[end] / (end + 1)
        previous_recall = recall
    return float(ap)


def measures(probabilities: np.ndarray, labels: np.ndarray, subjects: list[str]) -> dict:
    if probabilities.shape != labels.shape or labels.ndim != 2 or labels.shape[1] != 10:
        raise ValueError("Expected aligned [Study,10] arrays")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Holdout labels must be complete binary targets")
    details = []
    for j, name in enumerate(CLASS_NAMES):
        y, p = labels[:, j], probabilities[:, j]
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        pred = p >= 0.5
        tp, fp, fn = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
        details.append({
            "class_id": j + 1, "class_name": name,
            "positive_studies": positives, "negative_studies": negatives,
            "positive_patients": len({subject for subject, label in zip(subjects, y) if label}),
            "auc": binary_auc(y, p) if positives and negatives else None,
            "ap": average_precision(y, p),
            "f1_at_0_5": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        })
    aucs = [row["auc"] for row in details if row["auc"] is not None]
    aps = [row["ap"] for row in details if row["ap"] is not None]
    return {"study_count": len(labels), "patient_count": len(set(subjects)),
            "macro_auc": float(np.mean(aucs)) if aucs else None,
            "macro_auc_defined_classes": len(aucs),
            "macro_ap": float(np.mean(aps)) if aps else None,
            "macro_f1_at_0_5": float(np.mean([row["f1_at_0_5"] for row in details])),
            "per_class": details}


def bootstrap(scores: dict[str, np.ndarray], labels: np.ndarray, subjects: list[str],
              *, replicates: int = 2000, seed: int = 42) -> dict:
    groups = defaultdict(list)
    for i, subject in enumerate(subjects):
        groups[subject].append(i)
    patient_indices = [np.asarray(indices, dtype=np.int32) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    estimates = {name: [] for name in scores}
    differences = {"fusion_minus_A": [], "fusion_minus_B": [], "B_minus_A": []}
    invalid = 0
    attempts_limit = max(10000, replicates * 20)
    while len(estimates["A"]) < replicates:
        if len(estimates["A"]) + invalid >= attempts_limit:
            raise RuntimeError("Too many invalid patient bootstrap draws")
        drawn = rng.integers(0, len(patient_indices), size=len(patient_indices))
        indices = np.concatenate([patient_indices[index] for index in drawn])
        y = labels[indices]
        if any(y[:, j].min() == y[:, j].max() for j in range(10)):
            invalid += 1
            continue
        aucs = {name: float(np.mean([binary_auc(y[:, j], p[indices, j]) for j in range(10)]))
                for name, p in scores.items()}
        for name, value in aucs.items():
            estimates[name].append(value)
        differences["fusion_minus_A"].append(aucs["fusion"] - aucs["A"])
        differences["fusion_minus_B"].append(aucs["fusion"] - aucs["B"])
        differences["B_minus_A"].append(aucs["B"] - aucs["A"])

    def interval(values: list[float]) -> list[float]:
        return [float(np.quantile(values, q)) for q in (0.025, 0.975)]

    return {"method": "paired patient-cluster percentile bootstrap, all Studies per sampled Subject_id",
            "patient_count": len(patient_indices), "seed": seed,
            "valid_replicates": replicates, "invalid_resamples": invalid,
            "invalid_rule": "any class lacks positive or negative Studies",
            "individual_macro_auc_ci95": {name: interval(values) for name, values in estimates.items()},
            "paired_difference_ci95": {name: interval(values) for name, values in differences.items()}}


def infer(checkpoint: Path, config: dict, archive: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, list[ImagePrediction]]:
    model = load_model_state(checkpoint, device="cuda")
    loader = make_dataloader(archive, "train", rows=rows,
                             batch_size=int(config["training"]["batch_size"]),
                             num_workers=int(config["data"]["num_workers"]), shuffle=False,
                             augmentation=None, preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                             input_size=int(config["data"]["input_size"]), unlisted_policy="negative")
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                probs = torch.sigmoid(model(batch["image"].to("cuda", non_blocking=True))).float().cpu().numpy()
            for i, vector in enumerate(probs):
                predictions.append(ImagePrediction(batch["subject_id"][i], batch["study_id"][i],
                                                   batch["image_id"][i], tuple(map(float, vector))))
    loader.dataset.close()
    del model
    torch.cuda.empty_cache()
    study_predictions = aggregate_mean_probabilities(predictions)
    return np.asarray([item.probabilities for item in study_predictions]), predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != digest(args.config) or lock["fusion"] != {"A": 0.5, "B": 0.5}:
        raise ValueError("Selection lock does not match the fixed approved protocol")
    if args.output_dir.exists():
        raise FileExistsError("Holdout evaluation directory already exists; automatic re-evaluation forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    configs = {name: load_config(ROOT / protocol["runs"][name]["config"]) for name in ("A", "B")}
    for name in ("A", "B"):
        run = ROOT / "outputs" / "experiments" / protocol["runs"][name]["run_id"]
        frozen = json.loads((run / "phase15_freeze_manifest.json").read_text(encoding="utf-8"))
        if frozen != lock["models"][name] or digest(run / "best_model.pt") != frozen["checkpoint_sha256"]:
            raise ValueError(f"Frozen model {name} does not match selection lock")
    archive = archive_path(configs["A"])
    if archive != archive_path(configs["B"]):
        raise ValueError("Two models use different official archives")
    original = read_split_rows(archive, "train")
    internal_train = select_internal_train_rows(original, configs["A"]["data"]["train_subset"])
    train_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"]) for r in internal_train}
    holdout = [row for row in original if (row["Subject_id"], row["Study_id"], row["Dicom_id"]) not in train_keys]
    targets = study_targets_from_rows(holdout, unlisted_policy="negative")
    keys = [(target.subject_id, target.study_id) for target in targets]
    if (len(holdout), len(targets), len({target.subject_id for target in targets})) != (3043, 1846, 1106):
        raise ValueError("Internal Holdout coverage differs from frozen split")
    if {r["Subject_id"] for r in internal_train} & {r["Subject_id"] for r in holdout}:
        raise ValueError("Internal Train/Holdout patient overlap")
    subjects = [target.subject_id for target in targets]
    labels = np.asarray([target.labels for target in targets], dtype=np.int8)
    image_counts = defaultdict(int)
    for row in holdout:
        image_counts[(row["Subject_id"], row["Study_id"])] += 1
    args.output_dir.mkdir(parents=True)
    scores = {}
    for name in ("A", "B"):
        run = ROOT / "outputs" / "experiments" / protocol["runs"][name]["run_id"]
        p, image_predictions = infer(run / "best_model.pt", configs[name], archive, holdout)
        actual_keys = [(item.subject_id, item.study_id) for item in aggregate_mean_probabilities(image_predictions)]
        if actual_keys != keys or len(image_predictions) != 3043 or p.shape != (1846, 10):
            raise ValueError(f"{name} Holdout Study alignment failed")
        scores[name] = p
        write_csv(args.output_dir / f"{name}_image_probabilities.csv", [
            {"Subject_id": item.subject_id, "Study_id": item.study_id, "Dicom_id": item.image_id,
             **{f"probability_{j+1}": item.probabilities[j] for j in range(10)}} for item in image_predictions])
        print(f"HOLDOUT_{name}_INFERRED images={len(image_predictions)} studies={len(keys)}", flush=True)
    scores["fusion"] = 0.5 * scores["A"] + 0.5 * scores["B"]
    for name, p in scores.items():
        write_csv(args.output_dir / f"{name}_study_probabilities.csv", [
            {"Subject_id": subject, "Study_id": study, "image_count": image_counts[(subject, study)],
             **{f"probability_{j+1}": float(p[i, j]) for j in range(10)},
             **{f"target_{j+1}": int(labels[i, j]) for j in range(10)}}
            for i, (subject, study) in enumerate(keys)])
    overall = {name: measures(p, labels, subjects) for name, p in scores.items()}
    subgroups = {}
    for group_name, predicate in (("single_image", lambda n: n == 1), ("multi_image", lambda n: n > 1)):
        indices = np.asarray([i for i, key in enumerate(keys) if predicate(image_counts[key])])
        subgroups[group_name] = {name: measures(p[indices], labels[indices], [subjects[i] for i in indices])
                                 for name, p in scores.items()}
    boot = bootstrap(scores, labels, subjects,
                     replicates=int(protocol["bootstrap"]["replicates"]),
                     seed=int(protocol["bootstrap"]["seed"]))
    result = {"scope": "independent internal patient Holdout; not official validation or leaderboard",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "holdout_images": len(holdout), "holdout_studies": len(keys), "holdout_patients": len(set(subjects)),
              "official_validation_macro_auc": {name: lock["models"][name]["official_val_macro_auc"] for name in ("A", "B")},
              "scores": overall, "subgroups": subgroups, "patient_bootstrap": boot,
              "paired_point_differences": {
                  "fusion_minus_A": overall["fusion"]["macro_auc"] - overall["A"]["macro_auc"],
                  "fusion_minus_B": overall["fusion"]["macro_auc"] - overall["B"]["macro_auc"],
                  "B_minus_A": overall["B"]["macro_auc"] - overall["A"]["macro_auc"]}}
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Phase 15: independent-patient Holdout evaluation", "",
             f"Internal Holdout: {len(set(subjects))} patients, {len(keys)} Studies, {len(holdout)} images.",
             "These patients were excluded from both new training runs. Model selection used only "
             "the official 859-Study validation set. This Holdout was scored once after checkpoint "
             "and fusion choices were locked.", "",
             "| Scheme | Holdout Macro-AUC | Patient bootstrap 95% CI | Macro-AP | Macro-F1@0.5 |",
             "|---|---:|---:|---:|---:|"]
    for name in ("A", "B", "fusion"):
        item = overall[name]
        ci = boot["individual_macro_auc_ci95"][name]
        lines.append(f"| {name} | {item['macro_auc']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] | "
                     f"{item['macro_ap']:.5f} | {item['macro_f1_at_0_5']:.5f} |")
    lines.extend(["", f"Bootstrap: {boot['valid_replicates']} valid paired patient draws; "
                  f"{boot['invalid_resamples']} invalid draws excluded.", ""])
    for name, point in result["paired_point_differences"].items():
        ci = boot["paired_difference_ci95"][name]
        lines.append(f"- {name}: {point:+.5f}; 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}].")
    lines.extend(["", "Ten-class AUC/AP, positive patient support, and single/multi-image Study "
                  "results are in `result.json`. Subgroup comparisons are descriptive because class "
                  "composition may differ.", "", "## Official validation used for model selection", "",
                  f"P15A Macro-AUC {lock['models']['A']['official_val_macro_auc']:.5f}; "
                  f"P15B Macro-AUC {lock['models']['B']['official_val_macro_auc']:.5f}; "
                  f"fixed fusion Macro-AUC {lock['official_validation_fixed_fusion']['fixed_fusion_macro_auc']:.5f}. "
                  "These are selection-set results and must not be described as internal Holdout estimates.",
                  "", "## Existing leaderboard result", "",
                  "E01 previously received displayed official score 75.51. No Phase 15 model was "
                  "run on hidden test data or submitted. This leaderboard score is distinct from "
                  "both local official-validation and internal-Holdout metrics.", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"selection_lock_sha256": digest(args.lock),
                "files_sha256": {path.name: digest(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: result["scores"][name]["macro_auc"] for name in ("A", "B", "fusion")}), flush=True)


if __name__ == "__main__":
    main()
