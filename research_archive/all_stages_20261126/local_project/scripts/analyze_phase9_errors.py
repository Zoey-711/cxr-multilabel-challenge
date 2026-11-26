"""Analyze frozen E03 validation errors without reading any test labels."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cxr.data.archive import archive_path, read_split_rows
from cxr.engine.b01_analysis import QUANTILES, binary_auc, read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}
FOCUS = (0, 3, 7)


def average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    """Noninterpolated AP, grouping score ties at the same threshold."""
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    return float(np.sum((tp - np.r_[0, tp[:-1]]) * tp / (ends + 1)) / positives)


def quantiles(values: np.ndarray) -> dict:
    return {f"p{int(q*100)}": float(np.quantile(values, q)) for q in QUANTILES} if len(values) else {}


def scores(y: np.ndarray, p: np.ndarray, subjects: list[str], indices: np.ndarray) -> dict:
    yy, pp = y[indices], p[indices]
    n_pos = int(yy.sum())
    return {"studies": len(indices), "positive_studies": n_pos, "negative_studies": len(indices)-n_pos,
            "patients": len({subjects[i] for i in indices}),
            "positive_patients": len({subjects[i] for i in indices if y[i] == 1}),
            "auc": binary_auc(yy, pp), "ap": average_precision(yy, pp),
            "positive_probability_quantiles": quantiles(pp[yy == 1]),
            "negative_probability_quantiles": quantiles(pp[yy == 0])}


def patient_distribution(y: np.ndarray, subjects: list[str]) -> dict:
    by_patient = defaultdict(lambda: [0, 0])
    for i, subject in enumerate(subjects):
        by_patient[subject][int(y[i])] += 1
    positives = [value[1] for value in by_patient.values() if value[1]]
    return {"patients_with_positive_study": len(positives),
            "patients_with_only_negative_studies": sum(value[1] == 0 for value in by_patient.values()),
            "patients_with_both": sum(value[0] > 0 and value[1] > 0 for value in by_patient.values()),
            "positive_studies_per_positive_patient": quantiles(np.asarray(positives))}


def verified_probabilities() -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    reference, targets, subjects, studies = None, None, None, None
    probabilities, hashes = {}, {}
    for name, run in RUNS.items():
        frozen = trio[name] if name in trio else json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
        path = run / "validation_analysis/study_predictions.csv"
        pred_hash, ckpt_hash = sha256(path), sha256(run / "best_model.pt")
        if pred_hash != frozen["validation_analysis/study_predictions.csv"]["sha256"] or ckpt_hash != frozen["best_model.pt"]["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {name}")
        p, y, ss, st = read_study_predictions(path)
        keys = list(zip(ss, st))
        if len(keys) != 859 or len(set(keys)) != 859 or len(set(st)) != 859:
            raise ValueError(f"Incorrect validation Study coverage: {name}")
        if reference is None:
            reference, subjects, studies = keys, ss, st
            order = list(range(859))
            targets = y
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(reference):
                raise ValueError(f"Subject/Study mismatch: {name}")
            order = [lookup[key] for key in reference]
            if not np.array_equal(y[order], targets):
                raise ValueError(f"Label mismatch: {name}")
        probabilities[name] = p[order]
        hashes[name] = {"prediction_sha256": pred_hash, "checkpoint_sha256": ckpt_hash}
    e01 = (probabilities["B01"] + probabilities["B02"] + probabilities["B03"]) / 3
    e03 = 0.5 * e01 + 0.5 * probabilities["M02"]
    aucs = [binary_auc(targets[:, j], e03[:, j]) for j in range(10)]
    if any(a is None for a in aucs) or abs(float(np.mean(aucs)) - 0.8099309507263751) > 1e-12:
        raise ValueError("Rebuilt E03 differs from frozen validation score")
    return e03, targets, subjects, studies, hashes


def view_groups(config: dict, subjects: list[str], studies: list[str]) -> tuple[list[str], list[int], dict, list[set[str]]]:
    rows = read_split_rows(archive_path(config), "val")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["Subject_id"], row["Study_id"])].append(row)
    if set(grouped) != set(zip(subjects, studies)):
        raise ValueError("Validation archive differs from frozen prediction studies")
    views, counts, raw, view_sets = [], [], Counter(), []
    for key in zip(subjects, studies):
        group = grouped[key]
        values = {r["ViewPosition"].strip().upper() for r in group}
        raw.update(values)
        view_sets.append(values)
        if values == {"AP"}:
            view = "AP_only"
        elif values == {"PA"}:
            view = "PA_only"
        elif "AP" in values and "PA" in values:
            view = "AP_and_PA"
        else:
            view = "other_or_mixed"
        views.append(view)
        counts.append(len(group))
    return views, counts, dict(raw), view_sets


def matrix_phi(y: np.ndarray) -> list[list[float | None]]:
    out = []
    for j in range(10):
        row = []
        for k in range(10):
            a, b = y[:, j].astype(float), y[:, k].astype(float)
            denom = np.std(a) * np.std(b)
            row.append(float(np.corrcoef(a, b)[0, 1]) if denom else None)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/phase9_e03_errors_20261017"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.config)
    if config != load_config(RUNS["M02"] / "config.yaml"):
        raise ValueError("Expected frozen M02 runtime config")
    e03, y, subjects, studies, hashes = verified_probabilities()
    views, counts, raw_views, view_sets = view_groups(config, subjects, studies)
    groups = {"all": np.arange(len(studies))}
    for label in ("AP_only", "PA_only", "AP_and_PA", "other_or_mixed"):
        groups[label] = np.flatnonzero(np.asarray(views) == label)
    groups["AP_present"] = np.asarray([i for i, value in enumerate(view_sets) if "AP" in value])
    groups["PA_present"] = np.asarray([i for i, value in enumerate(view_sets) if "PA" in value])
    groups["single_image"] = np.flatnonzero(np.asarray(counts) == 1)
    groups["multiple_images"] = np.flatnonzero(np.asarray(counts) > 1)
    focus = {}
    for j in FOCUS:
        yy, pp = y[:, j], e03[:, j]
        pairs = []
        for k in range(10):
            if k == j:
                continue
            both = int(((yy == 1) & (y[:, k] == 1)).sum())
            pairs.append({"class_id": k+1, "co_positive_studies": both,
                          "fraction_of_focus_positives": both / int(yy.sum()),
                          "label_phi": matrix_phi(y)[j][k],
                          "probability_correlation": float(np.corrcoef(pp, e03[:, k])[0, 1]),
                          "residual_correlation": float(np.corrcoef(pp-yy, e03[:, k]-y[:, k])[0, 1])})
        errors = {"false_negative_at_0_5": int(((yy == 1) & (pp < 0.5)).sum()),
                  "false_positive_at_0_5": int(((yy == 0) & (pp >= 0.5)).sum()),
                  "high_score_false_positives": [
                      {"subject_id": subjects[i], "study_id": studies[i], "probability": float(pp[i])}
                      for i in np.argsort(-pp) if yy[i] == 0][:10],
                  "low_score_false_negatives": [
                      {"subject_id": subjects[i], "study_id": studies[i], "probability": float(pp[i])}
                      for i in np.argsort(pp) if yy[i] == 1][:10]}
        focus[str(j+1)] = {"class_name": CLASS_NAMES[j],
                           "groups": {name: scores(yy, pp, subjects, idx) for name, idx in groups.items()},
                           "patient_distribution": patient_distribution(yy, subjects),
                           "cooccurrence_and_error_correlations": pairs, "errors": errors}
    all_classes = [{"class_id": j+1, "class_name": CLASS_NAMES[j],
                    **scores(y[:, j], e03[:, j], subjects, groups["all"]),
                    "patient_distribution": patient_distribution(y[:, j], subjects)} for j in range(10)]
    cooccurrence = (y.T @ y).astype(int).tolist()
    result = {"provenance": "E03 rebuilt exactly from hash-verified frozen Study probabilities; no standalone E03 validation CSV",
              "source_sha256": hashes, "studies": len(studies), "patients": len(set(subjects)),
              "macro_auc": float(np.mean([r["auc"] for r in all_classes])),
              "raw_view_study_occurrences": raw_views,
              "view_group_counts": {name: len(idx) for name, idx in groups.items()},
              "focus_classes": focus, "all_classes": all_classes,
              "label_cooccurrence_studies": cooccurrence, "label_phi": matrix_phi(y)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "e03_validation_probabilities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id"] + [f"target_{i}" for i in range(1, 11)] + [f"probability_{i}" for i in range(1, 11)])
        for i in range(len(studies)):
            writer.writerow([subjects[i], studies[i], *y[i], *e03[i]])
    with (args.output_dir / "group_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "group", "studies", "positive_studies", "negative_studies", "patients", "positive_patients", "auc", "ap"])
        for class_id, item in focus.items():
            for name, measure in item["groups"].items():
                writer.writerow([class_id, name] + [measure[k] for k in ("studies", "positive_studies", "negative_studies", "patients", "positive_patients", "auc", "ap")])
    with (args.output_dir / "all_class_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", "negative_studies", "positive_patients",
                         "negative_only_patients", "patients_with_both", "auc", "ap"])
        for item in all_classes:
            dist = item["patient_distribution"]
            writer.writerow([item["class_id"], item["class_name"], item["positive_studies"], item["negative_studies"],
                             dist["patients_with_positive_study"], dist["patients_with_only_negative_studies"],
                             dist["patients_with_both"], item["auc"], item["ap"]])
    print(json.dumps({"macro_auc": result["macro_auc"], "focus": {k: {"auc": v["groups"]["all"]["auc"], "ap": v["groups"]["all"]["ap"]} for k, v in focus.items()}, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
