"""Read-only, study-level diagnostics for saved validation predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cxr.utils.labels import CLASS_NAMES


THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def read_study_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty study prediction file")
    subjects = [row["Subject_id"] for row in rows]
    studies = [row["Study_id"] for row in rows]
    if any(not subject or not study for subject, study in zip(subjects, studies)):
        raise ValueError("Blank subject or study identifier")
    if len(set(zip(subjects, studies))) != len(rows):
        raise ValueError("Duplicate study prediction")
    probabilities = np.asarray([[float(row[f"probability_{i}"]) for i in range(1, 11)] for row in rows])
    targets = np.asarray([[int(row[f"target_{i}"]) for i in range(1, 11)] for row in rows])
    if probabilities.shape != targets.shape or probabilities.shape[1] != 10:
        raise ValueError("Expected ten classes for each study")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must be complete binary labels")
    return probabilities, targets, subjects, studies


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware AUC; return None when a resample has one label value."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average_ranks = starts + (counts + 1) / 2
    ranks = np.empty(len(labels), dtype=np.float64)
    ranks[order] = np.repeat(average_ranks, counts)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_analysis(probabilities: np.ndarray, targets: np.ndarray) -> tuple[list[dict], list[dict]]:
    summaries, class_rows = [], []
    for threshold in THRESHOLDS:
        predicted = probabilities >= threshold
        per_class = []
        for index, name in enumerate(CLASS_NAMES):
            truth, guess = targets[:, index] == 1, predicted[:, index]
            tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            row = {"threshold": threshold, "class_id": index + 1, "class_name": name,
                   "positive_studies": int(truth.sum()), "negative_studies": int((~truth).sum()),
                   "predicted_positive_studies": int(guess.sum()), "precision": precision,
                   "recall": recall, "f1": f1}
            row.update({f"p{int(q * 100)}": float(np.quantile(probabilities[:, index], q)) for q in QUANTILES})
            per_class.append(row)
            class_rows.append(row)
        summaries.append({"threshold": threshold, "macro_f1": float(np.mean([r["f1"] for r in per_class])),
                          "macro_precision": float(np.mean([r["precision"] for r in per_class])),
                          "macro_recall": float(np.mean([r["recall"] for r in per_class])),
                          "zero_predicted_class_studies": int((predicted.sum(axis=1) == 0).sum()),
                          "study_count": len(targets)})
    return summaries, class_rows


def patient_bootstrap_auc(probabilities: np.ndarray, targets: np.ndarray, subjects: list[str],
                          *, replicates: int = 2000, seed: int = 42) -> dict:
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    groups: dict[str, list[int]] = {}
    for index, subject in enumerate(subjects):
        groups.setdefault(subject, []).append(index)
    patient_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    accepted, invalid = [], 0
    for _ in range(replicates):
        sampled = rng.integers(0, len(patient_indices), size=len(patient_indices))
        indices = np.concatenate([patient_indices[index] for index in sampled])
        aucs = [binary_auc(targets[indices, class_index], probabilities[indices, class_index])
                for class_index in range(10)]
        if any(value is None for value in aucs):
            invalid += 1
            continue
        accepted.append(float(np.mean(aucs)))
    if len(accepted) < 100:
        raise ValueError("Too few valid ten-class patient bootstrap replicates")
    point_aucs = [binary_auc(targets[:, index], probabilities[:, index]) for index in range(10)]
    if any(value is None for value in point_aucs):
        raise ValueError("Original validation set lacks both labels for a class")
    lower, upper = np.quantile(accepted, (0.025, 0.975))
    return {"method": "patient cluster bootstrap; sample patients with replacement and retain all their studies",
            "seed": seed, "requested_replicates": replicates, "valid_replicates": len(accepted),
            "invalid_replicates_excluded": invalid, "invalid_rule": "Exclude a replicate if any of the ten classes has no positive or no negative studies; never impute AUC",
            "patient_count": len(patient_indices), "point_macro_auc": float(np.mean(point_aucs)),
            "ci_95_percentile_lower": float(lower), "ci_95_percentile_upper": float(upper)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def epoch_curves_svg(records: list[dict], path: Path) -> None:
    """Dependency-free SVG with AUC and loss on separate vertical scales."""
    width, height = 1000, 620
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="500" y="34" text-anchor="middle" font-size="20" font-family="Arial">B01 training loss and Study-level validation AUC</text>']
    epochs = [int(row["epoch"]) for row in records]
    x_at = lambda epoch: 85 + (epoch - min(epochs)) * 840 / max(1, max(epochs) - min(epochs))
    for top, title, key, color, low, high in ((65, "Study-level Macro-AUC", "auc", "#1463a5", 0.70, 0.81),
                                               (345, "Training BCE loss", "loss", "#bd4a27", 0.0, 0.40)):
        bottom = top + 210
        y_at = lambda value: bottom - (value - low) * 190 / (high - low)
        parts.extend([f'<text x="85" y="{top-15}" font-size="16" font-family="Arial">{title}</text>',
                      f'<path d="M85 {top+20} V{bottom} H925" fill="none" stroke="#555"/>'])
        for tick in np.linspace(low, high, 5):
            y = y_at(tick)
            parts.append(f'<text x="76" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{tick:.2f}</text>')
        points = " ".join(f'{x_at(int(row["epoch"])):.1f},{y_at(row[key]):.1f}' for row in records)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in records:
            x, y = x_at(int(row["epoch"])), y_at(row[key])
            parts.extend([f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>',
                          f'<text x="{x:.1f}" y="{bottom+19}" text-anchor="middle" font-size="11" font-family="Arial">{row["epoch"]}</text>'])
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def read_epoch_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or [row["epoch"] for row in records] != list(range(1, len(records) + 1)):
        raise ValueError("Epoch records are missing or out of order")
    return [{"epoch": row["epoch"], "loss": row["train_loss"], "auc": row["val"]["macro_auc"],
             "f1": row["val"]["macro_f1"], "elapsed_seconds": row["elapsed_seconds"]} for row in records]
