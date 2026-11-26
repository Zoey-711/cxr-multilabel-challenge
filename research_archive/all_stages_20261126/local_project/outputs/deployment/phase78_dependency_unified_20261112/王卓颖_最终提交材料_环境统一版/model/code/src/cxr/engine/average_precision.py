"""Tie-aware average precision for complete binary competition labels."""

import numpy as np


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probabilities.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("AP requires equally sized binary targets and probabilities")
    positives = int(labels.sum())
    if not positives or not np.isfinite(probabilities).all():
        raise ValueError("AP requires positives and finite probabilities")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels, sorted_scores = labels[order], probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp = np.cumsum(sorted_labels)[ends]
    return float(np.sum((tp - np.r_[0, tp[:-1]]) * tp / (ends + 1)) / positives)
