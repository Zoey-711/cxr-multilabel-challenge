"""Study-level multilabel metrics after image probability fusion."""

import numpy as np

from cxr.engine.validate import StudyTarget, prepare_study_validation
from cxr.inference.aggregate import ImagePrediction
from cxr.utils.labels import CLASS_NAMES


def binary_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    y = np.asarray(targets, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("AUC requires equal-length finite one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("AUC targets must be binary")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both positive and negative studies")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(y), dtype=np.float64)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def study_multilabel_metrics(image_predictions: list[ImagePrediction], targets: list[StudyTarget],
                             *, threshold: float = 0.5) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    pairs = prepare_study_validation(image_predictions, targets)
    if not pairs:
        raise ValueError("No study predictions")
    probabilities = np.asarray([p for p, _ in pairs], dtype=np.float64)
    labels = np.asarray([[np.nan if value is None else value for value in y] for _, y in pairs])
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        valid = np.isfinite(labels[:, index])
        y = labels[valid, index].astype(np.int8)
        p = probabilities[valid, index]
        if len(y) == 0:
            raise ValueError(f"No valid labels for class {index + 1}")
        auc = binary_auc(y, p)
        predicted = p >= threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append({"class_id": index + 1, "class_name": name, "positive_studies": int(y.sum()),
                          "negative_studies": int((y == 0).sum()), "auc": auc, "f1": f1})
    return {"study_count": len(pairs), "macro_auc": float(np.mean([row["auc"] for row in per_class])),
            "macro_f1": float(np.mean([row["f1"] for row in per_class])),
            "threshold": threshold, "per_class": per_class}
