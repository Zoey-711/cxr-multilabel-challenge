"""Detailed validation diagnostics from image predictions fused by Study."""

import numpy as np

from cxr.engine.metrics import study_multilabel_metrics
from cxr.engine.validate import StudyTarget
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities


QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def analyze_validation(image_predictions: list[ImagePrediction], targets: list[StudyTarget],
                       *, threshold: float = 0.5) -> tuple[dict, list[dict]]:
    """Return class diagnostics and a traceable, one-row-per-Study probability table."""
    metrics = study_multilabel_metrics(image_predictions, targets, threshold=threshold)
    fused = aggregate_mean_probabilities(image_predictions)
    target_map = {(target.subject_id, target.study_id): target for target in targets}
    probabilities = np.asarray([study.probabilities for study in fused], dtype=np.float64)
    labels = np.asarray([target_map[(study.subject_id, study.study_id)].labels for study in fused], dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Detailed analysis requires complete validation targets")
    predicted = probabilities >= threshold
    study_rows = []
    for index, study in enumerate(fused):
        row = {"Subject_id": study.subject_id, "Study_id": study.study_id,
               "image_count": study.image_count, "predicted_class_count": int(predicted[index].sum())}
        row.update({f"probability_{class_id}": float(probabilities[index, class_id - 1]) for class_id in range(1, 11)})
        row.update({f"target_{class_id}": int(labels[index, class_id - 1]) for class_id in range(1, 11)})
        study_rows.append(row)
    for index, class_result in enumerate(metrics["per_class"]):
        y = labels[:, index] == 1
        p = predicted[:, index]
        tp, fp, fn, tn = (int((y & p).sum()), int((~y & p).sum()),
                          int((y & ~p).sum()), int((~y & ~p).sum()))
        class_result.update({
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "predicted_positive_studies": tp + fp,
            "predicted_positive_rate": (tp + fp) / len(fused),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "probability_quantiles": {f"p{int(q * 100)}": float(np.quantile(probabilities[:, index], q)) for q in QUANTILES},
        })
    metrics["zero_predicted_class_studies"] = int((predicted.sum(axis=1) == 0).sum())
    metrics["zero_predicted_class_rate"] = metrics["zero_predicted_class_studies"] / len(fused)
    return metrics, study_rows
