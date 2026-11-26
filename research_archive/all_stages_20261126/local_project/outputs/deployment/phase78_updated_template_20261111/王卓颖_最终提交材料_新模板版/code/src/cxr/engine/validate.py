"""Study-level validation interface."""

from dataclasses import dataclass
from typing import Sequence

from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities


@dataclass(frozen=True)
class StudyTarget:
    subject_id: str
    study_id: str
    labels: tuple[float | None, ...]


def prepare_study_validation(
    image_predictions: Sequence[ImagePrediction],
    targets: Sequence[StudyTarget],
) -> list[tuple[tuple[float, ...], tuple[float | None, ...]]]:
    """Align fused study probabilities to study targets, preserving missing labels."""
    fused = {(item.subject_id, item.study_id): item.probabilities for item in aggregate_mean_probabilities(image_predictions)}
    pairs = []
    seen = set()
    for target in targets:
        key = (target.subject_id, target.study_id)
        if key in seen:
            raise ValueError(f"Duplicate study target: {key}")
        seen.add(key)
        if len(target.labels) != 10:
            raise ValueError("Each study target must contain 10 labels")
        if key not in fused:
            raise ValueError(f"Missing image predictions for study: {key}")
        pairs.append((fused[key], target.labels))
    if set(fused) != seen:
        raise ValueError("Predictions contain studies with no targets")
    return pairs
