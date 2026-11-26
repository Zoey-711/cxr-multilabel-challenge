"""Deterministic image-to-study probability aggregation."""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class ImagePrediction:
    subject_id: str
    study_id: str
    image_id: str
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class StudyPrediction:
    subject_id: str
    study_id: str
    image_count: int
    probabilities: tuple[float, ...]


def aggregate_mean_probabilities(predictions: Iterable[ImagePrediction]) -> list[StudyPrediction]:
    """Group by subject and study; average probabilities across all images."""
    sums: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], int] = {}
    seen_images: set[tuple[str, str, str]] = set()
    for item in predictions:
        if not item.subject_id or not item.study_id or not item.image_id:
            raise ValueError("Prediction IDs cannot be blank")
        if len(item.probabilities) != 10 or any(not isfinite(p) or not 0 <= p <= 1 for p in item.probabilities):
            raise ValueError("Each image needs 10 finite probabilities in [0, 1]")
        image_key = (item.subject_id, item.study_id, item.image_id)
        if image_key in seen_images:
            raise ValueError(f"Duplicate image prediction: {image_key}")
        seen_images.add(image_key)
        key = (item.subject_id, item.study_id)
        sums.setdefault(key, [0.0] * 10)
        counts[key] = counts.get(key, 0) + 1
        for index, probability in enumerate(item.probabilities):
            sums[key][index] += probability
    return [
        StudyPrediction(subject, study, counts[(subject, study)], tuple(value / counts[(subject, study)] for value in sums[(subject, study)]))
        for subject, study in sorted(sums)
    ]
