"""Study-level CSV rows selected by the configured probability threshold."""

import csv
from pathlib import Path

from cxr.inference.aggregate import StudyPrediction
from cxr.utils.labels import original_class_id


def submission_row(study: StudyPrediction, model_index: int) -> dict[str, str | int | float]:
    """Map one model output to its original competition class ID."""
    class_id = original_class_id(model_index)
    return {
        "Subject_id": study.subject_id,
        "Study_id": study.study_id,
        "Predict_class": class_id,
        "Probability": study.probabilities[model_index],
    }


def make_submission_rows(
    studies: list[StudyPrediction], *, policy: str, threshold: float = 0.5
) -> list[dict[str, str | int | float]]:
    if policy not in {"all_classes", "threshold"}:
        raise ValueError("Choose an explicit submission policy: all_classes or threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    rows = []
    for study in studies:
        selected = [
            index for index, probability in enumerate(study.probabilities)
            if policy == "all_classes" or probability >= threshold
        ]
        if not selected:
            raise ValueError(f"Threshold produces no rows for study {study.study_id}; confirm official empty-study policy")
        rows.extend(submission_row(study, index) for index in selected)
    return rows


def write_submission(
    studies: list[StudyPrediction], output: str | Path, *, policy: str, threshold: float = 0.5
) -> Path:
    """Write a UTF-8 CSV after selecting a row policy and threshold."""
    rows = make_submission_rows(studies, policy=policy, threshold=threshold)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Subject_id", "Study_id", "Predict_class", "Probability"))
        writer.writeheader()
        writer.writerows(rows)
    return path
