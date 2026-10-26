"""CPU-only guards for Phase22 result import and fixed-fold analysis."""

import csv
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from import_phase17_friend_results import prediction_keys  # noqa: E402


FIELDS = ["Subject_id", "Study_id", *(f"Prob_{i}" for i in range(1, 11))]


def write_predictions(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for subject, study, value in records:
            writer.writerow([subject, study, *([value] * 10)])


def test_rejects_duplicate_studies(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions(path, [("subject_1", "study_1", 0.1),
                             ("subject_1", "study_1", 0.2)])
    with pytest.raises(ValueError, match="Duplicate"):
        prediction_keys(path, {("subject_1", "study_1")})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_rejects_nonfinite_or_out_of_range(tmp_path, value):
    path = tmp_path / "predictions.csv"
    write_predictions(path, [("subject_1", "study_1", value)])
    with pytest.raises(ValueError, match="Invalid probability"):
        prediction_keys(path, {("subject_1", "study_1")})


def test_rejects_missing_study(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions(path, [("subject_1", "study_1", 0.5)])
    with pytest.raises(ValueError, match="coverage mismatch"):
        prediction_keys(path, {("subject_1", "study_1"), ("subject_2", "study_2")})
