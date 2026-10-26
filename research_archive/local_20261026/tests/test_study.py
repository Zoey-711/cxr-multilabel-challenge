import pytest

from cxr.engine.validate import StudyTarget, prepare_study_validation
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.inference.submission import submission_row, write_submission


def test_fusion_is_study_level_and_class_ids_start_at_one():
    a = (0.2,) * 10
    b = (0.8,) * 10
    predictions = [
        ImagePrediction("p1", "s1", "a", a),
        ImagePrediction("p1", "s1", "b", b),
        ImagePrediction("p1", "s2", "c", a),
    ]
    studies = aggregate_mean_probabilities(predictions)
    assert len(studies) == 2
    assert studies[0].image_count == 2
    assert studies[0].probabilities == pytest.approx((0.5,) * 10)
    pairs = prepare_study_validation(predictions, [
        StudyTarget("p1", "s1", (1.0,) * 10),
        StudyTarget("p1", "s2", (None,) * 10),
    ])
    assert len(pairs) == 2
    assert submission_row(studies[0], 0)["Predict_class"] == 1
    assert submission_row(studies[0], 9)["Predict_class"] == 10


def test_submission_writer_requires_explicit_policy(tmp_path):
    study = aggregate_mean_probabilities([ImagePrediction("p1", "s1", "a", (0.6,) * 10)])
    with pytest.raises(TypeError):
        write_submission(study, tmp_path / "submission.csv")
    path = write_submission(study, tmp_path / "submission.csv", policy="all_classes")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "Subject_id,Study_id,Predict_class,Probability"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 11


def test_threshold_emits_multiple_rows_and_rejects_empty_study(tmp_path):
    probabilities = (0.9, 0.2, 0.5) + (0.1,) * 7
    studies = aggregate_mean_probabilities([ImagePrediction("p1", "s1", "a", probabilities)])
    path = write_submission(studies, tmp_path / "submission.csv", policy="threshold", threshold=0.5)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("p1,s1,1,") and lines[2].startswith("p1,s1,3,")
    with pytest.raises(ValueError, match="no rows"):
        write_submission(studies, tmp_path / "empty.csv", policy="threshold", threshold=1.0)
