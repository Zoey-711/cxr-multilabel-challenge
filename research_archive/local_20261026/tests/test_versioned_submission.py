import csv
import json

import pytest

from cxr.inference.aggregate import StudyPrediction
from cxr.inference.versioned_submission import build_candidate_rows, write_bundle, verify_candidate
from scripts.retract_submission_format_status import retract


def test_versioned_candidate_preserves_ids_and_reports_zero_class_study(tmp_path):
    studies = [StudyPrediction("subject_00001", "study_00001", 2,
                               (0.7, 0.1, 0.8, *(0.1 for _ in range(7)))),
               StudyPrediction("subject_00002", "study_00002", 1, (0.2,) * 10)]
    keys = {(item.subject_id, item.study_id) for item in studies}
    config = {"models": {"B01": {"checkpoint_sha256": "a" * 64}},
              "validation_macro_auc": 0.80344, "fusion_weights": {"B01": 1.0}}
    root = tmp_path / "submissions"
    directory, checked = write_bundle(root, "E02_20261016", studies, keys, config,
                                      image_rows_by_model={}, study_rows_by_model={}, seal=False)
    assert checked["matrix_studies"] == 2
    assert checked["candidate_rows"] == 2
    assert checked["candidate_covered_studies"] == 1
    assert checked["zero_prediction_studies"] == 1
    assert checked["format_status"] == "blocked_pending_official_empty_study_rule"
    with (directory / "submission.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Subject_id", "Study_id", "Predict_class", "Probability"]
        rows = list(reader)
    assert [row["Predict_class"] for row in rows] == ["1", "3"]
    assert all(row["Study_id"] == "study_00001" for row in rows)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["csv_sha256"] == checked["csv_sha256"]
    with (root / "submission_history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 1 and history[0]["upload_status"] == "not_uploaded"
    with pytest.raises(FileExistsError):
        write_bundle(root, "E02_20261016", studies, keys, config, {}, {}, seal=False)
    confirmed_dir, confirmed = write_bundle(root, "E02_20261016_v2", studies, keys, config,
                                            {}, {}, seal=False, empty_study_policy_confirmed=True)
    assert confirmed["format_status"] == "ready_for_manual_upload"
    assert confirmed["csv_sha256"] == checked["csv_sha256"]
    assert json.loads((confirmed_dir / "manifest.json").read_text(encoding="utf-8"))[
        "empty_study_omission_confirmed"] is True
    sealed_report = (confirmed_dir / "prediction_check.json").read_bytes()
    retract(root, "E02_20261016_v2", "Official empty-Study permission was not established")
    assert (confirmed_dir / "prediction_check.json").read_bytes() == sealed_report
    with (root / "submission_history.csv").open(encoding="utf-8", newline="") as handle:
        updated = {row["version"]: row for row in csv.DictReader(handle)}
    assert updated["E02_20261016_v2"]["format_status"] == "blocked_pending_official_empty_study_rule"
    with (root / "submission_status_events.csv").open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    assert len(events) == 1 and events[0]["version"] == "E02_20261016_v2"


def test_candidate_rejects_duplicate_study_and_does_not_force_a_label():
    empty = StudyPrediction("subject_00007", "study_00008", 1, (0.49,) * 10)
    matrix, candidate, zero = build_candidate_rows([empty], threshold=0.5)
    assert len(matrix) == 1 and candidate == []
    assert zero == [{"Subject_id": "subject_00007", "Study_id": "study_00008"}]
    with pytest.raises(ValueError, match="Duplicate"):
        build_candidate_rows([empty, empty])


def test_all_classes_policy_includes_zero_positive_study_and_records_generated(tmp_path):
    studies = [StudyPrediction("subject_00001", "study_00001", 2, (0.2,) * 10),
               StudyPrediction("subject_00002", "study_00002", 1, (0.7,) + (0.1,) * 9)]
    keys = {(item.subject_id, item.study_id) for item in studies}
    config = {"models": {"B01": {"checkpoint_sha256": "a" * 64}},
              "validation_macro_auc": 0.8, "fusion_weights": {"B01": 1.0}}
    root = tmp_path / "submissions"
    directory, checked = write_bundle(root, "E01_20261016", studies, keys, config, {}, {},
                                      policy="all_classes", upload_status="generated", seal=False)
    assert checked["candidate_rows"] == 20
    assert checked["candidate_covered_studies"] == 2
    assert checked["zero_prediction_studies"] == 0
    assert checked["zero_prediction_studies_at_0_5"] == 1
    assert checked["format_status"] == "ready_for_manual_upload"
    with (directory / "submission.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Subject_id", "Study_id", "Predict_class", "Probability"]
        rows = list(reader)
    for key in keys:
        assert {int(row["Predict_class"]) for row in rows
                if (row["Subject_id"], row["Study_id"]) == key} == set(range(1, 11))
    assert verify_candidate(directory / "probability_matrix.csv", directory / "submission.csv",
                            keys, policy="all_classes")["candidate_rows"] == 20
    with (root / "submission_history.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["upload_status"] == "generated"
