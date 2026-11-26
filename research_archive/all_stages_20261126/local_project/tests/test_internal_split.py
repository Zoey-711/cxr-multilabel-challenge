import csv
import hashlib

import pytest

from cxr.data.internal_split import select_internal_train_rows


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_split_restricts_gradient_rows_and_rejects_changed_index(tmp_path):
    rows = [
        {"Subject_id": "subject_1", "Study_id": "study_1", "Dicom_id": "pic_1", "Predict_class": "1;8"},
        {"Subject_id": "subject_2", "Study_id": "study_2", "Dicom_id": "pic_2", "Predict_class": "2"},
    ]
    hashes = {
        "patient_split.csv": _csv(tmp_path / "patient_split.csv", ["Subject_id", "split"],
                                  [{"Subject_id": "subject_1", "split": "internal_train"},
                                   {"Subject_id": "subject_2", "split": "internal_val"}]),
        "study_split.csv": _csv(tmp_path / "study_split.csv", ["Subject_id", "Study_id", "split", "Predict_class"],
                                [{"Subject_id": "subject_1", "Study_id": "study_1", "split": "internal_train", "Predict_class": "1;8"},
                                 {"Subject_id": "subject_2", "Study_id": "study_2", "split": "internal_val", "Predict_class": "2"}]),
        "image_split.csv": _csv(tmp_path / "image_split.csv",
                                ["Subject_id", "Study_id", "Dicom_id", "archive_member", "split"],
                                [{"Subject_id": "subject_1", "Study_id": "study_1", "Dicom_id": "pic_1",
                                  "archive_member": "cxr_first/train/subject_1/study_1/pic_1.jpg", "split": "internal_train"},
                                 {"Subject_id": "subject_2", "Study_id": "study_2", "Dicom_id": "pic_2",
                                  "archive_member": "cxr_first/train/subject_2/study_2/pic_2.jpg", "split": "internal_val"}]),
    }
    settings = {"directory": str(tmp_path), "subset": "internal_train", "sha256": hashes,
                "expected_counts": {"images": 1, "studies": 1, "patients": 1}, "expected_holdout_images": 1}
    assert select_internal_train_rows(rows, settings) == rows[:1]
    with pytest.raises(ValueError, match="differs from official Train CSV"):
        select_internal_train_rows(rows[:1], settings)
    corrupted = {**settings, "sha256": {**hashes, "image_split.csv": "0" * 64}}
    with pytest.raises(ValueError, match="hash mismatch"):
        select_internal_train_rows(rows, corrupted)
    changed_label = [dict(rows[0], Predict_class="1"), rows[1]]
    with pytest.raises(ValueError, match="record mismatch"):
        select_internal_train_rows(changed_label, settings)
