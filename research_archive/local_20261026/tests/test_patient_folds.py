"""Patient folds and immutable subset selection on synthetic records."""

import csv

import pytest

from cxr.data.archive import image_member
from cxr.data.patient_folds import INDEX_FILES, build_patient_folds, select_fold_rows, sha256


def sample_rows():
    rows = []
    for patient in range(100):
        subject = f"subject_{patient:05d}"
        for study_index in range(2):
            study = f"study_{patient * 2 + study_index:06d}"
            label = str((patient + study_index) % 10 + 1)
            for image_index in range(2):
                rows.append({"Subject_id": subject, "Study_id": study,
                             "Dicom_id": f"pic_{patient * 4 + study_index * 2 + image_index:06d}",
                             "Predict_class": label})
    return rows


def write_index(path, fields, records):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def test_folds_are_reproducible_and_keep_whole_patients():
    rows = sample_rows()
    first = build_patient_folds(rows, seed=42)
    assert first == build_patient_folds(list(reversed(rows)), seed=42)
    assert set(first.values()) == {1, 2, 3, 4, 5}
    assert sorted(sum(value == fold for value in first.values()) for fold in range(1, 6)) == [20] * 5


def test_frozen_fold_selection_rejects_tampering(tmp_path):
    rows = sample_rows()
    folds = build_patient_folds(rows, seed=42)
    patient_records = [{"Subject_id": subject, "fold": fold} for subject, fold in sorted(folds.items())]
    studies = {(row["Subject_id"], row["Study_id"]): row["Predict_class"] for row in rows}
    study_records = [{"Subject_id": subject, "Study_id": study,
                      "fold": folds[subject], "Predict_class": label}
                     for (subject, study), label in sorted(studies.items())]
    image_records = [{"Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                      "Dicom_id": row["Dicom_id"], "archive_member": image_member(row, "train"),
                      "fold": folds[row["Subject_id"]]} for row in rows]
    write_index(tmp_path / INDEX_FILES[0], ["Subject_id", "fold"], patient_records)
    write_index(tmp_path / INDEX_FILES[1], ["Subject_id", "Study_id", "fold", "Predict_class"], study_records)
    write_index(tmp_path / INDEX_FILES[2], ["Subject_id", "Study_id", "Dicom_id", "archive_member", "fold"], image_records)
    def counts(selected):
        return {"patients": len({row["Subject_id"] for row in selected}),
                "studies": len({(row["Subject_id"], row["Study_id"]) for row in selected}),
                "images": len(selected)}
    training = [row for row in rows if folds[row["Subject_id"]] != 1]
    oof = [row for row in rows if folds[row["Subject_id"]] == 1]
    settings = {"directory": str(tmp_path), "fold": 1,
                "sha256": {name: sha256(tmp_path / name) for name in INDEX_FILES},
                "expected_counts": {"train": counts(training), "oof": counts(oof)}}
    assert select_fold_rows(rows, settings, role="train") == training
    assert select_fold_rows(rows, settings, role="oof") == oof
    changed = [dict(row) for row in rows]
    changed[0]["Predict_class"] = "10" if changed[0]["Predict_class"] != "10" else "1"
    with pytest.raises(ValueError, match="mismatch"):
        select_fold_rows(changed, settings, role="train")
    with (tmp_path / "patient_folds.csv").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        select_fold_rows(rows, settings, role="oof")
