"""Select official Train images using a frozen, patient-disjoint split index."""

import csv
import hashlib
from pathlib import Path

from cxr.data.archive import image_member, parse_positive_classes


SPLIT_FILES = ("patient_split.csv", "study_split.csv", "image_split.csv")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_internal_train_rows(rows: list[dict[str, str]], settings: dict) -> list[dict[str, str]]:
    """Reject changed indices and retain only the fixed internal Train patients.

    The filter verifies every official Train image, Study label, and patient
    assignment before returning any rows to the training DataLoader.
    """
    if settings.get("subset") != "internal_train":
        raise ValueError("Only the frozen internal_train subset may update model weights")
    directory = Path(settings["directory"])
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[3] / directory
    hashes = settings["sha256"]
    if set(hashes) != set(SPLIT_FILES):
        raise ValueError("All three split index hashes are required")
    for name in SPLIT_FILES:
        path = directory / name
        if not path.is_file() or _sha256(path) != hashes[name]:
            raise ValueError(f"Frozen internal split hash mismatch: {name}")
    patient_rows = _read_csv(directory / "patient_split.csv")
    study_rows = _read_csv(directory / "study_split.csv")
    image_rows = _read_csv(directory / "image_split.csv")
    patient_map = {row["Subject_id"]: row["split"] for row in patient_rows}
    study_map = {(row["Subject_id"], row["Study_id"]): row for row in study_rows}
    image_map = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]): row for row in image_rows}
    if (len(patient_map), len(study_map), len(image_map)) != (len(patient_rows), len(study_rows), len(image_rows)):
        raise ValueError("Duplicate patient, Study, or image in frozen split")
    if set(patient_map.values()) != {"internal_train", "internal_val"}:
        raise ValueError("Frozen split must contain internal_train and internal_val")
    actual_keys = [(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(image_map):
        raise ValueError("Frozen image index differs from official Train CSV")
    if {(subject, study) for subject, study, _ in actual_keys} != set(study_map):
        raise ValueError("Frozen Study index differs from official Train CSV")
    if {subject for subject, _, _ in actual_keys} != set(patient_map):
        raise ValueError("Frozen patient index differs from official Train CSV")
    selected = []
    for row, key in zip(rows, actual_keys):
        subject, study, _ = key
        assignment = patient_map[subject]
        indexed_study = study_map[(subject, study)]
        indexed_image = image_map[key]
        if (indexed_study["split"] != assignment or indexed_image["split"] != assignment or
                indexed_image["archive_member"] != image_member(row, "train") or
                parse_positive_classes(indexed_study["Predict_class"]) != parse_positive_classes(row["Predict_class"])):
            raise ValueError(f"Frozen patient/Study/image record mismatch: {key}")
        if assignment == "internal_train":
            selected.append(row)
    counts = settings["expected_counts"]
    if (len(selected), len({(row["Subject_id"], row["Study_id"]) for row in selected}),
            len({row["Subject_id"] for row in selected})) != (
            counts["images"], counts["studies"], counts["patients"]):
        raise ValueError("Internal Train size differs from the approved frozen split")
    if len(rows) - len(selected) != settings["expected_holdout_images"]:
        raise ValueError("Internal Holdout image count differs")
    return selected
