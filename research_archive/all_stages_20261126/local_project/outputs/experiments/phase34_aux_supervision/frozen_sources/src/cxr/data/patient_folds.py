"""Frozen patient-level folds over the official training archive."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from cxr.data.archive import image_member, parse_positive_classes


INDEX_FILES = ("patient_folds.csv", "study_folds.csv", "image_folds.csv")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_patient_folds(rows: list[dict[str, str]], *, seed: int = 42, n_splits: int = 5) -> dict[str, int]:
    """Assign whole patients with a fixed rare-label-first greedy balance rule."""
    if n_splits < 2:
        raise ValueError("At least two folds are required")
    studies = {}
    images_per_study = defaultdict(int)
    for row in rows:
        key = (row["Subject_id"], row["Study_id"])
        label = parse_positive_classes(row["Predict_class"])
        if key in studies and studies[key] != label:
            raise ValueError(f"Inconsistent Study labels: {key}")
        studies[key] = label
        images_per_study[key] += 1
    subjects = sorted({subject for subject, _ in studies})
    patient_labels = {subject: np.zeros(10, dtype=np.int64) for subject in subjects}
    patient_studies = defaultdict(int)
    patient_images = defaultdict(int)
    for (subject, study), positives in studies.items():
        patient_studies[subject] += 1
        patient_images[subject] += images_per_study[(subject, study)]
        for class_id in positives:
            patient_labels[subject][class_id - 1] = 1
    total_labels = np.sum(list(patient_labels.values()), axis=0)
    if np.any(total_labels < n_splits):
        raise ValueError("A class has fewer positive patients than folds")
    rng = np.random.default_rng(seed)
    tie = {subject: float(value) for subject, value in zip(subjects, rng.random(len(subjects)))}
    rarity = 1.0 / total_labels
    order = sorted(subjects, key=lambda subject: (
        -float(np.dot(patient_labels[subject], rarity)),
        -int(patient_labels[subject].sum()), -patient_studies[subject], tie[subject], subject))
    fold_labels = np.zeros((n_splits, 10), dtype=np.int64)
    fold_patients = np.zeros(n_splits, dtype=np.int64)
    fold_studies = np.zeros(n_splits, dtype=np.int64)
    fold_images = np.zeros(n_splits, dtype=np.int64)
    target_patients = len(subjects) / n_splits
    target_studies = len(studies) / n_splits
    target_images = len(rows) / n_splits
    assignment = {}
    for subject in order:
        label = patient_labels[subject]
        active = np.flatnonzero(label)
        candidates = [fold for fold in range(n_splits) if fold_patients[fold] < int(np.ceil(target_patients))]
        if not candidates:
            raise RuntimeError("No patient fold has capacity")
        def score(fold: int) -> tuple[float, int, int]:
            class_score = float(np.sum((fold_labels[fold, active] + 1) / total_labels[active])) if len(active) else 0.0
            balance = (float(fold_patients[fold] / target_patients) * 0.10 +
                       float(fold_studies[fold] / target_studies) * 0.04 +
                       float(fold_images[fold] / target_images) * 0.02)
            return (class_score + balance, int(fold_patients[fold]), fold)
        fold = min(candidates, key=score)
        assignment[subject] = fold + 1
        fold_labels[fold] += label
        fold_patients[fold] += 1
        fold_studies[fold] += patient_studies[subject]
        fold_images[fold] += patient_images[subject]
    if set(assignment) != set(subjects) or np.any(fold_labels == 0):
        raise ValueError("Incomplete or class-empty patient fold assignment")
    return assignment


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_fold_rows(rows: list[dict[str, str]], settings: dict, *, role: str) -> list[dict[str, str]]:
    """Verify all frozen indices against official Train, then select train or OOF."""
    if role not in {"train", "oof"}:
        raise ValueError("role must be train or oof")
    directory = Path(settings["directory"])
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[3] / directory
    expected = settings["sha256"]
    if set(expected) != set(INDEX_FILES):
        raise ValueError("All three frozen fold index hashes are required")
    for name in INDEX_FILES:
        if not (directory / name).is_file() or sha256(directory / name) != expected[name]:
            raise ValueError(f"Frozen fold index hash mismatch: {name}")
    fold = int(settings["fold"])
    if fold not in range(1, 6):
        raise ValueError("Fold must be 1 through 5")
    patient_rows = read_index(directory / "patient_folds.csv")
    study_rows = read_index(directory / "study_folds.csv")
    image_rows = read_index(directory / "image_folds.csv")
    patient_fold = {row["Subject_id"]: int(row["fold"]) for row in patient_rows}
    study_map = {(row["Subject_id"], row["Study_id"]): row for row in study_rows}
    image_map = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]): row for row in image_rows}
    if len(patient_fold) != len(patient_rows) or len(study_map) != len(study_rows) or len(image_map) != len(image_rows):
        raise ValueError("Duplicate patient, Study, or image in fold indices")
    actual = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in rows}
    if len(actual) != len(rows) or actual != set(image_map):
        raise ValueError("Fold image index differs from official Train")
    if {(subject, study) for subject, study, _ in actual} != set(study_map):
        raise ValueError("Fold Study index differs from official Train")
    if {subject for subject, _, _ in actual} != set(patient_fold):
        raise ValueError("Fold patient index differs from official Train")
    selected = []
    for row in rows:
        subject, study, image = (row[key] for key in ("Subject_id", "Study_id", "Dicom_id"))
        indexed_study = study_map[(subject, study)]
        indexed_image = image_map[(subject, study, image)]
        assigned = patient_fold[subject]
        if (assigned != int(indexed_study["fold"]) or assigned != int(indexed_image["fold"]) or
                indexed_image["archive_member"] != image_member(row, "train") or
                parse_positive_classes(indexed_study["Predict_class"]) != parse_positive_classes(row["Predict_class"])):
            raise ValueError(f"Fold patient/Study/image mismatch: {(subject, study, image)}")
        if (assigned != fold) == (role == "train"):
            selected.append(row)
    expected_counts = settings["expected_counts"][role]
    counts = {"patients": len({row["Subject_id"] for row in selected}),
              "studies": len({(row["Subject_id"], row["Study_id"]) for row in selected}),
              "images": len(selected)}
    if counts != expected_counts:
        raise ValueError(f"Fold {role} counts changed: {counts} != {expected_counts}")
    return selected
