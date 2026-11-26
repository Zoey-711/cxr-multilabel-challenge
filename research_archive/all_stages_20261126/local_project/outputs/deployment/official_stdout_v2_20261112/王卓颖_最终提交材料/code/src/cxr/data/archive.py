"""Read the inspected cxr_first ZIP layout without extracting or changing it."""

import csv
import io
import os
import re
import zipfile
from pathlib import Path


SPLITS = {"train": "train", "val": "valid", "test": "test"}
COMMON_COLUMNS = ("Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation", "Rows", "Columns")
ID_PATTERNS = {
    "Dicom_id": re.compile(r"pic_[0-9]+\Z"),
    "Subject_id": re.compile(r"subject_[0-9]+\Z"),
    "Study_id": re.compile(r"study_[0-9]+\Z"),
}


def archive_path(config: dict) -> Path:
    value = os.environ.get("CXR_DATA_ROOT") or config["data"].get("archive_path")
    if not value:
        raise ValueError("Set CXR_DATA_ROOT or data.archive_path to the official CXR archive")
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        single = path / "cxr_first.zip"
        if single.is_file():
            return single
        if not (path / "train.zip").is_file() or not (path / "valid.zip").is_file():
            raise FileNotFoundError("CXR_DATA_ROOT directory requires train.zip and valid.zip")
    elif not path.is_file():
        raise FileNotFoundError(path)
    return path


def split_archive_path(path: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    base = Path(path)
    target = base / f"{SPLITS[split]}.zip" if base.is_dir() else base
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def read_split_rows(path: str | Path, split: str) -> list[dict[str, str]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    with zipfile.ZipFile(split_archive_path(path, split)) as archive:
        with archive.open(f"cxr_first/{SPLITS[split]}.csv") as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
            fields = tuple(reader.fieldnames or ())
            expected = COMMON_COLUMNS + (() if split == "test" else ("Predict_class",))
            if fields != expected:
                raise ValueError(f"Unexpected {split} fields: {fields}; expected {expected}")
            return list(reader)


def image_member(row: dict[str, str], split: str) -> str:
    """Map inspected CSV identifiers to a ZIP member, rejecting path traversal."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    image, subject, study = (row[key] for key in ("Dicom_id", "Subject_id", "Study_id"))
    if not all(ID_PATTERNS[key].fullmatch(row[key]) for key in ID_PATTERNS):
        raise ValueError(f"Invalid image/subject/study ID: {(image, subject, study)}")
    return f"cxr_first/{SPLITS[split]}/{subject}/{study}/{image}.jpg"


def parse_positive_classes(raw: str) -> frozenset[int]:
    """Parse listed positives only; unlisted classes retain unknown status."""
    if not raw:
        return frozenset()
    parts = raw.split(";")
    if any(not part.isdecimal() or str(int(part)) != part for part in parts):
        raise ValueError(f"Invalid Predict_class value: {raw!r}")
    ids = [int(part) for part in parts]
    if any(not 1 <= value <= 10 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid or repeated Predict_class ID: {raw!r}")
    return frozenset(ids)
