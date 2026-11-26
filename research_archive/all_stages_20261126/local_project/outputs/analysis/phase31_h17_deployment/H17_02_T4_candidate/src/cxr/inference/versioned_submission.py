"""Versioned Study-level candidate bundles with explicit empty-study reporting."""

import csv
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

from cxr.inference.aggregate import StudyPrediction


CSV_FIELDS = ("Subject_id", "Study_id", "Predict_class", "Probability")
MATRIX_FIELDS = ("Subject_id", "Study_id", *(f"probability_{i}" for i in range(1, 11)))
HISTORY_FIELDS = ("version", "generated_at", "candidate_csv", "csv_sha256", "validation_macro_auc",
                  "row_count", "study_count", "zero_prediction_studies", "format_status",
                  "upload_status", "uploaded_at", "official_score")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_candidate_rows(studies: list[StudyPrediction], threshold: float = 0.5,
                         policy: str = "threshold"):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    if policy not in {"threshold", "all_classes"}:
        raise ValueError(f"Unknown submission policy: {policy}")
    seen, candidate, matrix, empty = set(), [], [], []
    for study in studies:
        key = (study.subject_id, study.study_id)
        if key in seen or not all(key):
            raise ValueError(f"Duplicate or blank Study key: {key}")
        seen.add(key)
        if len(study.probabilities) != 10 or study.image_count < 1 or \
                any(not 0 <= value <= 1 for value in study.probabilities):
            raise ValueError(f"Invalid Study probabilities or image count: {key}")
        matrix.append({"Subject_id": key[0], "Study_id": key[1],
                       **{f"probability_{i}": repr(float(value))
                          for i, value in enumerate(study.probabilities, 1)}})
        chosen = [i for i, probability in enumerate(study.probabilities, 1)
                  if probability >= threshold]
        if not chosen:
            empty.append({"Subject_id": key[0], "Study_id": key[1]})
        if policy == "all_classes":
            chosen = range(1, 11)
        for class_id in chosen:
            candidate.append({"Subject_id": key[0], "Study_id": key[1],
                              "Predict_class": class_id,
                              "Probability": repr(float(study.probabilities[class_id - 1]))})
    return matrix, candidate, empty


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_candidate(matrix_path: Path, csv_path: Path, expected_keys: set[tuple[str, str]],
                     threshold: float = 0.5, policy: str = "threshold") -> dict:
    if policy not in {"threshold", "all_classes"}:
        raise ValueError(f"Unknown submission policy: {policy}")
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
            raise ValueError("Internal probability matrix header differs")
        matrix_rows = list(reader)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("Submission header/order differs from official example")
        candidate_rows = list(reader)
    if len(matrix_rows) != len(expected_keys):
        raise ValueError("Internal matrix does not cover all expected Studies")
    lookup = {}
    for row in matrix_rows:
        key = (row["Subject_id"], row["Study_id"])
        if key in lookup:
            raise ValueError("Duplicate Study in probability matrix")
        values = [float(row[f"probability_{i}"]) for i in range(1, 11)]
        if any(not 0 <= p <= 1 for p in values):
            raise ValueError("Invalid probability in matrix")
        lookup[key] = values
    if set(lookup) != expected_keys:
        raise ValueError("Probability matrix Study coverage differs from test metadata")
    actual = {}
    for row in candidate_rows:
        key = (row["Subject_id"], row["Study_id"])
        class_id = int(row["Predict_class"])
        probability = float(row["Probability"])
        if key not in lookup or not 1 <= class_id <= 10 or \
                not 0 <= probability <= 1 or (policy == "threshold" and probability < threshold) or \
                probability != lookup[key][class_id - 1]:
            raise ValueError("Candidate CSV row does not match probability matrix")
        class_key = (*key, class_id)
        if class_key in actual:
            raise ValueError("Duplicate Subject/Study/class record")
        actual[class_key] = probability
    expected = {(*key, i): probabilities[i - 1] for key, probabilities in lookup.items()
                for i in range(1, 11) if policy == "all_classes" or probabilities[i - 1] >= threshold}
    if actual != expected:
        raise ValueError("Candidate rows do not match the configured threshold")
    covered = {(row["Subject_id"], row["Study_id"]) for row in candidate_rows}
    zero_at_threshold = sum(all(value < threshold for value in values) for values in lookup.values())
    return {"matrix_studies": len(lookup), "candidate_rows": len(candidate_rows),
            "candidate_covered_studies": len(covered),
            "zero_prediction_studies": len(expected_keys - covered),
            "zero_prediction_studies_at_0_5": zero_at_threshold,
            "csv_bytes": csv_path.stat().st_size, "csv_sha256": sha256(csv_path),
            "encoding": "UTF-8 without BOM", "header": list(CSV_FIELDS),
            "duplicate_class_records": 0, "probability_range_valid": True,
            "matrix_coverage_valid": True}


def append_history(root: Path, record: dict) -> None:
    history = root / "submission_history.csv"
    if history.exists():
        with history.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
                raise ValueError("Submission history header differs")
            if any(row["version"] == record["version"] for row in reader):
                raise FileExistsError(f"Submission history already contains {record['version']}")
    with history.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if history.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)


def seal_bundle(directory: Path) -> None:
    """Mark bundle files read-only; directory-exists guard prevents regeneration."""
    for path in directory.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    os.chmod(directory, stat.S_IREAD | stat.S_IEXEC)


def write_bundle(root: Path, version: str, studies: list[StudyPrediction],
                 expected_keys: set[tuple[str, str]], fusion_config: dict,
                 image_rows_by_model: dict[str, list[dict]],
                 study_rows_by_model: dict[str, list[dict]],
                 *, seal: bool = True, empty_study_policy_confirmed: bool = False,
                 policy: str = "threshold", upload_status: str = "not_uploaded") -> tuple[Path, dict]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version) or version in {".", ".."}:
        raise ValueError("Version must be a safe directory name")
    directory = root / version
    if directory.exists():
        raise FileExistsError(f"Submission version already exists: {directory}")
    root.mkdir(parents=True, exist_ok=True)
    matrix, candidate, empty = build_candidate_rows(studies, threshold=0.5, policy=policy)
    if {(row["Subject_id"], row["Study_id"]) for row in matrix} != expected_keys:
        raise ValueError("Fused Study keys differ from test metadata")
    directory.mkdir(exist_ok=False)
    try:
        write_csv(directory / "probability_matrix.csv", MATRIX_FIELDS, matrix)
        write_csv(directory / "submission.csv", CSV_FIELDS, candidate)
        write_csv(directory / "zero_prediction_studies.csv", ("Subject_id", "Study_id"), empty)
        for name, rows in image_rows_by_model.items():
            write_csv(directory / f"{name.lower()}_image_probabilities.csv",
                      ("Subject_id", "Study_id", "Image_id", *(f"probability_{i}" for i in range(1, 11))), rows)
        for name, rows in study_rows_by_model.items():
            write_csv(directory / f"{name.lower()}_study_probabilities.csv", MATRIX_FIELDS, rows)
        (directory / "fusion_config.json").write_text(json.dumps(fusion_config, ensure_ascii=False, indent=2), encoding="utf-8")
        validation = verify_candidate(directory / "probability_matrix.csv", directory / "submission.csv",
                                      expected_keys, threshold=0.5, policy=policy)
        validation["format_status"] = ("ready_for_manual_upload" if policy == "all_classes" or
                                       not empty or empty_study_policy_confirmed else
                                       "blocked_pending_official_empty_study_rule")
        validation["empty_study_policy"] = ("all_ten_classes_submitted" if policy == "all_classes" else
                                            "user_confirmed_omission_allowed" if empty_study_policy_confirmed else
                                            "not confirmed by official instructions")
        validation["submission_policy"] = policy
        (directory / "prediction_check.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        manifest = {"version": version, "generated_at": timestamp,
                    "model_versions": list(fusion_config["models"]),
                    "checkpoint_sha256": {name: item["checkpoint_sha256"] for name, item in fusion_config["models"].items()},
                    "validation_macro_auc": fusion_config["validation_macro_auc"],
                    "fusion_weights": fusion_config["fusion_weights"],
                    "submission_policy": policy,
                    "empty_study_omission_confirmed": empty_study_policy_confirmed,
                    "csv_sha256": validation["csv_sha256"],
                    "files": {path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in sorted(directory.iterdir()) if path.is_file()},
                    "upload_status": upload_status, "official_score": None}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        history_record = {"version": version, "generated_at": timestamp,
                          "candidate_csv": str(directory / "submission.csv"),
                          "csv_sha256": validation["csv_sha256"],
                          "validation_macro_auc": fusion_config["validation_macro_auc"],
                          "row_count": validation["candidate_rows"],
                          "study_count": validation["matrix_studies"],
                          "zero_prediction_studies": validation["zero_prediction_studies"],
                          "format_status": validation["format_status"],
                          "upload_status": upload_status, "uploaded_at": "", "official_score": ""}
        append_history(root, history_record)
        if seal:
            seal_bundle(directory)
        return directory, validation
    except Exception:
        # Leave the partial version in place so a retry must use a new version.
        raise
