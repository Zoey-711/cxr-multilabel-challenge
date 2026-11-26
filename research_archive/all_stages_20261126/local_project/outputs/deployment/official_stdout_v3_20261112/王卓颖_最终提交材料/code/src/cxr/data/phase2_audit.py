"""Full audit of the inspected cxr_first archive and its image-level CSVs."""

import csv
import io
import json
import platform
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from PIL import Image

from cxr.data.archive import image_member, parse_positive_classes, read_split_rows
from cxr.utils.labels import CLASS_NAMES


def environment_snapshot() -> dict:
    import torch
    import torchvision

    executable = shutil.which("nvidia-smi")
    driver = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ) if executable else None
    available = torch.cuda.is_available()
    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if available else None,
        "nvidia_driver": driver.stdout.strip().splitlines()[0] if driver and driver.returncode == 0 and driver.stdout.strip() else None,
        "project_path": str(Path.cwd().resolve()),
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _check_image(task: tuple[zipfile.ZipFile, str, int, int]) -> tuple[str, str, tuple[int, int] | None, str | None]:
    archive, member, expected_rows, expected_cols = task
    try:
        with Image.open(io.BytesIO(archive.read(member))) as image:
            size = image.size
            image.load()  # Force full JPEG decode; archive.read also validates ZIP CRC.
        issue = None
        if min(size) < 128 or max(size) > 10000:
            issue = "abnormal_dimensions"
        elif size != (expected_cols, expected_rows):
            issue = "metadata_dimension_mismatch"
        return member, "ok", size, issue
    except Exception as exc:
        return member, "unreadable", None, f"{type(exc).__name__}: {exc}"


def audit_archive(path: str | Path, output_dir: str | Path, *, workers: int = 4) -> dict:
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: read_split_rows(path, split) for split in ("train", "val", "test")}
    report: dict = {
        "environment": environment_snapshot(),
        "archive_path": str(path.resolve()),
        "archive_size_bytes": path.stat().st_size,
        "official_rules": {"complete_multilabel_annotations": True, "max_daily_submissions": 2,
                           "support_devices_field_described_but_absent_from_csv": True},
        "csv_files": {"train": "cxr_first/train.csv", "val": "cxr_first/valid.csv", "test": "cxr_first/test.csv"},
        "csv_headers": {split: list(rows[0]) for split, rows in rows_by_split.items()},
        "label_encoding": {
            "format": "semicolon-delimited listed positive class IDs 1-10 in Predict_class",
            "official_basis": "train/validation sets provide complete multilabel annotations; classes are marked only when positively mentioned in the report and present in the image",
            "unlisted_class_status": "negative competition target under complete-annotation rule; clinical absence is not established",
            "explicit_negative_code_observed": False,
            "uncertain_code_observed": False,
            "blank_labeled_rows_observed": False,
            "test_has_label_column": False,
        },
        "column_types": {
            "Dicom_id": "string ID", "Subject_id": "string ID", "Study_id": "string ID",
            "ViewPosition": "categorical string; blank possible", "PatientOrientation": "categorical string; blank possible",
            "Rows": "integer string", "Columns": "integer string", "Predict_class": "semicolon-delimited string (train/val only)",
        },
        "splits": {},
    }
    class_rows: list[dict] = []
    study_rows: list[dict] = []
    co_rows: list[dict] = []
    expected_members: dict[str, tuple[str, int, int]] = {}
    subjects_by_split: dict[str, set[str]] = {}
    studies_by_split: dict[str, set[str]] = {}

    for split, rows in rows_by_split.items():
        study_images: Counter[tuple[str, str]] = Counter()
        study_labels: dict[tuple[str, str], frozenset[int]] = {}
        subjects: dict[str, set[str]] = defaultdict(set)
        views: Counter[str] = Counter()
        orientations: Counter[str] = Counter()
        image_positive: Counter[int] = Counter()
        image_unlisted: Counter[int] = Counter()
        blank_labels = 0
        bad_labels = []
        no_finding_conflicts = []
        inconsistent_labels = []
        repeated_associations = []
        duplicate_image_ids = []
        conflicting_study_subject = []
        bad_dimensions_metadata = []
        seen_associations: set[tuple[str, str, str]] = set()
        seen_images: set[str] = set()
        owner_by_study: dict[str, str] = {}
        member_rows = []
        for row in rows:
            subject, study, image = row["Subject_id"], row["Study_id"], row["Dicom_id"]
            association = (subject, study, image)
            if association in seen_associations:
                repeated_associations.append(association)
            seen_associations.add(association)
            if image in seen_images:
                duplicate_image_ids.append(image)
            seen_images.add(image)
            if study in owner_by_study and owner_by_study[study] != subject:
                conflicting_study_subject.append(study)
            owner_by_study[study] = subject
            subjects[subject].add(study)
            key = (subject, study)
            study_images[key] += 1
            views[row["ViewPosition"] or "<blank>"] += 1
            orientations[row["PatientOrientation"] or "<blank>"] += 1
            try:
                height, width = int(row["Rows"]), int(row["Columns"])
                if height <= 0 or width <= 0:
                    raise ValueError("nonpositive dimensions")
            except ValueError:
                bad_dimensions_metadata.append(image)
                height, width = -1, -1
            if split != "test":
                raw = row["Predict_class"]
                blank_labels += not bool(raw)
                try:
                    positive = parse_positive_classes(raw)
                except ValueError:
                    bad_labels.append((image, raw))
                    positive = frozenset()
                for class_id in range(1, 11):
                    (image_positive if class_id in positive else image_unlisted)[class_id] += 1
                if 10 in positive and len(positive) > 1:
                    no_finding_conflicts.append(study)
                if key in study_labels and study_labels[key] != positive:
                    inconsistent_labels.append(study)
                study_labels[key] = positive
            member = image_member(row, split)
            member_rows.append((member, height, width))
            expected_members[member] = (split, height, width)
        study_positive = Counter(class_id for labels in study_labels.values() for class_id in labels)
        coocc = Counter((a, b) for labels in study_labels.values() for a in labels for b in labels)
        for a in range(1, 11):
            class_rows.append({
                "split": split, "class_id": a, "class_name": CLASS_NAMES[a - 1],
                "positive_studies": study_positive[a] if split != "test" else "",
                "positive_rate_studies": round(study_positive[a] / len(study_images), 8) if split != "test" else "",
                "explicit_negative_studies": 0 if split != "test" else "",
                "uncertain_studies": 0 if split != "test" else "",
                "competition_negative_studies": len(study_images) - study_positive[a] if split != "test" else "",
                "positive_images": image_positive[a] if split != "test" else "",
                "competition_negative_images": image_unlisted[a] if split != "test" else "",
            })
            if split != "test":
                for b in range(1, 11):
                    co_rows.append({"split": split, "class_id_1": a, "class_id_2": b,
                                    "study_cooccurrence": coocc[a, b]})
        for (subject, study), count in sorted(study_images.items()):
            study_rows.append({"split": split, "Subject_id": subject, "Study_id": study,
                               "image_count": count,
                               "listed_positive_ids": ";".join(map(str, sorted(study_labels[(subject, study)]))) if split != "test" else ""})
        subjects_by_split[split] = set(subjects)
        studies_by_split[split] = {study for _, study in study_images}
        report["splits"][split] = {
            "csv_rows": len(rows), "patients": len(subjects), "studies": len(study_images),
            "images_referenced": len(member_rows), "views": dict(views), "patient_orientation": dict(orientations),
            "images_per_study": dict(sorted(Counter(study_images.values()).items())),
            "studies_per_patient": dict(sorted(Counter(len(v) for v in subjects.values()).items())),
            "duplicate_image_associations": len(repeated_associations),
            "duplicate_image_ids": len(duplicate_image_ids),
            "study_ids_with_conflicting_subject": len(set(conflicting_study_subject)),
            "inconsistent_study_labels": len(set(inconsistent_labels)) if split != "test" else None,
            "no_finding_cooccurrence_images": len(no_finding_conflicts) if split != "test" else None,
            "blank_Predict_class_rows": blank_labels if split != "test" else None,
            "invalid_Predict_class_rows": bad_labels[:20] if split != "test" else None,
            "invalid_dimension_metadata": bad_dimensions_metadata[:20],
            "samples": {"duplicate_associations": repeated_associations[:10], "duplicate_image_ids": duplicate_image_ids[:10],
                        "inconsistent_study_labels": inconsistent_labels[:10], "no_finding_conflicts": no_finding_conflicts[:10]},
        }

    with zipfile.ZipFile(path) as archive:
        image_members = {name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))}
        missing = sorted(set(expected_members) - image_members)
        orphan = sorted(image_members - set(expected_members))
        report["image_files"] = {
            "archive_image_count": len(image_members), "expected_unique_images": len(expected_members),
            "missing_count": len(missing), "missing_examples": missing[:20],
            "unreferenced_count": len(orphan), "unreferenced_examples": orphan[:20],
            "extensions": dict(Counter(Path(name).suffix.lower() for name in image_members)),
        }
        report["image_validation"] = {"checked": 0, "abnormal_dimension_rule": "min side <128px or max side >10000px",
                                      "unreadable": 0, "abnormal_dimensions": 0,
                                      "metadata_dimension_mismatch": 0, "examples": []}
        tasks = ((archive, member, height, width) for member, (_, height, width) in expected_members.items() if member in image_members)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for member, status, size, issue in pool.map(_check_image, tasks):
                report["image_validation"]["checked"] += 1
                if status == "unreadable":
                    report["image_validation"]["unreadable"] += 1
                elif issue in ("abnormal_dimensions", "metadata_dimension_mismatch"):
                    report["image_validation"][issue] += 1
                if status != "ok" or issue:
                    examples = report["image_validation"]["examples"]
                    if len(examples) < 25:
                        examples.append({"member": member, "size": size, "issue": issue or status})

    report["cross_split"] = {
        f"{a}_{b}_subject_overlap": len(subjects_by_split[a] & subjects_by_split[b])
        for a, b in combinations(("train", "val", "test"), 2)
    } | {
        f"{a}_{b}_study_id_overlap": len(studies_by_split[a] & studies_by_split[b])
        for a, b in combinations(("train", "val", "test"), 2)
    }
    _write_csv(output_dir / "class_distribution.csv", list(class_rows[0]), class_rows)
    _write_csv(output_dir / "study_distribution.csv", list(study_rows[0]), study_rows)
    _write_csv(output_dir / "label_cooccurrence.csv", list(co_rows[0]), co_rows)
    (output_dir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
