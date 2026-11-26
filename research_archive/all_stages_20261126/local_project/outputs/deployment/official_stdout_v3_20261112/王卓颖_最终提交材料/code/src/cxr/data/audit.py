"""Schema-driven manifest audit; never guesses source column names or label codes."""

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ManifestSchema:
    subject_id: str
    study_id: str
    image_path: str
    labels: tuple[str, ...] = ()

    def required_columns(self) -> tuple[str, ...]:
        return (self.subject_id, self.study_id, self.image_path, *self.labels)


@dataclass
class AuditReport:
    row_count: int = 0
    subject_count: int = 0
    study_count: int = 0
    duplicate_image_rows: int = 0
    studies_with_multiple_images: int = 0
    subjects_with_multiple_studies: int = 0
    blank_ids: int = 0
    blank_image_paths: int = 0
    label_values: dict[str, dict[str, int]] = field(default_factory=dict)
    missing_image_paths: list[str] = field(default_factory=list)
    inconsistent_study_subjects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_rows(
    rows: Iterable[Mapping[str, str | None]],
    schema: ManifestSchema,
    *,
    image_root: str | Path | None = None,
) -> AuditReport:
    """Audit image-level rows while counting unique subjects and studies separately."""
    report = AuditReport()
    studies: dict[str, set[str]] = defaultdict(set)
    study_images: Counter[str] = Counter()
    subjects: dict[str, set[str]] = defaultdict(set)
    seen_images: set[tuple[str, str]] = set()
    labels: dict[str, Counter[str]] = {name: Counter() for name in schema.labels}
    root = Path(image_root) if image_root is not None else None

    for row in rows:
        absent = [name for name in schema.required_columns() if name not in row]
        if absent:
            raise ValueError(f"Manifest is missing configured columns: {absent}")
        report.row_count += 1
        subject = str(row[schema.subject_id] or "").strip()
        study = str(row[schema.study_id] or "").strip()
        image = str(row[schema.image_path] or "").strip()
        if not subject or not study:
            report.blank_ids += 1
        if not image:
            report.blank_image_paths += 1
        if subject and study:
            studies[study].add(subject)
            subjects[subject].add(study)
            if image:
                study_images[study] += 1
        if image:
            key = (study, image)
            if key in seen_images:
                report.duplicate_image_rows += 1
            seen_images.add(key)
            if root is not None and not (root / image).is_file():
                report.missing_image_paths.append(image)
        for name in schema.labels:
            value = row[name]
            labels[name]["<blank>" if value is None or str(value).strip() == "" else str(value).strip()] += 1

    report.subject_count = len(subjects)
    report.study_count = len(studies)
    report.studies_with_multiple_images = sum(n > 1 for n in study_images.values())
    report.subjects_with_multiple_studies = sum(len(v) > 1 for v in subjects.values())
    report.inconsistent_study_subjects = sorted(k for k, v in studies.items() if len(v) > 1)
    report.label_values = {name: dict(counts) for name, counts in labels.items()}
    return report


def audit_csv(path: str | Path, schema: ManifestSchema, *, image_root: str | Path | None = None) -> AuditReport:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return audit_rows(csv.DictReader(handle), schema, image_root=image_root)
