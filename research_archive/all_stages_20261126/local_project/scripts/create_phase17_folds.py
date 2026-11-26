"""Create and audit one immutable five-fold patient index over official Train."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from cxr.data.archive import archive_path, image_member, parse_positive_classes, read_split_rows
from cxr.data.patient_folds import INDEX_FILES, build_patient_folds, sha256
from cxr.utils.config import load_config


def write_csv(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    archive = archive_path(config)
    rows = read_split_rows(archive, "train")
    official_val = read_split_rows(archive, "val")
    folds = build_patient_folds(rows, seed=args.seed)
    studies = {}
    images = []
    for row in rows:
        subject, study, image = (row[key] for key in ("Subject_id", "Study_id", "Dicom_id"))
        key = (subject, study)
        positives = parse_positive_classes(row["Predict_class"])
        if key in studies and studies[key] != positives:
            raise ValueError(f"Inconsistent Study label: {key}")
        studies[key] = positives
        images.append({"Subject_id": subject, "Study_id": study, "Dicom_id": image,
                       "archive_member": image_member(row, "train"), "fold": folds[subject]})
    if (len(folds), len(studies), len(images)) != (11055, 18513, 30344):
        raise ValueError("Official Train size differs from frozen audit")
    val_subjects = {row["Subject_id"] for row in official_val}
    val_studies = {(row["Subject_id"], row["Study_id"]) for row in official_val}
    if set(folds) & val_subjects or set(studies) & val_studies:
        raise ValueError("Official Val overlaps a Train fold")
    if (len(val_subjects), len(val_studies), len(official_val)) != (228, 859, 1422):
        raise ValueError("Official Val counts differ from frozen protocol")
    patient_records = [{"Subject_id": subject, "fold": folds[subject]} for subject in sorted(folds)]
    study_records = [{"Subject_id": subject, "Study_id": study, "fold": folds[subject],
                      "Predict_class": ";".join(map(str, sorted(label)))}
                     for (subject, study), label in sorted(studies.items())]
    counts = {str(fold): {"patients": 0, "studies": 0, "images": 0,
                          "positive_patients": [0] * 10, "positive_studies": [0] * 10}
              for fold in range(1, 6)}
    patient_labels = defaultdict(set)
    for record in study_records:
        bucket = counts[str(record["fold"])]
        bucket["studies"] += 1
        for class_id in parse_positive_classes(record["Predict_class"]):
            bucket["positive_studies"][class_id - 1] += 1
            patient_labels[record["Subject_id"]].add(class_id)
    for record in patient_records:
        bucket = counts[str(record["fold"])]
        bucket["patients"] += 1
        for class_id in patient_labels[record["Subject_id"]]:
            bucket["positive_patients"][class_id - 1] += 1
    for record in images:
        counts[str(record["fold"])]["images"] += 1
    if any(not all(bucket["positive_patients"]) or not all(bucket["positive_studies"])
           for bucket in counts.values()):
        raise ValueError("At least one fold lacks a positive class")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Fold directory already exists: {output}")
    output.mkdir(parents=True)
    write_csv(output / "patient_folds.csv", ["Subject_id", "fold"], patient_records)
    write_csv(output / "study_folds.csv", ["Subject_id", "Study_id", "fold", "Predict_class"], study_records)
    write_csv(output / "image_folds.csv", ["Subject_id", "Study_id", "Dicom_id", "archive_member", "fold"], images)
    hashes = {name: sha256(output / name) for name in INDEX_FILES}
    report = {"source_archive": str(archive), "seed": args.seed, "fold_count": 5,
              "method": "patient-level rare-label-first greedy balance; fixed RNG tie ordering",
              "source_counts": {"patients": len(folds), "studies": len(studies), "images": len(images)},
              "official_val_counts": {"patients": len(val_subjects), "studies": len(val_studies), "images": len(official_val)},
              "patient_overlap_across_folds": 0, "study_overlap_across_folds": 0,
              "official_val_patient_overlap": 0, "official_val_study_overlap": 0,
              "folds": counts, "index_sha256": hashes,
              "official_val_and_hidden_test_used_for_fold_assignment": False}
    (output / "fold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "fold_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
