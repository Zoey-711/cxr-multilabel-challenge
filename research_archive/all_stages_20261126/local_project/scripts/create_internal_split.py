"""Create a fixed, patient-disjoint index split of official training data."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cxr.data.archive import archive_path, image_member, parse_positive_classes, read_split_rows
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 0.5:
        parser.error("holdout-fraction must be between 0 and 0.5")
    config = load_config(args.config)
    archive = archive_path(config)
    train_rows = read_split_rows(archive, "train")
    official_val_rows = read_split_rows(archive, "val")
    study_info = {}
    for row in train_rows:
        key = (row["Subject_id"], row["Study_id"])
        positives = parse_positive_classes(row["Predict_class"])
        if key in study_info and study_info[key] != positives:
            raise ValueError(f"Inconsistent Study labels: {key}")
        study_info[key] = positives
    train_patients = sorted({row["Subject_id"] for row in train_rows})
    official_patients = {row["Subject_id"] for row in official_val_rows}
    if official_patients.intersection(train_patients):
        raise ValueError("Official validation and training patients overlap")
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(train_patients)
    holdout_count = round(len(train_patients) * args.holdout_fraction)
    holdout = set(shuffled[:holdout_count])
    split_of = lambda subject: "internal_val" if subject in holdout else "internal_train"
    patient_rows = [{"Subject_id": subject, "split": split_of(subject)} for subject in train_patients]
    study_rows = [{"Subject_id": subject, "Study_id": study,
                   "split": split_of(subject), "Predict_class": ";".join(map(str, sorted(positives)))}
                  for (subject, study), positives in sorted(study_info.items())]
    image_rows = [{"Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                   "Dicom_id": row["Dicom_id"], "archive_member": image_member(row, "train"),
                   "split": split_of(row["Subject_id"])} for row in train_rows]
    split_studies = defaultdict(list)
    for row in study_rows:
        split_studies[row["split"]].append(row)
    class_rows = []
    for split in ("internal_train", "internal_val"):
        rows = split_studies[split]
        for class_id, name in enumerate(CLASS_NAMES, 1):
            positives = sum(class_id in parse_positive_classes(row["Predict_class"]) for row in rows)
            class_rows.append({"split": split, "class_id": class_id, "class_name": name,
                               "positive_studies": positives, "negative_studies": len(rows) - positives,
                               "positive_rate": positives / len(rows)})
    counts = {split: {"patients": sum(row["split"] == split for row in patient_rows),
                      "studies": len(split_studies[split]),
                      "images": sum(row["split"] == split for row in image_rows)}
              for split in ("internal_train", "internal_val")}
    if len(patient_rows) != 11055 or len(study_rows) != 18513 or len(image_rows) != 30344:
        raise ValueError("Official training data counts changed")
    if any(row["positive_studies"] == 0 or row["negative_studies"] == 0 for row in class_rows):
        raise ValueError("A class lacks positive or negative Studies in one internal split")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "patient_split.csv", ["Subject_id", "split"], patient_rows)
    write_csv(output / "study_split.csv", ["Subject_id", "Study_id", "split", "Predict_class"], study_rows)
    write_csv(output / "image_split.csv", ["Subject_id", "Study_id", "Dicom_id", "archive_member", "split"], image_rows)
    write_csv(output / "class_distribution.csv", ["split", "class_id", "class_name", "positive_studies",
                                                   "negative_studies", "positive_rate"], class_rows)
    report = {"source_archive": str(archive), "seed": args.seed,
              "holdout_fraction_requested": args.holdout_fraction,
              "patient_selection": "sorted Subject_id, numpy default_rng(seed).permutation, first round(N*fraction)",
              "counts": counts, "patient_overlap": 0, "official_validation_patient_overlap": 0,
              "official_validation_patients": len(official_patients),
              "all_ten_classes_have_positive_and_negative_studies_in_both_splits": True,
              "class_distribution": class_rows,
              "purpose": "Fixed internal validation indices only; no model training performed"}
    (output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Fixed patient-level internal split", "",
             f"Seed {args.seed}; held out {len(holdout)}/{len(train_patients)} train patients using one fixed random permutation.",
             "All images and Studies of each Subject_id remain in one split. Official validation patients are disjoint.",
             "No training or cross-validation was run.", "",
             "| Split | Patients | Studies | Images |", "|---|---:|---:|---:|"]
    for split, count in counts.items():
        lines.append(f"| {split} | {count['patients']} | {count['studies']} | {count['images']} |")
    lines += ["", "The class-level distribution is in `class_distribution.csv`; index files refer to the original ZIP without copying images."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
