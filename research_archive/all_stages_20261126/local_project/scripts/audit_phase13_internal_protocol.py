"""Audit frozen patient split and write a no-training retrain protocol."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from cxr.data.archive import parse_positive_classes


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Recheck and replace this script's own audit outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split = root / cfg["source"]["split_directory"]
    output = args.output_dir or root / "outputs/analysis/phase13_internal_protocol"
    if output.exists() and not args.refresh:
        raise FileExistsError(output)
    if output.exists() and {p.name for p in output.iterdir()} != {"audit.json", "report.md", "class_support.csv"}:
        raise ValueError("Refusing to replace unexpected output contents")
    patients = read_csv(split / "patient_split.csv")
    studies = read_csv(split / "study_split.csv")
    images = read_csv(split / "image_split.csv")
    reference = json.loads((split / "split_report.json").read_text(encoding="utf-8"))
    patient_map = {row["Subject_id"]: row["split"] for row in patients}
    if len(patient_map) != len(patients):
        raise ValueError("Duplicate patient split records")
    groups = {label: {patient for patient, assignment in patient_map.items() if assignment == label}
              for label in ("internal_train", "internal_val")}
    if groups["internal_train"] & groups["internal_val"]:
        raise ValueError("Patient leakage")
    study_keys = {(row["Subject_id"], row["Study_id"]) for row in studies}
    image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in images}
    if len(study_keys) != len(studies) or len(image_keys) != len(images):
        raise ValueError("Duplicate Study or image keys")
    if any(patient_map.get(row["Subject_id"]) != row["split"] for row in studies + images):
        raise ValueError("Study/image violates patient assignment")
    if {(row["Subject_id"], row["Study_id"]) for row in images} != study_keys:
        raise ValueError("Study/image mapping mismatch")
    counts = {label: {"patients": len(groups[label]),
                      "studies": sum(row["split"] == label for row in studies),
                      "images": sum(row["split"] == label for row in images)}
              for label in groups}
    if counts != reference["counts"] or counts["internal_train"] != {"patients": 9949, "studies": 16667, "images": 27301} or counts["internal_val"] != {"patients": 1106, "studies": 1846, "images": 3043}:
        raise ValueError("Frozen split counts disagree")
    supports = []
    for label in groups:
        subset = [row for row in studies if row["split"] == label]
        for class_id in range(1, 11):
            positives = [row for row in subset if class_id in parse_positive_classes(row["Predict_class"])]
            if not positives or len(positives) == len(subset):
                raise ValueError(f"Class {class_id} lacks support in {label}")
            supports.append({"split": label, "class_id": class_id,
                             "positive_studies": len(positives), "negative_studies": len(subset) - len(positives),
                             "positive_patients": len({row["Subject_id"] for row in positives})})
    m02 = root / cfg["source"]["original_m02_run"]
    pretrained = json.loads((m02 / "pretrained_source.json").read_text(encoding="utf-8"))
    if pretrained["sha256"] != cfg["source"]["pretrained_sha256"]:
        raise ValueError("Internal retrain ImageNet weight hash differs from M02")
    logs = [json.loads(row) for row in (m02 / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    observed_seconds = sum(row["elapsed_seconds"] for row in logs)
    projected_epoch = observed_seconds / len(logs) * counts["internal_train"]["images"] / 30344
    estimate = {"source_m02_epochs": len(logs), "source_m02_hours": observed_seconds / 3600,
                "internal_backbone_minutes_per_epoch_estimate": projected_epoch / 60,
                "internal_backbone_hours_if_11_epochs": projected_epoch * 11 / 3600,
                "internal_backbone_hours_if_30_epochs": projected_epoch * 30 / 3600,
                "method": "M02 observed mean epoch time scaled by internal Train image count; excludes feature extraction and head training"}
    if cfg["source"]["original_m02_checkpoint_usable_for_holdout_evaluation"] is not False:
        raise ValueError("Protocol must reject frozen M02 for internal holdout")
    output.mkdir(parents=True, exist_ok=args.refresh)
    with (output / "class_support.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(supports[0]))
        writer.writeheader(); writer.writerows(supports)
    report = {"counts": counts, "patient_overlap": 0, "class_support": supports,
              "split_sha256": {p.name: sha(p) for p in split.iterdir() if p.is_file()},
              "pretrained_sha256": pretrained["sha256"],
              "protocol_config_sha256": sha(args.config), "training_time_estimate": estimate,
              "frozen_m02_internal_holdout_is_invalid": True,
              "official_validation_role": "Only checkpoint selection for new backbone and V03 head",
              "internal_holdout_role": "Single final comparison after choices frozen; never Train, early stopping or tuning"}
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase 13 independent-patient protocol", "",
             "The existing patient-level Train/Holdout files were read without changes. All images and Studies follow their Subject_id split; no patient crosses splits.",
             "", "| Split | Patients | Studies | Images |", "|---|---:|---:|---:|"]
    for label, count in counts.items():
        lines.append(f"| {label} | {count['patients']} | {count['studies']} | {count['images']} |")
    lines += ["", "Every class has positive and negative Studies in both splits. The scarcest Holdout class is class 1: 46 positive Studies across 43 patients. Exact Study and positive-patient supports are in `class_support.csv`.",
              "", "The frozen M02 checkpoint already trained on all official Train patients, including internal Holdout. It cannot evaluate generalization to that Holdout.",
              "", "Future authorized run: initialize ConvNeXt-Tiny from the same verified ImageNet weights, train only the 27,301 internal-Train images with the frozen M02 augmentation, 384 preprocessing, image shuffle, BCE, AdamW rates, schedule, seed and Study Mean validation. Select the backbone checkpoint using only the official 859-Study validation set. Extract deterministic features from the newly trained backbone for internal Train, official validation and internal Holdout. Fit V03 only on internal Train Studies, selecting its checkpoint using the official validation set; compare its Study predictions against Mean probability from the same backbone. Evaluate internal Holdout exactly once, with ten-class AUC/AP and paired patient bootstrap. Neither method may use Holdout for early stopping, tuning or training.",
              "", f"Estimated backbone time: {estimate['internal_backbone_minutes_per_epoch_estimate']:.1f} min/epoch, {estimate['internal_backbone_hours_if_11_epochs']:.1f} h if 11 epochs or {estimate['internal_backbone_hours_if_30_epochs']:.1f} h at the 30-epoch maximum on this machine. Add feature extraction, head fitting and validation time. Only one backbone retrain is needed because baseline and V03 share it. This is a scaling estimate, not a measured run.",
              "", "The official validation set has been used in prior experiment selection. The internal Holdout remains untouched and is the final independent comparison for this protocol. No internal retraining was started."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "estimated_minutes_per_epoch": estimate["internal_backbone_minutes_per_epoch_estimate"]}))


if __name__ == "__main__":
    main()
