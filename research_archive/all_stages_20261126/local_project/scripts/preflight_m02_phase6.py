"""Validate M02 data/augmentation protocol against frozen M01 before training."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import archive_path, parse_positive_classes, read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.models.factory import imagenet_weight_info
from cxr.utils.config import load_config


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    m02 = load_config(args.config)
    frozen = args.m01_run_dir.resolve()
    m01 = load_config(frozen / "config.yaml")
    output = args.output_dir.resolve()
    if output == frozen or frozen in output.parents:
        parser.error("Preflight output must be outside frozen M01")
    run_dir = Path("outputs/experiments") / args.run_id
    if run_dir.exists():
        parser.error(f"M02 formal output directory already exists: {run_dir}")
    proposed = json.loads(json.dumps(m02))
    proposed["experiment"]["name"] = m01["experiment"]["name"]
    augmentation = proposed["data"].pop("train_augmentation", None)
    if proposed != m01:
        parser.error("M02 differs from frozen M01 beyond experiment name and augmentation")
    expected_aug = {"enabled": True, "name": "conservative_cxr_v1", "affine_probability": 0.5,
                    "rotation_degrees": 3.0, "translation_fraction": 0.02,
                    "photometric_probability": 0.5, "brightness_contrast_fraction": 0.05}
    if augmentation != expected_aug:
        parser.error("M02 enhancement parameters changed")
    if m02["model"] != m01["model"] or m02["training"] != m01["training"] or \
            m02["validation"] != m01["validation"]:
        parser.error("Model/training/validation protocol changed")
    if m02["training"]["batch_size"] != 8 or m02["training"]["gradient_accumulation_steps"] != 2 or \
            m02["training"]["early_stopping_patience"] != 6 or \
            m02["validation"]["study_fusion"] != "mean_probability":
        parser.error("Requested batch/early-stop/aggregation settings changed")
    project_root = Path(__file__).resolve().parents[1]
    _, _, weight_name = imagenet_weight_info("convnext_tiny")
    weight_path = project_root / ".wheels" / "torch_cache" / "hub" / "checkpoints" / weight_name
    weight_sha = digest(weight_path)
    m01_source = json.loads((frozen / "pretrained_source.json").read_text(encoding="utf-8"))
    if weight_sha != m01_source["sha256"]:
        parser.error("Cached ImageNet weight differs from frozen M01 source")
    frozen_manifest = json.loads((frozen / "frozen_manifest.json").read_text(encoding="utf-8"))
    for name in ("config.yaml", "best_model.pt", "validation_analysis/image_predictions.csv"):
        if digest(frozen / name) != frozen_manifest[name]["sha256"]:
            parser.error(f"Frozen M01 artifact changed: {name}")
    archive = archive_path(m02)
    split_rows = {split: read_split_rows(archive, split) for split in ("train", "val", "test")}
    train_rows = split_rows["train"][:8]
    val_rows = split_rows["val"][:8]
    test_rows = split_rows["test"][:2]
    policy = m02["data"]["label_encoding"]["unlisted_policy"]
    torch.manual_seed(int(m02["experiment"]["seed"]))
    datasets = {
        "train_augmented": CXRArchiveDataset(archive, "train", rows=train_rows,
                                              unlisted_policy=policy, augmentation=augmentation),
        "train_baseline": CXRArchiveDataset(archive, "train", rows=train_rows, unlisted_policy=policy),
        "val_m01": CXRArchiveDataset(archive, "val", rows=val_rows, unlisted_policy=policy),
        "val_m02": CXRArchiveDataset(archive, "val", rows=val_rows, unlisted_policy=policy),
        "test_m01": CXRArchiveDataset(archive, "test", rows=test_rows),
        "test_m02": CXRArchiveDataset(archive, "test", rows=test_rows),
    }
    augmented_pixels = []
    differences = 0
    try:
        for index, row in enumerate(train_rows):
            augmented = datasets["train_augmented"][index]
            baseline = datasets["train_baseline"][index]
            for key in ("subject_id", "study_id", "image_id"):
                if augmented[key] != baseline[key]:
                    raise ValueError(f"Training label/image ID alignment changed: {key}")
            if not torch.equal(augmented["target"], baseline["target"]) or \
                    not torch.equal(augmented["valid_mask"], baseline["valid_mask"]):
                raise ValueError("Training augmentation changed targets/masks")
            expected = parse_positive_classes(row["Predict_class"])
            if {j + 1 for j in augmented["target"].nonzero().flatten().tolist()} != expected:
                raise ValueError("Training label index misalignment")
            pixels = augmented["image"]
            if tuple(pixels.shape) != (3, 384, 384) or not bool(torch.isfinite(pixels).all()):
                raise ValueError("Augmented image shape or finiteness failed")
            augmented_pixels.append(pixels.flatten().numpy())
            differences += int(not torch.equal(pixels, baseline["image"]))
        for split, rows in (("val", val_rows), ("test", test_rows)):
            for index, row in enumerate(rows):
                first = datasets[f"{split}_m01"][index]
                second = datasets[f"{split}_m02"][index]
                if not torch.equal(first["image"], second["image"]):
                    raise ValueError(f"{split} preprocessing changed")
                if (first["subject_id"], first["study_id"], first["image_id"]) != \
                        (row["Subject_id"], row["Study_id"], row["Dicom_id"]):
                    raise ValueError(f"{split} image association mismatch")
                if split == "test" and ("target" in first or "target" in second):
                    raise ValueError("Test dataset constructed a fake label")
        if differences == 0:
            raise ValueError("No sampled training image received augmentation")
    finally:
        for dataset in datasets.values():
            dataset.close()
    pixels = np.concatenate(augmented_pixels)
    result = {"passed": True, "m01_run_dir": str(frozen), "m02_run_dir_reserved": str(run_dir.resolve()),
              "m02_run_dir_absent": True, "only_config_differences": ["experiment.name", "data.train_augmentation"],
              "enhancement_is_joint_strategy_comparison": True,
              "augmentation": augmentation,
              "train_samples_checked": len(train_rows), "train_samples_pixels_changed": differences,
              "augmented_image_shape": [3, 384, 384], "normalized_pixel_min": float(pixels.min()),
              "normalized_pixel_max": float(pixels.max()), "normalized_pixel_mean": float(pixels.mean()),
              "normalized_pixel_std": float(pixels.std()), "nan_or_inf": 0,
              "validation_samples_exact_match": len(val_rows), "test_samples_exact_match": len(test_rows),
              "test_labels_constructed": False, "labels_and_ids_exact_match": True,
              "model": m02["model"], "training": m02["training"], "validation": m02["validation"],
              "seed": m02["experiment"]["seed"], "pretrained_weight_sha256": weight_sha,
              "m01_frozen_artifact_hashes_match": True, "training_started": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# M02 formal preflight", "",
             "Passed. Frozen M01 runtime config and artifact hashes verified. Only M02 experiment name and train augmentation differ.",
             "Augmentation is one combined strategy; any future effect cannot be attributed to a single operator.",
             f"8 train images checked, {differences} changed; pixel range after normalization "
             f"[{result['normalized_pixel_min']:.3f}, {result['normalized_pixel_max']:.3f}], "
             f"mean {result['normalized_pixel_mean']:.3f}, SD {result['normalized_pixel_std']:.3f}; no NaN/Inf.",
             "All targets and subject/study/image identifiers remained aligned. Eight validation and two test images exactly matched M01 deterministic preprocessing; test has no constructed targets.",
             "Model, seed, AdamW/BCE, learning rates, batch 8 × accumulation 2, mean Study aggregation, patience 6, and ImageNet weight hash match M01.",
             f"Reserved formal output: `{run_dir}` (absent at check). No training has started in this preflight."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "train_changed": differences,
                      "pixel_range": [result["normalized_pixel_min"], result["normalized_pixel_max"]],
                      "run_dir": str(run_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
