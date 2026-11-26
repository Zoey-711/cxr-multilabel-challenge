"""Validate a fully extracted M02 cache after a metadata check interruption."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from cxr.utils.config import load_config


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = root / config["source"]["m02_run"]
    cache = root / config["source"]["feature_cache"]
    if (cache / "manifest.json").exists():
        raise FileExistsError("Cache already finalized")
    frozen = json.loads((source / "frozen_manifest.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha(source / "best_model.pt")
    if checkpoint_hash != frozen["best_model.pt"]["sha256"] or checkpoint_hash != config["source"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA mismatch")
    m02 = load_config(source / "config.yaml")
    if m02["data"]["input_size"] != 384 or m02["model"]["name"] != "convnext_tiny":
        raise ValueError("Unexpected frozen M02 config")
    report = {"checkpoint_sha256": checkpoint_hash, "m02_config_sha256": sha(source / "config.yaml"),
              "archive_path": m02["data"]["archive_path"], "input_size": 384,
              "preprocessing": "black_pad_v1; RGB, aspect-preserving resize and black pad, ImageNet normalize",
              "training_feature_difference": "Training images were randomly augmented by M02; this cache uses no random augmentation.",
              "splits": {}}
    all_subjects = {}
    for split, expected_images, expected_studies in (("train", 30344, 18513), ("val", 1422, 859)):
        features = np.load(cache / f"{split}_features.npy", mmap_mode="r")
        logits = np.load(cache / f"{split}_logits.npy", mmap_mode="r")
        with (cache / f"{split}_images.csv").open(encoding="utf-8", newline="") as handle:
            images = list(csv.DictReader(handle))
        with (cache / f"{split}_studies.csv").open(encoding="utf-8", newline="") as handle:
            studies = list(csv.DictReader(handle))
        if features.shape != (expected_images, 768) or logits.shape != (expected_images, 10):
            raise ValueError(f"Incomplete {split} arrays")
        if len(images) != expected_images or len(studies) != expected_studies:
            raise ValueError(f"Incomplete {split} mapping")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise ValueError(f"Nonfinite {split} features/logits")
        image_keys = [(r["Subject_id"], r["Study_id"], r["Image_id"]) for r in images]
        study_keys = {(r["Subject_id"], r["Study_id"]) for r in studies}
        if len(set(image_keys)) != len(images) or {key[:2] for key in image_keys} != study_keys:
            raise ValueError(f"Invalid {split} image-to-Study mapping")
        if [int(r["image_index"]) for r in images] != list(range(expected_images)):
            raise ValueError(f"Invalid {split} feature row indices")
        if any(any(r[f"label_{j}"] not in ("0", "1") for j in range(1, 11)) for r in studies):
            raise ValueError(f"Invalid {split} label encoding")
        all_subjects[split] = {r["Subject_id"] for r in studies}
        report["splits"][split] = {"images": len(images), "studies": len(studies),
                                   "patients": len(all_subjects[split])}
    if all_subjects["train"] & all_subjects["val"]:
        raise ValueError("Train/Val patient overlap")
    with (source / "validation_analysis" / "image_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        frozen_probs = {(r["Subject_id"], r["Study_id"], r["Image_id"]): np.array(
            [float(r[f"probability_{j}"]) for j in range(1, 11)]) for r in csv.DictReader(handle)}
    val_logits = np.load(cache / "val_logits.npy", mmap_mode="r")
    with (cache / "val_images.csv").open(encoding="utf-8") as handle:
        val_images = list(csv.DictReader(handle))
    if len(frozen_probs) != len(val_images):
        raise ValueError("Frozen M02 image prediction coverage differs")
    differences = []
    for r in val_images:
        key = (r["Subject_id"], r["Study_id"], r["Image_id"])
        actual = torch.sigmoid(torch.tensor(val_logits[int(r["image_index"])], dtype=torch.float16)).float().numpy()
        differences.append(float(np.max(np.abs(actual - frozen_probs[key]))))
    maximum = max(differences)
    report["max_probability_difference_from_frozen_m02_val_images"] = maximum
    if maximum > 0.0005:
        raise ValueError(f"Frozen M02 validation probabilities differ by {maximum}")
    report["files_sha256"] = {p.name: sha(p) for p in cache.iterdir() if p.is_file()}
    (cache / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "max_probability_difference": maximum}))


if __name__ == "__main__":
    main()
