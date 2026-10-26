"""H17-02 batch=1 inference shared with cost.py; no Test labels are read."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from cxr.data.archive import image_member, read_split_rows, split_archive_path
from cxr.data.dataset import preprocess_image
from cxr.models.factory import create_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    fixed = {"model_id": "H17_DISTILL_02", "architecture": "efficientnet_v2_s",
             "num_classes": 10, "class_order": list(range(1, 11)), "input_size": 384,
             "preprocessing": "black_pad_v1", "aggregation": "study_mean_of_image_probabilities",
             "inference_precision": "fp16_autocast", "batch_size": 1}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Deployment config changed: {key}")
    return config


def load_models(config: dict, device: str = "cuda") -> dict:
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA GPU required for the fixed FP16 deployment path")
    weight = (HERE / config["model_file"]).resolve()
    if not weight.is_relative_to(HERE) or not weight.is_file():
        raise FileNotFoundError("H17 inference weight missing from package")
    if weight.stat().st_size != 81651999 or sha256(weight) != config["model_sha256"]:
        raise ValueError("H17 frozen inference weight size/SHA mismatch")
    model = create_model("efficientnet_v2_s", num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    if model.training:
        raise RuntimeError("Model must be in eval mode")
    return {"H17_DISTILL_02": model}


def predict_image(image: str | Path | Image.Image, models: dict, config: dict,
                  device: str = "cuda") -> np.ndarray:
    """One image to ten probabilities; Path form includes file read and decode."""
    if isinstance(image, Image.Image):
        tensor = preprocess_image(image, 384, preprocessing="black_pad_v1")
    else:
        with Image.open(image) as opened:
            tensor = preprocess_image(opened, 384, preprocessing="black_pad_v1")
    tensor = tensor.unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = models["H17_DISTILL_02"](tensor)
        probs = torch.sigmoid(logits).squeeze(0).float().cpu().numpy().astype(np.float64)
    if probs.shape != (10,) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid ten-class image probabilities")
    return probs


def read_image_manifest(manifest: Path, root: Path) -> list[tuple[str, str, Path]]:
    root = root.resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Subject_id", "Study_id", "image_path"]:
            raise ValueError("Image manifest columns must be Subject_id,Study_id,image_path")
        records = list(reader)
    rows = []
    for record in records:
        relative = Path(record["image_path"])
        if relative.is_absolute():
            raise ValueError("Absolute image path in manifest")
        image = (root / relative).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise FileNotFoundError(image)
        rows.append((record["Subject_id"], record["Study_id"], image))
    if len(rows) != len({str(path) for _, _, path in rows}):
        raise ValueError("Duplicate image paths")
    return rows


def predict_test(config: dict, models: dict, *, archive: Path | None = None,
                 image_manifest: Path | None = None, image_root: Path | None = None) -> tuple[dict, int]:
    grouped = defaultdict(list)
    seen_images = set()
    if archive is not None:
        rows = read_split_rows(archive, "test")  # Official Test metadata has no labels.
        with zipfile.ZipFile(split_archive_path(archive, "test")) as source:
            for row in rows:
                image_id = row["Dicom_id"]
                if image_id in seen_images:
                    raise ValueError("Duplicate Test image ID")
                seen_images.add(image_id)
                with Image.open(io.BytesIO(source.read(image_member(row, "test")))) as image:
                    prob = predict_image(image, models, config)
                grouped[(row["Subject_id"], row["Study_id"])].append(prob)
    else:
        if image_manifest is None or image_root is None:
            raise ValueError("Provide --archive or both --image-manifest and --image-root")
        rows = read_image_manifest(image_manifest, image_root)
        for subject, study, path in rows:
            seen_images.add(str(path))
            grouped[(subject, study)].append(predict_image(path, models, config))
    if len(rows) != 2218 or len(grouped) != 1414 or len(seen_images) != 2218:
        raise ValueError(f"Official Test coverage mismatch: {len(rows)} images / {len(grouped)} Studies")
    studies = {key: np.mean(value, axis=0) for key, value in grouped.items()}
    if any(value.shape != (10,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in studies.values()):
        raise ValueError("Invalid Study probabilities")
    return studies, len(rows)


def save_results(studies: dict, image_count: int, config_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "test_predictions.csv", out / "submission.csv", out / "prediction_manifest.json"]
    if any(path.exists() for path in files):
        raise FileExistsError("Refusing to overwrite existing candidate results")
    wide_temp, sub_temp = (out / "test_predictions.csv.tmp", out / "submission.csv.tmp")
    if wide_temp.exists() or sub_temp.exists():
        raise FileExistsError("Incomplete temporary output already exists; inspect before retrying")
    with wide_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)])
        for (subject, study), prob in sorted(studies.items()):
            writer.writerow([subject, study] + [format(float(x), ".17g") for x in prob])
    with sub_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability"])
        for (subject, study), prob in sorted(studies.items()):
            for class_id, value in enumerate(prob, start=1):
                writer.writerow([subject, study, class_id, format(float(value), ".17g")])
    os.replace(wide_temp, files[0])
    os.replace(sub_temp, files[1])
    manifest = {"model_id": "H17_DISTILL_02", "batch_size": 1, "precision": "FP16 autocast",
                "study_aggregation": "mean of image probabilities", "test_images": image_count,
                "test_studies": len(studies), "submission_rows": len(studies) * 10,
                "test_auc": None, "hidden_test_labels_used": False,
                "source_checkpoint_sha256": sha256(HERE / "model/H17_02.pth"),
                "config_sha256": sha256(config_path),
                "test_predictions_sha256": sha256(files[0]), "submission_sha256": sha256(files[1]),
                "latency_status": "T4_UNVERIFIED"}
    files[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--image-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.image_manifest and not args.image_root:
        parser.error("--image-root is required with --image-manifest")
    if args.archive and args.image_root:
        parser.error("--image-root requires --image-manifest")
    models = load_models(config)
    studies, count = predict_test(config, models, archive=args.archive,
                                  image_manifest=args.image_manifest, image_root=args.image_root)
    print(json.dumps(save_results(studies, count, args.config, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
