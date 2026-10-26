"""Audit frozen M02 geometry and preflight the single-variable M03 padding change."""

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_MEAN_RGB8, IMAGENET_STD, make_dataloader, resize_pad_image
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.config import load_config


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="M03 config")
    parser.add_argument("--m02-config", required=True, type=Path, help="Frozen M02 runtime config")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    m02, m03 = load_config(args.m02_config), load_config(args.config)
    before, after = flatten(m02), flatten(m03)
    differences = {key: {"M02": before.get(key), "M03": after.get(key)}
                   for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
    if set(differences) != {"experiment.name", "data.preprocessing"}:
        raise ValueError(f"Unexpected M02/M03 configuration differences: {differences}")
    if differences["data.preprocessing"] != {"M02": None, "M03": "imagenet_mean_pad_v1"}:
        raise ValueError("M03 must only change black padding to ImageNet mean padding")
    source = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/pretrained_source.json")
    provenance = json.loads(source.read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parents[1]
    _, _, weight_name = imagenet_weight_info("convnext_tiny")
    weights = project / ".wheels" / "torch_cache" / "hub" / "checkpoints" / weight_name
    if not weights.is_file() or sha256(weights) != provenance["sha256"]:
        raise ValueError("Cached ImageNet weight differs from frozen M02 provenance")
    archive = archive_path(m02)
    split_rows = {split: read_split_rows(archive, split) for split in ("train", "val")}
    examples, padding = [], []
    for split, rows in split_rows.items():
        for row in rows:
            width, height = int(row["Columns"]), int(row["Rows"])
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image geometry: {row['Dicom_id']}")
            fraction = 1 - min(width, height) / max(width, height)
            padding.append(fraction)
        sorted_rows = sorted(rows, key=lambda r: int(r["Columns"]) / int(r["Rows"]))
        for label, row in (("tall", sorted_rows[0]),
                           ("square", min(rows, key=lambda r: abs(int(r["Columns"]) / int(r["Rows"]) - 1))),
                           ("wide", sorted_rows[-1])):
            examples.append((split, label, row))
    args.output_dir.mkdir(parents=True)
    visual = args.output_dir / "visuals"
    visual.mkdir()
    sample_records = []
    with zipfile.ZipFile(archive) as zf:
        for split, label, row in examples:
            member = image_member(row, split)
            original_bytes = zf.read(member)
            stem = f"{split}_{label}_{row['Dicom_id']}"
            original_path = visual / f"{stem}_original.jpg"
            original_path.write_bytes(original_bytes)
            with Image.open(io.BytesIO(original_bytes)) as image:
                image.load()
                black = resize_pad_image(image, preprocessing="black_pad_v1")
                mean = resize_pad_image(image, preprocessing="imagenet_mean_pad_v1")
                black.save(visual / f"{stem}_m02.png")
                mean.save(visual / f"{stem}_m03.png")
                sample_records.append({"split": split, "selection": label,
                                       "subject_id": row["Subject_id"], "study_id": row["Study_id"],
                                       "image_id": row["Dicom_id"], "source_size": list(image.size),
                                       "source_aspect": image.width / image.height,
                                       "original": str(original_path.relative_to(args.output_dir)),
                                       "M02": f"visuals/{stem}_m02.png", "M03": f"visuals/{stem}_m03.png"})
    if not torch.cuda.is_available():
        raise RuntimeError("M03 forward smoke requires CUDA; no CPU fallback")
    torch.manual_seed(int(m03["experiment"]["seed"]))
    policy = m03["data"]["label_encoding"]["unlisted_policy"]
    smoke = {}
    for split in ("train", "val"):
        # Two actual ZIP images, no training update or persistent worker process.
        loader = make_dataloader(archive, split, batch_size=2, num_workers=0,
                                 rows=split_rows[split][:2], unlisted_policy=policy,
                                 augmentation=m03["data"].get("train_augmentation") if split == "train" else None,
                                 preprocessing=m03["data"]["preprocessing"])
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 3, 384, 384)
        assert batch["target"].shape == batch["valid_mask"].shape == (2, 10)
        assert torch.isfinite(batch["image"]).all() and torch.isfinite(batch["target"]).all()
        assert all(batch["study_id"]) and all(batch["subject_id"])
        smoke[split] = {"shape": list(batch["image"].shape),
                        "study_ids": list(batch["study_id"]),
                        "finite": True, "label_shape": list(batch["target"].shape)}
        if split == "val":
            val_images = batch["image"].to("cuda")
        loader.dataset.close()
    os.environ["TORCH_HOME"] = str(project / ".wheels" / "torch_cache")
    torch.cuda.reset_peak_memory_stats()
    model = create_model("convnext_tiny", pretrained=True).cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(val_images)
    if logits.shape != (2, 10) or not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite or misaligned M03 GPU forward")
    smoke["gpu_forward"] = {"shape": list(logits.shape), "finite": True,
                             "gpu": torch.cuda.get_device_name(),
                             "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    del model, logits, val_images
    torch.cuda.empty_cache()
    report = {
        "source_archive": str(archive), "M02_config_sha256": sha256(args.m02_config),
        "M03_config_sha256": sha256(args.config), "pretrained_sha256": sha256(weights),
        "config_differences": differences,
        "geometry": {"train_images": len(split_rows["train"]), "val_images": len(split_rows["val"]),
                     "total_image_records": len(padding), "crop_fraction": 0.0,
                     "cropped_image_records": 0, "nonuniform_resize_records": 0,
                     "padding_area_fraction_quantiles": {str(q): float(np.quantile(padding, q))
                                                         for q in (0, .25, .5, .75, .95, 1)}},
        "M02_pipeline": ["JPEG decode", "RGB", "ImageOps.pad bilinear fit 384 centered black",
                         "ToTensor / 255", "ImageNet normalize"],
        "M03_pipeline": ["JPEG decode", "RGB", "ImageOps.pad bilinear fit 384 centered ImageNet mean RGB8",
                         "ToTensor / 255", "ImageNet normalize"],
        "M03_fill_rgb8": list(IMAGENET_MEAN_RGB8),
        "M02_normalized_fill": [round((0 - m) / s, 6) for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        "M03_normalized_fill": [round((c / 255 - m) / s, 6)
                                  for c, m, s in zip(IMAGENET_MEAN_RGB8, IMAGENET_MEAN, IMAGENET_STD)],
        "examples": sample_records, "smoke": smoke,
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# M03 preprocessing visualization audit", "",
             "M02 already preserves the whole source image with aspect-ratio fit and black padding. "
             "M03 keeps identical geometry and uses the rounded ImageNet mean RGB fill. "
             "It tests border intensity, not a crop repair.", "",
             f"Train/validation image records: {len(padding)}; preprocessing crop records: 0. "
             "This is a code/metadata geometry conclusion; it cannot rule out anatomy outside the acquired source image.",
             "Approximate padding area fraction quantiles: " + str(report["geometry"]["padding_area_fraction_quantiles"]),
             "The M03 border has RGB color (124,116,104) around a grayscale chest radiograph; "
             "this is a visible distribution change and a possible artifact, not a guaranteed improvement.",
             "", "| Split | Aspect | Original | M02 black pad | M03 mean pad |", "|---|---|---|---|---|"]
    for sample in sample_records:
        lines.append(f"| {sample['split']} | {sample['source_aspect']:.3f} | "
                     f"[original]({sample['original']}) | [M02]({sample['M02']}) | [M03]({sample['M03']}) |")
    (args.output_dir / "visual_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "config_diff.md").write_text(
        "# M02/M03 config differences\n\n" +
        "Only algorithmic change: `data.preprocessing` (M02 implicit `black_pad_v1` → "
        "M03 `imagenet_mean_pad_v1`). `experiment.name` is bookkeeping. "
        "All model, optimizer, sampling, augmentation, seed, validation, and resolution fields match.\n\n" +
        "```json\n" + json.dumps(differences, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    (args.output_dir / "training_preflight.md").write_text(
        "# M03 training preflight\n\n"
        f"Cached official ImageNet ConvNeXt-Tiny SHA-256: `{report['pretrained_sha256']}` (matches M02).\n\n"
        f"CUDA forward: {smoke['gpu_forward']}; train/val DataLoader samples: valid.\n\n"
        "Independent future run ID: `m03_convnext_tiny_meanpad_384_<timestamp>`; no run was started.\n\n"
        "Training command after user approval: "
        "`.\\.venv\\Scripts\\python.exe scripts\\train.py --config configs\\m03_convnext_tiny_meanpad_384.yaml`.\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "crop_records": 0,
                      "images_audited": len(padding), "gpu_forward": smoke["gpu_forward"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
