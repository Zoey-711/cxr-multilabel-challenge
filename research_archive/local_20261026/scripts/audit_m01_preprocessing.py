"""Inspect frozen M01's image geometry on a few real train/validation images."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as TF

from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from cxr.utils.config import load_config


def selected_rows(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    candidates = [(int(row["Columns"]) / int(row["Rows"]), row) for row in rows
                  if int(row["Columns"]) > 0 and int(row["Rows"]) > 0]
    return [("tallest", min(candidates, key=lambda pair: pair[0])[1]),
            ("near_square", min(candidates, key=lambda pair: abs(pair[0] - 1))[1]),
            ("widest", max(candidates, key=lambda pair: pair[0])[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run, output = args.m01_run_dir.resolve(), args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("Audit must not write to frozen M01")
    config = load_config(args.config)
    if config != load_config(run / "config.yaml"):
        parser.error("Config differs from frozen M01 runtime config")
    source = archive_path(config)
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    montage = Image.new("RGB", (2 * 430, 6 * 460), "#eeeeee")
    draw = ImageDraw.Draw(montage)
    with zipfile.ZipFile(source) as archive:
        for split_index, split in enumerate(("train", "val")):
            rows = read_split_rows(source, split)
            for position, (kind, row) in enumerate(selected_rows(rows)):
                member = image_member(row, split)
                original_bytes = archive.read(member)
                with Image.open(io.BytesIO(original_bytes)) as opened:
                    rgb = opened.convert("RGB")
                width, height = rgb.size
                ratio = width / height
                processed = ImageOps.pad(rgb, (384, 384), method=Image.Resampling.BILINEAR,
                                         color=(0, 0, 0))
                tensor = preprocess_image(rgb)
                expected = TF.normalize(TF.to_tensor(processed), IMAGENET_MEAN, IMAGENET_STD)
                if not torch.equal(tensor, expected) or not torch.isfinite(tensor).all():
                    raise ValueError(f"Frozen preprocessing mismatch or nonfinite tensor: {member}")
                stem = f"{split}_{kind}_{row['Dicom_id']}"
                (output / f"{stem}_original.jpg").write_bytes(original_bytes)
                processed.save(output / f"{stem}_processed.png")
                row_y = (split_index * 3 + position) * 460
                thumbnail = rgb.copy()
                thumbnail.thumbnail((384, 384), Image.Resampling.BILINEAR)
                montage.paste(thumbnail, (23 + (384 - thumbnail.width) // 2, row_y + 35))
                montage.paste(processed, (453, row_y + 35))
                label = f"{split} {kind}: {width}x{height}, ratio={ratio:.3f}"
                draw.text((18, row_y + 8), label, fill="black")
                draw.text((450, row_y + 8), "M01 actual ImageOps.pad 384x384", fill="black")
                # ImageOps.pad fits the full image, then pads the short side.
                padding_fraction = 1.0 - min(width, height) / max(width, height)
                examples.append({"split": split, "selection": kind, "Subject_id": row["Subject_id"],
                                 "Study_id": row["Study_id"], "Dicom_id": row["Dicom_id"],
                                 "width": width, "height": height, "aspect_ratio": ratio,
                                 "fraction_of_square_side_padded_approx": padding_fraction,
                                 "original_file": f"{stem}_original.jpg",
                                 "processed_file": f"{stem}_processed.png",
                                 "pixel_range_after_normalize": [float(tensor.min()), float(tensor.max())]})
    montage.save(output / "comparison_montage.jpg", quality=92)
    result = {"source_archive": str(source),
              "actual_order": ["decode JPEG", "convert to RGB", "ImageOps.pad(384x384, BILINEAR, black)",
                               "to_tensor / 255", "ImageNet mean/std normalize"],
              "geometry": "ImageOps.pad preserves scale ratio by fitting the entire image inside 384x384 and padding the remaining dimension with black. It does not crop source pixels.",
              "nonuniform_resize": False, "anatomical_cropping_by_preprocessing": False,
              "example_count": len(examples), "examples": examples,
              "m03_candidate": "Not a distinct geometry change: complete aspect-ratio resize plus padding is already M01's actual behavior. Any M03 preprocessing proposal needs a separately specified change.",
              "m02_change": "none; M02 retains frozen M01 deterministic preprocessing"}
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# M01 preprocessing audit", "",
             "Actual sequence: JPEG decode → RGB conversion → `ImageOps.pad(384×384, BILINEAR, black)` → tensor scaled to [0,1] → ImageNet mean/std normalization.",
             "`ImageOps.pad` preserves aspect ratio, fits the complete image inside the square, and adds black padding on the short dimension. It does not apply a crop or nonuniform stretch.",
             "The montage and six original/processed pairs show full image content with padding; this preprocessing does not remove lung apex or costophrenic angle pixels.",
             "M02 keeps this preprocessing unchanged. Full-image fit resize plus padding is already M01's behavior, so it is not a distinct M03 candidate.",
             "", "| Split | Selection | Original size | Aspect ratio | Approx. square side padded |",
             "|---|---|---:|---:|---:|"]
    for item in examples:
        lines.append(f"| {item['split']} | {item['selection']} | {item['width']}×{item['height']} | "
                     f"{item['aspect_ratio']:.3f} | {item['fraction_of_square_side_padded_approx']:.1%} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"geometry": result["geometry"], "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
