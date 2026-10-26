"""Measure end-to-end single-image latency on a CUDA GPU."""

import argparse
import io
import json
import statistics
import time
import zipfile
from pathlib import Path

from cxr.inference.predict import infer_image, make_transform
from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.utils.checkpoint import check_model_file_size, load_model_state
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="Standalone image; omit to benchmark the configured ZIP")
    parser.add_argument("--archive-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be nonnegative and --runs must be positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the inference benchmark")
    config = load_config(args.config)
    size_bytes = check_model_file_size(args.checkpoint)
    model = load_model_state(args.checkpoint, device="cuda")
    transform = make_transform(config["data"]["input_size"])

    if args.image:
        if not args.image.is_file():
            parser.error(f"Image not found: {args.image}")
        def infer_one():
            return infer_image(model, args.image, transform=transform, device="cuda")
        source = str(args.image)
    else:
        from PIL import Image

        archive = zipfile.ZipFile(archive_path(config))
        member = image_member(read_split_rows(archive.filename, args.archive_split)[0], args.archive_split)
        source = f"{archive.filename}!{member}"

        def infer_one():
            with Image.open(io.BytesIO(archive.read(member))) as image:
                tensor = transform(image).unsqueeze(0).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return torch.sigmoid(model(tensor)).squeeze(0).float().cpu().tolist()

    for _ in range(args.warmup):
        infer_one()
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(args.runs):
        start = time.perf_counter()
        infer_one()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)

    ordered = sorted(samples_ms)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "source": source,
        "runs": args.runs,
        "image_size": config["data"]["input_size"],
        "model_bytes": size_bytes,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": max(samples_ms),
        "passes_100ms_p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] <= 100,
        "passes_500mb": size_bytes <= 500_000_000,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
