"""Measure local single-image ZIP-to-probability latency for N01 and fixed E08."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.inference.versioned_submission import sha256
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


E03_RUNS = (
    Path("outputs/experiments/b01_densenet121_384_20261016"),
    Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
)


def timed(models, dataset, *, weights: list[float], warmup: int, count: int) -> dict:
    if len(weights) != len(models) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("Benchmark fusion weights must align and sum to one")
    durations = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(warmup + count):
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = dataset[i]["image"].unsqueeze(0).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                predictions = [torch.sigmoid(model(image)) for model in models]
            mean = sum(weight * prediction for weight, prediction in zip(weights, predictions)).float().cpu()
            torch.cuda.synchronize()
            if mean.shape != (1, 10) or not torch.isfinite(mean).all():
                raise ValueError("Invalid benchmark prediction")
            if i >= warmup:
                durations.append((time.perf_counter() - start) * 1000)
    return {"warmup_images": warmup, "measured_images": count,
            "latency_ms_mean": float(np.mean(durations)),
            "latency_ms_median": float(np.median(durations)),
            "latency_ms_p95": float(np.quantile(durations, 0.95)),
            "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--images", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    if config != load_config(args.run_dir / "config.yaml") or config["model"]["name"] != "efficientnet_v2_s":
        raise ValueError("N01 config differs from runtime copy")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("N01 training incomplete")
    archive = archive_path(config)
    for run in E03_RUNS:
        previous = load_config(run / "config.yaml")
        if archive_path(previous) != archive or previous["data"]["input_size"] != 384 or \
                previous["data"].get("preprocessing", "black_pad_v1") != config["data"].get("preprocessing", "black_pad_v1"):
            raise ValueError(f"E03 model has different deterministic preprocessing: {run}")
    original_manifest = json.loads(Path("outputs/submissions/E03_20261017/manifest.json").read_text(encoding="utf-8"))
    for run, name in zip(E03_RUNS, original_manifest["model_versions"]):
        if sha256(run / "best_model.pt") != original_manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Frozen E03 checkpoint hash mismatch: {name}")
    rows = read_split_rows(archive, "val")
    dataset = CXRArchiveDataset(archive, "val", rows=rows[:args.warmup + args.images],
                                preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                input_size=384)
    raw = load_model_state(args.run_dir / "best_model.pt", device="cuda")
    ema = load_model_state(args.run_dir / "best_ema_model.pt", device="cuda")
    try:
        results = {"N01_raw": timed([raw], dataset, weights=[1.0], warmup=args.warmup, count=args.images),
                   "N01_EMA": timed([ema], dataset, weights=[1.0], warmup=args.warmup, count=args.images)}
        e03_models = [load_model_state(run / "best_model.pt", device="cuda") for run in E03_RUNS]
        results["E08_all_five_models"] = timed([*e03_models, raw], dataset,
                                                weights=[1/12, 1/12, 1/12, 1/4, 1/2],
                                                warmup=args.warmup, count=args.images)
    finally:
        dataset.close()
    total_bytes = sum((run / "best_model.pt").stat().st_size for run in E03_RUNS) + \
        (args.run_dir / "best_model.pt").stat().st_size
    if total_bytes > 500_000_000:
        raise ValueError("E08 inference weights exceed 500 MB")
    report = {"gpu": torch.cuda.get_device_name(0), "checkpoint_total_bytes_e08": total_bytes,
              "checkpoint_limit_bytes": 500_000_000,
              "scope": "Local RTX 4060 Laptop; ZIP read, decode, deterministic resize/pad, GPU transfer, FP16 forward, sigmoid and CPU copy. Official T4 latency remains unmeasured.",
              "results": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
