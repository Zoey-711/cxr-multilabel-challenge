"""Measure local GPU end-to-end validation-image latency for R01 and M02."""

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


M02 = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated")


def benchmark(run: Path, config: dict, rows: list[dict], archive: Path, *, warmup: int, images: int) -> dict:
    model = load_model_state(run / "best_model.pt", device="cuda").eval()
    dataset = CXRArchiveDataset(archive, "val", rows=rows[:warmup+images], unlisted_policy="negative",
                                preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                input_size=int(config["data"]["input_size"]))
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    try:
        with torch.inference_mode():
            for i in range(warmup + images):
                torch.cuda.synchronize()
                start = time.perf_counter()
                tensor = dataset[i]["image"].unsqueeze(0).to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    prob = torch.sigmoid(model(tensor)).float().cpu()
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                if not torch.isfinite(prob).all() or prob.shape != (1, 10):
                    raise ValueError("Invalid benchmark prediction")
                if i >= warmup:
                    latencies.append(ms)
    finally:
        dataset.close()
    result = {"checkpoint_sha256": sha256(run / "best_model.pt"), "input_size": config["data"]["input_size"],
              "warmup_images": warmup, "measured_images": images,
              "latency_ms_mean": float(np.mean(latencies)), "latency_ms_median": float(np.median(latencies)),
              "latency_ms_p95": float(np.quantile(latencies, .95)), "latency_ms_max": float(np.max(latencies)),
              "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--images", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    r01 = load_config(args.config)
    if r01 != load_config(args.run_dir / "config.yaml") or r01["data"]["input_size"] != 512:
        raise ValueError("R01 runtime config mismatch")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("R01 training incomplete")
    if not torch.cuda.is_available() or args.warmup < 1 or args.images < 10:
        raise ValueError("CUDA and positive warmup/sample sizes are required")
    m02 = load_config(M02 / "config.yaml")
    frozen = json.loads((M02 / "frozen_manifest.json").read_text(encoding="utf-8"))
    if sha256(M02 / "best_model.pt") != frozen["best_model.pt"]["sha256"]:
        raise ValueError("Frozen M02 checkpoint hash mismatch")
    archive = archive_path(r01)
    if archive_path(m02) != archive:
        raise ValueError("R01 and M02 datasets differ")
    rows = read_split_rows(archive, "val")
    if len(rows) != 1422:
        raise ValueError("Unexpected validation image count")
    results = {"M02": benchmark(M02, m02, rows, archive, warmup=args.warmup, images=args.images),
               "R01": benchmark(args.run_dir, r01, rows, archive, warmup=args.warmup, images=args.images)}
    report = {"host_gpu": torch.cuda.get_device_name(0),
              "scope": "local RTX 4060 Laptop single-image ZIP read+decode+resize/pad+transfer+FP16 model+sigmoid+CPU probability; not official T4 certification",
              "same_validation_images": True, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
