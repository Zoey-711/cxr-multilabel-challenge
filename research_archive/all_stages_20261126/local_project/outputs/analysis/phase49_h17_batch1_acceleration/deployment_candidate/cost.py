"""T4 batch=1 cost check using deployment_candidate.predict's exact model/forward."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from predict import load_model, predict_image, sha256


def images(directory: Path, count: int):
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != count:
        raise ValueError(f"Expected {count} images in {directory}; found {len(paths)}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--measure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-sample", action="store_true",
                        help="Assert the organizer supplied this fixed 5+100 image set")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    warmup, measured = images(args.warmup_dir, 5), images(args.measure_dir, 100)
    if {p.resolve() for p in warmup} & {p.resolve() for p in measured}:
        raise ValueError("Warmup and timed images overlap")
    model = load_model()
    for path in warmup:
        predict_image(path, model)
        torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True)
    elapsed = {p.name: [] for p in measured}
    output = args.output_dir / "cost_predictions.csv"
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "image"] + [f"Prob_{j}" for j in range(1, 11)])
        for round_id in range(1, 4):
            for path in measured:
                torch.cuda.synchronize()
                start = time.perf_counter()
                probs = predict_image(path, model)
                writer.writerow([round_id, path.name, *[format(float(p), ".17g") for p in probs]])
                stream.flush()
                torch.cuda.synchronize()
                elapsed[path.name].append((time.perf_counter() - start) * 1000)
    per_image = [statistics.mean(v) for v in elapsed.values()]
    result = dict(gpu=torch.cuda.get_device_name(), batch_size=1, warmup=5,
                  measured_images=100, rounds=3, scope="file read through CSV write; synchronized CUDA",
                  mean_ms=float(np.mean(per_image)), p50_ms=float(np.percentile(per_image, 50)),
                  p95_ms=float(np.percentile(per_image, 95)), max_ms=float(np.max(per_image)),
                  model_sha256=sha256(Path(__file__).parent / "h17_raw_fp32.onnx"),
                  predictions_sha256=sha256(output),
                  official_sample_user_asserted=bool(args.official_sample),
                  target_status=("T4_MEASURED_PREDICTION_CHECK_PENDING"
                                 if args.official_sample and "T4" in torch.cuda.get_device_name()
                                 else "T4_UNVERIFIED"))
    (args.output_dir / "cost_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
