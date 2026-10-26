"""Short diagnostic benchmark of frozen E08 inference on the local GPU only.

This is not an official-hardware compliance test. No training or artifact mutation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from cxr.data.archive import image_member, read_split_rows
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, resize_pad_image
from cxr.utils.checkpoint import load_model_state


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUB = ROOT / "outputs/submissions/E08_EMA_20261020/manifest.json"
RUN = ROOT / "outputs/experiments"
WARMUP, REPEATS = 5, 16
WEIGHTS = (1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 2)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict:
    return {"mean_ms": float(statistics.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def main() -> None:
    archive = Path(os.environ["CXR_ARCHIVE"])
    manifest = json.loads(SUB.read_text(encoding="utf-8"))
    names = ("B01", "B02", "B03", "M02", "N01_EMA")
    paths = []
    for name in names:
        entry = manifest["models"][name]
        path = RUN / entry["run_id"] / entry["checkpoint"]
        if sha(path) != entry["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint SHA mismatch: {name}")
        paths.append(path)
    rows = read_split_rows(archive, "val")
    member = image_member(rows[0], "val")
    models = [load_model_state(path, device="cuda") for path in paths]
    device = torch.cuda.get_device_name(0)
    samples = {k: [] for k in ("zip_read", "decode_rgb", "resize_pad", "normalize",
                              "host_to_device", *(f"forward_{name}" for name in names),
                              "sigmoid_fusion_cpu_copy", "study_mean", "full_end_to_end")}

    with zipfile.ZipFile(archive) as zf, torch.inference_mode():
        for i in range(WARMUP + REPEATS):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            t = begin
            data = zf.read(member)
            samples["zip_read"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            with Image.open(io.BytesIO(data)) as opened:
                image = opened.convert("RGB")
                image.load()
            samples["decode_rgb"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
            samples["resize_pad"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            tensor = TF.normalize(TF.to_tensor(padded), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
            samples["normalize"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            gpu = tensor.to("cuda")
            torch.cuda.synchronize()
            samples["host_to_device"].append((time.perf_counter() - t) * 1000)
            predictions = []
            for name, model in zip(names, models):
                t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions.append(torch.sigmoid(model(gpu)))
                torch.cuda.synchronize()
                samples[f"forward_{name}"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            fused = sum(w * p for w, p in zip(WEIGHTS, predictions)).float().cpu().numpy()
            torch.cuda.synchronize()
            samples["sigmoid_fusion_cpu_copy"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            study = np.mean(fused, axis=0)
            samples["study_mean"].append((time.perf_counter() - t) * 1000)
            if study.shape != (10,) or not np.isfinite(study).all():
                raise ValueError("Invalid fused probability")
            samples["full_end_to_end"].append((time.perf_counter() - begin) * 1000)

    # Warmup runs are excluded from all summaries. Stage timings include per-stage
    # synchronization, so they are diagnostic and should not be summed as an E2E time.
    result = []
    for stage, values in samples.items():
        result.append({"stage": stage, **stats(values[WARMUP:]), "device": device,
                       "precision": "FP16 autocast", "batch_size": 1,
                       "warmup": WARMUP, "repeats": REPEATS,
                       "scope": "one fixed Val image from ZIP; models resident in GPU memory",
                       "official_compliance": "LATENCY_UNVERIFIED"})
    target = OUT / "latency_breakdown.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    (OUT / "latency_benchmark.json").write_text(json.dumps({"member": member, "rows": result,
        "checkpoint_sha256": {name: sha(path) for name, path in zip(names, paths)},
        "model_load_in_timing": False, "stage_timings_additive": False,
        "limitation": "Local 4060 only; official device and timing statistic unspecified"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({row["stage"]: row["mean_ms"] for row in result}, indent=2))


if __name__ == "__main__":
    main()
