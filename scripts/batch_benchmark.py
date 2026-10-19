"""Short non-training cloud benchmark: ZIP loading and cached GPU optimizer steps."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


LOCAL_MINUTES_PER_EPOCH = {"densenet121": 7.3, "convnext_tiny": 22.3, "swin_t": 22.1}
YEN_PER_HOUR = 0.0  # Friend pricing unknown; do not invent monetary cost.


def sample_gpu_util(stop: threading.Event, values: list[int]) -> None:
    while not stop.wait(0.5):
        result = subprocess.run(["nvidia-smi", "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=False)
        try:
            values.append(int(result.stdout.strip().splitlines()[0]))
        except (IndexError, ValueError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-steps", default=3, type=int)
    parser.add_argument("--measure-steps", default=20, type=int)
    parser.add_argument("--read-images", default=128, type=int)
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.measure_steps < 2 or args.read_images < 16:
        parser.error("Benchmark requires positive warmup and enough measurements")
    # Inherit exact allocated GPU token from batch_preflight; never escape Slurm allocation.
    import torch
    from cxr.data.archive import archive_path, read_split_rows
    from cxr.data.dataset import make_dataloader
    from cxr.data.patient_folds import select_fold_rows
    from cxr.models.factory import create_model
    from cxr.utils.cloud_paths import pretrained_home
    from cxr.utils.config import load_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    config = load_config(args.config)
    archive = archive_path(config)
    rows = select_fold_rows(read_split_rows(archive, "train"), config["data"]["train_subset"], role="train")
    loader = make_dataloader(archive, "train", rows=rows[:args.read_images],
                             batch_size=int(config["training"]["batch_size"]),
                             num_workers=int(config["data"]["num_workers"]),
                             preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                             input_size=int(config["data"]["input_size"]), persistent_workers=False)
    started = time.perf_counter()
    read_count = 0
    cached = None
    for batch in loader:
        read_count += len(batch["image"])
        if cached is None:
            cached = (batch["image"].cuda(non_blocking=True), batch["target"].cuda(non_blocking=True))
    read_seconds = time.perf_counter() - started
    loader.dataset.close()
    os.environ["TORCH_HOME"] = str(pretrained_home(Path(__file__).resolve().parents[1]))
    model = create_model(config["model"]["name"], pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["backbone_lr"]))
    scaler = torch.amp.GradScaler("cuda")
    image, target = cached
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    def one_step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(image)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target) / accumulation
            if not torch.isfinite(loss):
                raise ValueError("Non-finite benchmark loss")
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
    for _ in range(args.warmup_steps):
        one_step()
    torch.cuda.reset_peak_memory_stats()
    utilization = []
    stop = threading.Event()
    thread = threading.Thread(target=sample_gpu_util, args=(stop, utilization), daemon=True)
    thread.start()
    started = time.perf_counter()
    for _ in range(args.measure_steps):
        one_step()
    compute_seconds = time.perf_counter() - started
    stop.set()
    thread.join(timeout=1)
    compute_images_s = args.measure_steps * len(image) * accumulation / compute_seconds
    read_images_s = read_count / read_seconds
    # Conservative two-stage estimate: overlap is unknown, so add read and compute times.
    epoch_images = len(rows)
    epoch_seconds = epoch_images / read_images_s + epoch_images / compute_images_s
    report = {"model": config["model"]["name"], "fold": config["data"]["train_subset"]["fold"],
              "gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
              "read_images": read_count, "read_seconds": read_seconds, "read_images_s": read_images_s,
              "gpu_micro_batch": len(image), "gradient_accumulation": accumulation,
              "effective_batch": len(image) * accumulation, "warmup_steps": args.warmup_steps,
              "measure_steps": args.measure_steps, "gpu_optimizer_step_seconds": compute_seconds / args.measure_steps,
              "gpu_compute_images_s": compute_images_s,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_utilization_samples_percent": utilization,
              "gpu_utilization_mean_percent": sum(utilization) / len(utilization) if utilization else None,
              "estimated_epoch_minutes": epoch_seconds / 60,
              "estimated_10_epoch_hours": epoch_seconds * 10 / 3600,
              "estimated_30_epoch_hours": epoch_seconds * 30 / 3600,
              "estimated_10_epoch_yuan": epoch_seconds * 10 / 3600 * YEN_PER_HOUR,
              "estimated_30_epoch_yuan": epoch_seconds * 30 / 3600 * YEN_PER_HOUR,
              "local_full_train_historical_minutes_per_epoch": LOCAL_MINUTES_PER_EPOCH.get(config["model"]["name"]),
              "estimate_note": "Short cached-GPU plus ZIP-read estimate; validation, startup, checkpoints and early stop vary."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Benchmark output already exists: {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
