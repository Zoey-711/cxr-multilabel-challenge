"""GPU-only ConvNeXt-Tiny readiness test; no optimizer step or formal training."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.checkpoint import load_model_state, save_model_state
from cxr.utils.config import load_config


def board_used_mib():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 and result.stdout.strip() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    config = load_config(args.config)
    if config["model"]["name"] != "convnext_tiny" or config["model"]["pretrained"] != "imagenet":
        parser.error("M01 requires pretrained ConvNeXt-Tiny")
    if not torch.cuda.is_available():
        parser.error("M01 preflight requires CUDA GPU")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Preflight output must be new: {output}")
    output.mkdir(parents=True)
    shutil.copy2(args.config, output / "config.yaml")
    project = Path(__file__).resolve().parents[1]
    os.environ["TORCH_HOME"] = str(project / ".wheels" / "torch_cache")
    source, url, filename = imagenet_weight_info("convnext_tiny")
    weight = project / ".wheels" / "torch_cache" / "hub" / "checkpoints" / filename
    if not weight.is_file():
        parser.error(f"Official ImageNet weight is not cached: {weight}")
    with weight.open("rb") as handle:
        weight_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    archive = archive_path(config)
    train_rows = read_split_rows(archive, "train")
    loader = make_dataloader(archive, "train", rows=train_rows, batch_size=8, num_workers=int(config["data"]["num_workers"]),
                             seed=int(config["experiment"]["seed"]), shuffle=False, unlisted_policy="negative")
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        loader.dataset.close()
    images = batch["image"]
    labels = batch["target"]
    mask = batch["valid_mask"]
    if images.shape != (8, 3, 384, 384) or labels.shape != (8, 10) or mask.shape != (8, 10):
        raise ValueError(f"Unexpected real-data batch shapes: {images.shape}, {labels.shape}, {mask.shape}")
    if not torch.isfinite(images).all() or not torch.isfinite(labels).all() or not bool(mask.all()):
        raise ValueError("DataLoader produced nonfinite data or invalid closed-world labels")
    model = create_model("convnext_tiny", num_classes=10, pretrained=True).to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dropout_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)]
    stochastic_depth_rates = [float(module.p) for module in model.modules()
                              if module.__class__.__name__ == "StochasticDepth"]
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 2**20
    baseline_board = board_used_mib()
    results = []
    for batch_size in (2, 4, 8):
        model.train()
        small_images = images[:batch_size].to("cuda", non_blocking=True)
        small_labels = labels[:batch_size].to("cuda", non_blocking=True)
        small_mask = mask[:batch_size].to("cuda", non_blocking=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        durations, losses = [], []
        try:
            for _ in range(args.repeats + 1):
                model.zero_grad(set_to_none=True)
                scaler = torch.amp.GradScaler("cuda")
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(small_images)
                    loss = (F.binary_cross_entropy_with_logits(logits, small_labels, reduction="none") * small_mask).sum() / small_mask.sum()
                if logits.shape != (batch_size, 10) or not torch.isfinite(loss):
                    raise ValueError("ConvNeXt logits shape or BCE loss is invalid")
                scaler.scale(loss).backward()
                nonfinite_gradients = sum(
                    not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters() if parameter.grad is not None
                )
                if nonfinite_gradients:
                    raise ValueError(f"Nonfinite FP16 gradients in {nonfinite_gradients} tensors")
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
                losses.append(float(loss.detach()))
                del logits, loss, scaler
            result = {"batch_size": batch_size, "status": "passed", "amp_dtype": "float16",
                      "optimizer_steps": 0, "forward_backward_seconds_median": float(sorted(durations[1:])[len(durations[1:]) // 2]),
                      "loss_last": losses[-1], "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                      "board_used_mib_after": board_used_mib()}
        except torch.cuda.OutOfMemoryError as exc:
            result = {"batch_size": batch_size, "status": "out_of_memory", "error": str(exc).splitlines()[0],
                      "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                      "board_used_mib_after": board_used_mib()}
        finally:
            model.zero_grad(set_to_none=True)
            if "logits" in locals():
                del logits
            if "loss" in locals():
                del loss
            del small_images, small_labels, small_mask
            torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    model.eval()
    checkpoint = save_model_state(model, output / "structure_checkpoint.pt", model_name="convnext_tiny")
    loaded = load_model_state(checkpoint, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        probe = images[:1].to("cuda")
        original = model(probe)
        restored = loaded(probe)
    checkpoint_match = bool(torch.allclose(original, restored, atol=0, rtol=0))
    if not checkpoint_match:
        raise ValueError("ConvNeXt checkpoint reload changed logits")
    del loaded, probe, original, restored
    passed = [item for item in results if item["status"] == "passed"]
    if not passed:
        raise RuntimeError("All tested M01 microbatches ran out of GPU memory")
    recommended = max((item for item in passed if item["peak_reserved_mib"] <= gpu_total - 1600),
                      key=lambda item: item["batch_size"], default=passed[0])
    microbatch = recommended["batch_size"]
    accumulation = math.ceil(16 / microbatch)
    result = {"preflight_only_no_optimizer_step": True, "python": platform.python_version(),
              "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
              "seed": int(config["experiment"]["seed"]),
              "gpu": torch.cuda.get_device_name(0), "gpu_total_mib": round(gpu_total),
              "baseline_board_used_mib": baseline_board, "input_shape": list(images.shape),
              "target_shape": list(labels.shape), "subject_ids": batch["subject_id"],
              "study_ids": batch["study_id"], "image_ids": batch["image_id"],
              "source": source, "url": url, "weight_file": str(weight), "pretrained_sha256": weight_sha,
              "pretrained_weight_bytes": weight.stat().st_size, "model_parameter_count": parameter_count,
              "classifier_dropout_modules": dropout_modules, "nonfinite_gradient_tensors": 0,
              "stochastic_depth_blocks": len(stochastic_depth_rates),
              "maximum_stochastic_depth_rate": max(stochastic_depth_rates, default=0.0),
              "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_reload_exact": checkpoint_match,
              "batch_tests": results, "recommended_microbatch": microbatch,
              "recommended_accumulation_steps": accumulation,
              "recommended_effective_batch": microbatch * accumulation,
              "selection_rule": "Largest tested batch whose peak reserved GPU memory leaves at least 1600 MiB headroom; otherwise smallest passing batch",
              "note": "Timings cover GPU forward/backward on one reused real batch; no optimizer step, full epoch, or T4 latency test."}
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model_parameter_count", "checkpoint_bytes",
                                                  "recommended_microbatch", "recommended_accumulation_steps",
                                                  "recommended_effective_batch")}, indent=2))


if __name__ == "__main__":
    main()
