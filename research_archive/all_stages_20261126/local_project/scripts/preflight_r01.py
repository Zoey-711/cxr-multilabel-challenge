"""Audit M02→R01 differences and run one real 512px training update on CUDA."""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.engine.losses import per_label_loss
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.config import load_config


M02 = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def flatten(value: dict, prefix: str = "") -> dict:
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        out.update(flatten(item, name) if isinstance(item, dict) else {name: item})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int, choices=(4, 8))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline, candidate = load_config(M02 / "config.yaml"), load_config(args.config)
    old, new = flatten(baseline), flatten(candidate)
    changes = {key: {"M02": old.get(key), "R01": new.get(key)} for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)}
    allowed = {"experiment.name", "data.input_size", "training.batch_size", "training.gradient_accumulation_steps"}
    if set(changes) - allowed or new["data.input_size"] != 512 or new["training.batch_size"] != args.batch_size:
        raise ValueError(f"Unexpected M02/R01 config differences: {changes}")
    if args.batch_size == 8 and set(changes) != {"experiment.name", "data.input_size"}:
        raise ValueError("R01 8x2 must only differ by resolution")
    if args.batch_size == 4 and set(changes) != allowed:
        raise ValueError("R01 4x4 must record both batch changes")
    accumulation = int(candidate["training"]["gradient_accumulation_steps"])
    if args.batch_size * accumulation != 16 or candidate["data"].get("train_sampling"):
        raise ValueError("Effective batch or image-shuffle sampling differs")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    if Path("outputs/experiments", candidate["experiment"]["name"]).exists():
        raise FileExistsError("R01 output directory exists")
    _, _, filename = imagenet_weight_info("convnext_tiny")
    weight = Path(".wheels/torch_cache/hub/checkpoints") / filename
    frozen_source = json.loads((M02 / "pretrained_source.json").read_text(encoding="utf-8"))
    if sha256(weight) != frozen_source["sha256"]:
        raise ValueError("Pretrained weight differs from M02")
    data = archive_path(candidate)
    train_rows, val_rows = read_split_rows(data, "train"), read_split_rows(data, "val")
    expected_groups = math.ceil(math.ceil(len(train_rows) / args.batch_size) / accumulation)
    if len(train_rows) != 30344 or len(val_rows) != 1422 or expected_groups != 1897:
        raise ValueError("Train/Val coverage or optimizer-group count differs")
    seed = int(candidate["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = make_dataloader(data, "train", rows=train_rows[:16], batch_size=args.batch_size,
                             num_workers=0, seed=seed, shuffle=False, unlisted_policy="negative",
                             augmentation=candidate["data"]["train_augmentation"], input_size=512,
                             preprocessing=candidate["data"].get("preprocessing", "black_pad_v1"))
    samples = []
    iterator = iter(loader)
    for _ in range(accumulation):
        samples.append(next(iterator))
    loader.dataset.close()
    if sum(len(batch["image"]) for batch in samples) != 16 or any(batch["image"].shape[2:] != (512, 512) for batch in samples):
        raise ValueError("Preflight image coverage or size differs")
    free_before, total = torch.cuda.mem_get_info()
    os.environ["TORCH_HOME"] = str(Path(".wheels/torch_cache").resolve())
    model = create_model("convnext_tiny", pretrained=True).to("cuda").train()
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": float(candidate["training"]["backbone_lr"])},
        {"params": model.classifier.parameters(), "lr": float(candidate["training"]["classifier_lr"])},
    ], weight_decay=float(candidate["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    valid_count = sum(int(batch["valid_mask"].sum()) for batch in samples)
    losses = []
    for batch in samples:
        images = batch["image"].to("cuda")
        targets = batch["target"].to("cuda")
        mask = batch["valid_mask"].to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            per_label = per_label_loss(logits, targets, candidate["training"])
            loss = (per_label * mask).sum() / valid_count
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite R01 preflight loss")
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()))
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(candidate["training"]["grad_clip_norm"]))
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError("Nonfinite R01 gradient norm")
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before:
        raise FloatingPointError("AMP skipped the first R01 optimizer step")
    scheduler.step()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free_after = torch.cuda.mem_get_info()[0]
    report = {"status": "passed", "config_differences": changes,
              "micro_batch": args.batch_size, "accumulation": accumulation, "effective_batch": 16,
              "train_images_per_epoch": len(train_rows), "val_images": len(val_rows),
              "micro_batches_per_epoch": math.ceil(len(train_rows) / args.batch_size),
              "optimizer_groups_per_epoch": expected_groups, "tested_image_count": 16,
              "test_group_loss": sum(losses), "unclipped_gradient_norm": float(grad_norm),
              "optimizer_steps": 1, "scheduler_steps": 1, "amp_skipped_steps": 0,
              "gpu": torch.cuda.get_device_name(0), "total_gpu_mib": round(total / 2**20),
              "free_gpu_before_mib": round(free_before / 2**20),
              "peak_allocated_mib": round(peak_allocated / 2**20),
              "peak_reserved_mib": round(peak_reserved / 2**20),
              "free_gpu_after_mib": round(free_after / 2**20),
              "estimated_free_at_peak_mib": round((free_before - peak_reserved) / 2**20),
              "pretrained_sha256": sha256(weight), "config_sha256": sha256(args.config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
