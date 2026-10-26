"""Audit T01 versus frozen M02 and run a real FP16 accumulation step."""

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.engine.losses import per_label_loss
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.config import load_config


EXPECTED_SHA = "704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    control = load_config(root / "outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/config.yaml")
    config = load_config(args.config)
    difference = [(section, key, control[section].get(key), config[section].get(key))
                  for section in control for key in control[section]
                  if control[section].get(key) != config[section].get(key)]
    if difference != [("experiment", "name", control["experiment"]["name"], config["experiment"]["name"]),
                      ("model", "name", "convnext_tiny", "swin_t")]:
        raise ValueError(f"T01/M02 config differs beyond experiment and backbone: {difference}")
    if config["training"]["batch_size"] != 8 or config["training"]["gradient_accumulation_steps"] != 2:
        raise ValueError("This preflight expects the preferred 8x2 configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    source, url, filename = imagenet_weight_info("swin_t")
    weights = root / ".wheels/torch_cache/hub/checkpoints" / filename
    if not weights.is_file() or sha(weights) != EXPECTED_SHA:
        raise ValueError("Official Swin-Tiny weight SHA mismatch")
    os.environ["TORCH_HOME"] = str(root / ".wheels/torch_cache")
    torch.manual_seed(config["experiment"]["seed"])
    torch.cuda.manual_seed_all(config["experiment"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    archive = archive_path(config)
    rows = read_split_rows(archive, "train")
    loader = make_dataloader(archive, "train", rows=rows, batch_size=8, num_workers=0,
                             shuffle=True, seed=42,
                             augmentation=config["data"]["train_augmentation"],
                             preprocessing=config["data"].get("preprocessing", "black_pad_v1"), input_size=384)
    model = create_model("swin_t", pretrained=True).cuda().train()
    if model.head.out_features != 10:
        raise ValueError("Swin-Tiny head is not ten logits")
    head = list(model.head.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": config["training"]["backbone_lr"]},
                                   {"params": head, "lr": config["training"]["classifier_lr"]}],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    groups = list(itertools.islice(iter(loader), 2))
    if len(groups) != 2:
        raise ValueError("Missing training microbatches")
    valid_count = sum(int(batch["valid_mask"].sum()) for batch in groups)
    before = model.head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    losses = []
    for batch in groups:
        x = batch["image"].cuda(); y = batch["target"].cuda(); valid = batch["valid_mask"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = (per_label_loss(logits, y, config["training"]) * valid).sum() / valid_count
        if logits.shape != (8, 10) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Invalid Swin forward or loss")
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()))
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("Nonfinite Swin gradients")
    scale_before = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    if scaler.get_scale() < scale_before or torch.equal(before, model.head.weight):
        raise ValueError("AMP skipped the step or head failed to update")
    report = {"control": "frozen M02 runtime config", "config_difference": difference,
              "model": "torchvision Swin_T_Weights.IMAGENET1K_V1", "weight_url": url,
              "weight_sha256": sha(weights), "weight_bytes": weights.stat().st_size,
              "train_images": len(rows), "micro_batch": 8, "accumulation": 2, "effective_batch": 16,
              "total_train_optimizer_groups_per_epoch": math.ceil(math.ceil(len(rows) / 8) / 2),
              "preflight_logits_shape": [8, 10], "loss_microbatches": losses,
              "unclipped_gradient_norm": float(norm), "amp_step_skipped": False,
              "elapsed_seconds_for_two_microbatches": time.perf_counter() - start,
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "gpu": torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
