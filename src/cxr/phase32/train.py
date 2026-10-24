# Isolated batch snapshot of scripts/train.py SHA256 13d9849207b712b284b38e4d16a4cbe6d2419be7d3f244080c59b94d5baa2611
"""Train a configured ImageNet classifier and validate at Study level."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cxr.data.archive import archive_path, read_split_rows
from cxr.phase32.dataset import make_dataloader, study_targets_from_rows
from cxr.data.internal_split import select_internal_train_rows
from cxr.data.patient_folds import select_fold_rows
from cxr.data.sampling import StudyBalancedSampler
from cxr.engine.early_stopping import MODE, MONITOR, EarlyStopTracker, default_patience
from cxr.phase32.methods import loss_elements as per_label_loss, mixup_batch, study_positive_weights
from cxr.engine.metrics import study_multilabel_metrics
from cxr.inference.aggregate import ImagePrediction
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.checkpoint import save_model_state
from cxr.utils.cloud_paths import output_root, pretrained_home
from cxr.phase32.config import load_config
from cxr.utils.training_resume import capture_rng_state, restore_rng_state
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


ACTIVE_RUN_DIR = None


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, targets, *, threshold: float) -> dict:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                probabilities = torch.sigmoid(model(images))
            probabilities = probabilities.float().cpu().tolist()
            for index, vector in enumerate(probabilities):
                predictions.append(ImagePrediction(
                    batch["subject_id"][index], batch["study_id"][index], batch["image_id"][index],
                    tuple(vector),
                ))
    return study_multilabel_metrics(predictions, targets, threshold=threshold)


def main() -> None:
    global ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", help="Optional output directory name below outputs/experiments")
    parser.add_argument("--resume", action="store_true", help="Resume last_training.pt in the selected run")
    parser.add_argument("--preflight-steps", type=int, default=0,
                        help="Run short training; G/H also round-trip a probe checkpoint and infer Val/Test")
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    model_name = config["model"]["name"]
    if model_name not in ("densenet121", "convnext_tiny", "efficientnet_v2_s", "swin_t", "convnext_small", "efficientnet_v2_m") or config["model"]["pretrained"] != "imagenet":
        parser.error("Training requires a supported ImageNet-pretrained classifier")
    if config["validation"].get("primary_metric") != "macro_auc" or config["validation"].get("metric_mode", MODE) != MODE:
        parser.error("Early stopping requires Study-level Macro-AUC in max mode")
    tracker = EarlyStopTracker(
        patience=config["training"].get("early_stopping_patience", default_patience(model_name)),
        min_delta=float(config["training"].get("early_stopping_min_delta", 0.0)),
    )
    ema_settings = config["training"].get("ema")
    if ema_settings is not None:
        if (model_name not in ("convnext_tiny", "efficientnet_v2_s") or ema_settings.get("enabled") is not True or
                float(ema_settings.get("decay", 0)) != 0.999 or
                ema_settings.get("update_after_step") != 1 or
                ema_settings.get("use_buffers") is not True):
            parser.error("Batch EMA fixed to declared ConvNeXt-Tiny / N01 EfficientNetV2-S settings")
    if args.resume and (not args.run_id or args.preflight_steps):
        parser.error("--resume requires --run-id and cannot be combined with --preflight-steps")
    if args.preflight_steps < 0:
        parser.error("--preflight-steps must be nonnegative")

    name = args.run_id or f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        parser.error("Run ID must contain only letters, digits, underscore, dot, or dash")
    run_dir = output_root() / "experiments" / name
    allowed_preexisting = {"train_stdout.log", "train_stderr.log"}
    if run_dir.exists() and not args.resume and any(item.name not in allowed_preexisting for item in run_dir.iterdir()):
        parser.error(f"Experiment directory already exists: {run_dir}")
    if args.resume:
        if load_config(run_dir / "config.yaml") != config:
            parser.error("Resume config differs from the saved runtime config")
        if json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("state") == "complete":
            parser.error("Completed experiments are frozen and cannot be resumed")
    run_dir.mkdir(parents=True, exist_ok=True)
    ACTIVE_RUN_DIR = run_dir
    shutil.copy2(args.config, run_dir / "config.yaml") if not args.resume else None
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="ascii")
    write_json(run_dir / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
    })
    project_root = Path(__file__).resolve().parents[3]
    weight_home = pretrained_home(project_root)
    os.environ["TORCH_HOME"] = str(weight_home)
    weight_source, weight_url, weight_name = imagenet_weight_info(model_name)
    weight_file = weight_home / "hub" / "checkpoints" / weight_name
    if not weight_file.is_file():
        parser.error(f"Pretrained weight is missing: {weight_file}")
    with weight_file.open("rb") as handle:
        weight_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    expected_weight_sha256 = config["model"].get("pretrained_sha256")
    if not expected_weight_sha256:
        parser.error("Pretrained full SHA-256 must be reviewed and sealed before any GPU training")
    if expected_weight_sha256 is not None and weight_sha256 != expected_weight_sha256:
        parser.error(f"Pretrained ImageNet weight hash mismatch: {weight_file}")
    provenance = {"source": weight_source, "url": weight_url,
                  "sha256": weight_sha256,
                  "size_bytes": weight_file.stat().st_size,
                  "external_labeled_data_used_for_additional_training": False}
    write_json(run_dir / "pretrained_source.json", provenance)
    data_path = archive_path(config)
    train_rows, val_rows = read_split_rows(data_path, "train"), read_split_rows(data_path, "val")
    subset = config["data"].get("train_subset")
    if subset is not None:
        train_rows = (select_fold_rows(train_rows, subset, role="train")
                      if subset.get("kind") == "patient_fold_v1" else
                      select_internal_train_rows(train_rows, subset))
        val_patients = {row["Subject_id"] for row in val_rows}
        val_studies = {(row["Subject_id"], row["Study_id"]) for row in val_rows}
        if (len(val_rows), len(val_studies), len(val_patients)) != (1422, 859, 228):
            parser.error("Official validation split differs from the frozen 859-Study protocol")
        if {row["Subject_id"] for row in train_rows} & val_patients:
            parser.error("Official validation patients overlap internal Train")
    full_expected = config["data"].get("full_train_expected_counts")
    if full_expected is not None:
        observed = {"patients": len({r["Subject_id"] for r in train_rows}),
                    "studies": len({(r["Subject_id"], r["Study_id"]) for r in train_rows}),
                    "images": len(train_rows)}
        if subset is not None or observed != full_expected:
            parser.error(f"Full-Train coverage differs from frozen protocol: {observed}")
        if {r["Subject_id"] for r in train_rows} & {r["Subject_id"] for r in val_rows}:
            parser.error("Official Val patients overlap full Train")
    if config['training']['loss']=='phase32_weighted_bce':
        config['training']['_resolved_pos_weight']=study_positive_weights(train_rows)
        write_json(run_dir/'class_weights.json',{'weights':config['training']['_resolved_pos_weight'],
            'source':'four training folds only; unique Study supports; sqrt negative/positive clipped 1..5'})
    policy = config["data"]["label_encoding"]["unlisted_policy"]
    if policy != "negative":
        parser.error("Study-level Macro-AUC baseline requires complete competition targets")
    distillation = config["training"].get("distillation")
    soft_targets = soft_weights = None
    if distillation is not None:
        if (subset is not None or full_expected is None or
                distillation != {"temperature": 1.0, "hard_fraction": 0.5, "soft_fraction": 0.5,
                  "soft_weighting": "inverse_images_per_study",
                  "normalization": "separate hard image-label and soft Study-label denominators"}):
            parser.error("Unreviewed distillation configuration")
        from cxr.batch.extension import load_soft_studies
        from cxr.batch.plan import digest
        teacher = config["data"]["teacher_soft_labels"]
        labels_path = output_root()/"manifests/train_labels.csv"
        if digest(labels_path) != teacher["label_sha256"] or teacher["formula"] != {"D17": .25, "C17": .25, "S17": .5}:
            parser.error("Teacher fixed labels or formula changed")
        soft_targets, soft_weights = load_soft_studies(teacher["path"], teacher["sha256"], train_rows, labels_path)
    workers = int(config["data"]["num_workers"])
    exact_epoch_resume = True  # Batch seed runs also restore all RNG states.
    batch_size = int(config["training"]["batch_size"])
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        parser.error("gradient_accumulation_steps must be at least 1")
    train_loader = make_dataloader(data_path, "train", batch_size=batch_size, num_workers=workers,
                                   shuffle=not bool(config["data"].get("train_sampling")), seed=seed,
                                   rows=train_rows, unlisted_policy=policy,
                                   augmentation=config["data"].get("train_augmentation"),
                                   preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                   input_size=int(config["data"]["input_size"]),
                                   sampling=config["data"].get("train_sampling"),
                                   persistent_workers=False if exact_epoch_resume else None,
                                   study_soft_targets=soft_targets, study_soft_weights=soft_weights)
    val_loader = make_dataloader(data_path, "val", batch_size=batch_size, num_workers=workers,
                                 rows=val_rows, unlisted_policy=policy,
                                 preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                 input_size=int(config["data"]["input_size"]),
                                 persistent_workers=False if exact_epoch_resume else None)
    val_targets = study_targets_from_rows(val_rows, unlisted_policy=policy)
    model = create_model(model_name, pretrained=True).to("cuda")
    ema_model = (AveragedModel(model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(float(ema_settings["decay"])),
                               use_buffers=True) if ema_settings else None)
    ema_best_auc, ema_best_epoch = -1.0, 0
    ema_updates = 0
    classification_head = model.head if model_name == "swin_t" else model.classifier
    head_parameters = list(classification_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    if not backbone_parameters or not head_parameters or len(backbone_parameters) + len(head_parameters) != len(list(model.parameters())):
        raise ValueError("Optimizer parameter groups do not cover the model exactly once")
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": float(config["training"]["backbone_lr"])},
        {"params": head_parameters, "lr": float(config["training"]["classifier_lr"])},
    ], weight_decay=float(config["training"]["weight_decay"]))
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    if args.resume:
        state = torch.load(run_dir / "last_training.pt", map_location="cpu", weights_only=True)
        for filename, blob in state.get("best_artifacts", {}).items():
            (run_dir / filename).write_bytes(blob)
        metrics_path = run_dir / "metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in records if r["epoch"] <= state["epoch"]), encoding="utf-8")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        if ema_model is not None:
            if not all(key in state for key in ("ema_model_state_dict", "ema_best_auc", "ema_best_epoch", "ema_updates")):
                parser.error("Resume checkpoint lacks EMA state")
            ema_model.load_state_dict(state["ema_model_state_dict"])
            ema_best_auc = float(state["ema_best_auc"])
            ema_best_epoch = int(state["ema_best_epoch"])
            ema_updates = int(state["ema_updates"])
        if state.get("early_stopping_patience", tracker.patience) != tracker.patience or \
                state.get("early_stopping_min_delta", tracker.min_delta) != tracker.min_delta:
            parser.error("Resume checkpoint early-stopping settings differ from config")
        if tracker.min_delta > 0 and "patience_reference_auc" not in state:
            parser.error("Resume checkpoint lacks the min_delta patience reference")
        best_record = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
        tracker = EarlyStopTracker(
            patience=tracker.patience, min_delta=tracker.min_delta,
            best_auc=float(state["best_auc"]),
            best_epoch=int(state.get("best_epoch", best_record["epoch"])),
            patience_reference_auc=float(state.get("patience_reference_auc", state["best_auc"])),
            stale=int(state["stale"]), last_epoch=int(state["epoch"]),
        )
        start_epoch = tracker.last_epoch + 1
        if exact_epoch_resume:
            restore_rng_state(state, train_loader, val_loader)
    write_json(run_dir / "status.json", {"state": "running", "start_epoch": start_epoch,
                                         **tracker.state_dict(), "config": str(args.config.resolve())})
    print(f"RUN {run_dir} epochs={epochs} batch={batch_size} accumulation={accumulation_steps} train_images={len(train_rows)} val_studies={len(val_targets)}", flush=True)
    if args.preflight_steps:
        print(f"PREFLIGHT: {args.preflight_steps} optimizer steps; "
              f"G/H verifies resume and short Val/Test files, no formal checkpoint", flush=True)

    if start_epoch > epochs or tracker.stale >= tracker.patience:
        write_json(run_dir / "status.json", {"state": "complete", "epoch": tracker.last_epoch,
                   **tracker.state_dict(), "stop_epoch": tracker.last_epoch,
                   "stop_reason": "restored_completed_epoch", "best_model": "best_model.pt"})
        return
    for epoch in range(start_epoch, epochs + 1):
        study_sampler = train_loader.sampler if isinstance(train_loader.sampler, StudyBalancedSampler) else None
        if study_sampler is not None:
            study_sampler.set_epoch(epoch)
            observed_study_draws = Counter()
        model.train()
        total_loss, total_valid = 0.0, 0
        optimizer_steps, amp_skipped_steps = 0, 0
        amp_skip_step_indices = []
        consecutive_amp_skips = 0
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        train_iter = iter(train_loader)
        for step in range(1, steps_per_epoch + 1):
            microbatches = list(itertools.islice(train_iter, accumulation_steps))
            group_valid = sum(int(batch["valid_mask"].sum().item()) for batch in microbatches)
            if not microbatches or group_valid == 0:
                raise ValueError("Training accumulation group has no valid labels")
            group_soft_valid = (sum(float((batch["soft_weight"][:, None] * batch["valid_mask"]).sum())
                                    for batch in microbatches) if distillation else 0.0)
            if distillation and group_soft_valid <= 0:
                raise ValueError("Distillation group has no valid teacher labels")
            optimizer.zero_grad(set_to_none=True)
            for batch in microbatches:
                if study_sampler is not None:
                    observed_study_draws.update(zip(batch["subject_id"], batch["study_id"]))
                images = batch["image"].to("cuda", non_blocking=True)
                targets = batch["target"].to("cuda", non_blocking=True)
                mask = batch["valid_mask"].to("cuda", non_blocking=True)
                images,targets,mask=mixup_batch(images,targets,mask,config['training'])
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    per_label = per_label_loss(logits, targets, config["training"])
                    loss_sum = (per_label * mask).sum()
                    loss = loss_sum / group_valid
                    if distillation:
                        teacher = batch["soft_target"].to("cuda", non_blocking=True)
                        study_weight = batch["soft_weight"].to("cuda", non_blocking=True)
                        if teacher.shape != logits.shape or not bool(torch.isfinite(teacher).all()):
                            raise ValueError("Teacher soft targets are not finite aligned ten-class probabilities")
                        soft_sum = (F.binary_cross_entropy_with_logits(logits.float(), teacher,
                                    reduction="none") * mask * study_weight[:, None]).sum()
                        loss = 0.5 * loss + 0.5 * soft_sum / group_soft_valid
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, step {step}")
                scaler.scale(loss).backward()
                total_loss += (float(loss_sum.detach()) if not distillation else
                               0.5 * float(loss_sum.detach()) +
                               0.5 * float(soft_sum.detach()) * group_valid / group_soft_valid)
                total_valid += int(mask.sum().item())
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           float(config["training"]["grad_clip_norm"]))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                amp_skipped_steps += 1
                amp_skip_step_indices.append(step)
                consecutive_amp_skips += 1
                if consecutive_amp_skips >= 20:
                    raise FloatingPointError(f"Persistent AMP gradient overflow: 20 consecutive skipped updates at epoch {epoch}, step {step}")
            else:
                consecutive_amp_skips = 0
                optimizer_steps += 1
                scheduler.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
                    ema_updates += 1
            if step == 1 or step % 100 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} train_loss={total_loss/total_valid:.5f} elapsed_s={time.perf_counter()-epoch_started:.0f}", flush=True)
            if args.preflight_steps and step >= args.preflight_steps:
                if config.get('phase32'):
                    from cxr.phase32.probe import complete_gh_preflight
                    complete_gh_preflight(
                        run_dir=run_dir, config_path=args.config, config=config, model=model,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        train_loader=train_loader, val_loader=val_loader, archive=data_path,
                        optimizer_steps=optimizer_steps, amp_skipped_steps=amp_skipped_steps,
                        gradient_norm=gradient_norm)
                write_json(run_dir / "status.json", {"state": "preflight_passed", "steps": step,
                                                     "train_loss": total_loss / total_valid,
                                                     "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                                                     "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                                                     "optimizer_steps": optimizer_steps,
                                                     "amp_skipped_steps": amp_skipped_steps,
                                                     "amp_skip_step_indices": amp_skip_step_indices,
                                                     "ema_updates": ema_updates,
                                                     "gradient_norm_before_clipping": (
                                                         float(gradient_norm) if bool(torch.isfinite(gradient_norm)) else None)})
                return
        sampling_record = None
        if study_sampler is not None:
            exposure_histogram = Counter(observed_study_draws.values())
            expected_rounds, remainder = divmod(len(train_rows), len(study_sampler.keys))
            expected_histogram = {expected_rounds: len(study_sampler.keys) - remainder,
                                  expected_rounds + 1: remainder}
            if sum(observed_study_draws.values()) != len(train_rows) or \
                    set(observed_study_draws) != set(study_sampler.keys) or \
                    dict(exposure_histogram) != {count: n for count, n in expected_histogram.items() if n}:
                raise ValueError(f"Actual Study sampling differs from expected coverage at epoch {epoch}")
            single = {key for key, image_indices in study_sampler.groups.items() if len(image_indices) == 1}
            detail_path = run_dir / f"sampling_epoch_{epoch:02d}.csv.gz"
            with gzip.open(detail_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("Subject_id", "Study_id", "source_image_count", "sampled_image_count"))
                for key in study_sampler.keys:
                    writer.writerow((*key, len(study_sampler.groups[key]), observed_study_draws[key]))
            sampling_record = {
                "actual_image_draws": sum(observed_study_draws.values()),
                "study_count": len(observed_study_draws),
                "study_exposure_histogram": dict(sorted(exposure_histogram.items())),
                "single_image_study_count": len(single),
                "single_image_study_draws": sum(observed_study_draws[key] for key in single),
                "multi_image_study_count": len(study_sampler.keys) - len(single),
                "multi_image_study_draws": sum(count for key, count in observed_study_draws.items() if key not in single),
                "detail_file": detail_path.name,
            }
        metrics = evaluate(model, val_loader, val_targets, threshold=float(config["validation"]["f1_threshold"]))
        ema_metrics = (evaluate(ema_model.module, val_loader, val_targets,
                                threshold=float(config["validation"]["f1_threshold"]))
                       if ema_model is not None else None)
        ema_improved = bool(ema_metrics is not None and ema_metrics["macro_auc"] > ema_best_auc)
        if ema_improved:
            ema_best_auc, ema_best_epoch = ema_metrics["macro_auc"], epoch
        decision = tracker.observe(metrics["macro_auc"], epoch)
        stop_reason = "patience_exhausted" if decision.should_stop else (
            "max_epochs_reached" if epoch == epochs else None)
        epoch_record = {"epoch": epoch, "train_loss": total_loss / total_valid, "val": metrics,
                        "elapsed_seconds": time.perf_counter() - epoch_started,
                        "learning_rates": [group["lr"] for group in optimizer.param_groups],
                        "optimizer_steps": optimizer_steps, "amp_skipped_steps": amp_skipped_steps,
                        "amp_skip_step_indices": amp_skip_step_indices,
                        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                        "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                        "study_sampling": sampling_record,
                        "ema_val": ema_metrics, "ema_best_auc": ema_best_auc,
                        "ema_best_epoch": ema_best_epoch, "ema_updates": ema_updates,
                        "ema_best_checkpoint_saved": ema_improved,
                        **tracker.state_dict(), "best_checkpoint_saved": decision.new_best_checkpoint,
                        "patience_reset": decision.patience_reset,
                        "stop_epoch": epoch if stop_reason else None, "stop_reason": stop_reason}
        if decision.new_best_checkpoint:
            save_model_state(model, run_dir / "best_model.pt", model_name=model_name)
            write_json(run_dir / "best_metrics.json", epoch_record)
        if ema_improved:
            save_model_state(ema_model.module, run_dir / "best_ema_model.pt", model_name=model_name)
            write_json(run_dir / "best_ema_metrics.json", epoch_record)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        state = {"epoch": epoch, **tracker.state_dict(),
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict()}
        if exact_epoch_resume:
            state.update(capture_rng_state(train_loader, val_loader))
        if ema_model is not None:
            state.update({"ema_model_state_dict": ema_model.state_dict(), "ema_best_auc": ema_best_auc,
                          "ema_best_epoch": ema_best_epoch, "ema_updates": ema_updates})
        state["best_artifacts"] = {name: (run_dir / name).read_bytes() for name in
            ("best_model.pt", "best_metrics.json", "best_ema_model.pt", "best_ema_metrics.json")
            if (run_dir / name).exists()}
        temp = run_dir / "last_training.tmp"
        torch.save(state, temp)
        temp.replace(run_dir / "last_training.pt")
        write_json(run_dir / "status.json", {"state": "running", "epoch": epoch,
                                             **tracker.state_dict(), "stop_reason": stop_reason})
        ema_text = (f" ema_auc={ema_metrics['macro_auc']:.5f} ema_best_auc={ema_best_auc:.5f} ema_best_epoch={ema_best_epoch}"
                    if ema_metrics is not None else "")
        print(f"EPOCH_DONE epoch={epoch} loss={epoch_record['train_loss']:.5f} macro_auc={metrics['macro_auc']:.5f} macro_f1={metrics['macro_f1']:.5f} best_auc={tracker.best_auc:.5f} best_epoch={tracker.best_epoch} patience={tracker.patience} stale={tracker.stale} seconds={epoch_record['elapsed_seconds']:.0f}{ema_text}", flush=True)
        if stop_reason:
            print(f"STOP epoch={epoch} reason={stop_reason} best_epoch={tracker.best_epoch} patience={tracker.patience}", flush=True)
            break
    write_json(run_dir / "status.json", {"state": "complete", "epoch": epoch,
                                         **tracker.state_dict(), "stop_epoch": epoch,
                                         "stop_reason": stop_reason, "best_model": "best_model.pt"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if ACTIVE_RUN_DIR is not None:
            status_path = ACTIVE_RUN_DIR / "status.json"
            prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            prior.update({"state": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "stop_epoch": prior.get("epoch"), "stop_reason": "error"})
            write_json(status_path, prior)
        raise
