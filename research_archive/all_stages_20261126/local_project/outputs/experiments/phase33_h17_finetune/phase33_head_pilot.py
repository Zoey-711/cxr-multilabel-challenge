"""One isolated H17-02 head-only continuation. Never writes to the source run."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cxr.data.archive import read_split_rows
from cxr.data.dataset import CXRArchiveDataset, make_dataloader, study_targets_from_rows
from cxr.engine.metrics import study_multilabel_metrics
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.models.factory import create_model
from cxr.utils.training_resume import capture_rng_state, restore_rng_state

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/analysis/phase29_deployment/weights/H17_DISTILL_02_best_model.pth"
TEACHER = ROOT.parent / "cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/analysis/gh_teacher_oof.csv"
FRIEND = TEACHER.parent.parent / "experiments/H17_DISTILL_02"
ARCHIVE = Path("F:/数据集/cxr_first.zip")
SOURCE_SHA = "35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
TEACHER_SHA = "13ded125b51b087fe82b2f63b3bbbaee9a115dea31434d6b66fd547bf390a324"
CONFIG = {"id": "P33_H17_HEAD_01", "seed": 42, "source_model": str(SOURCE),
          "source_sha256": SOURCE_SHA, "teacher_oof": str(TEACHER), "teacher_sha256": TEACHER_SHA,
          "archive": str(ARCHIVE), "architecture": "efficientnet_v2_s", "input_size": 384,
          "preprocessing": "black_pad_v1", "train_augmentation": "conservative_cxr_v1",
          "aggregation": "mean_probability", "loss": "0.5 hard BCE + 0.5 Study-balanced teacher BCE",
          "temperature": 1.0, "head_lr": 1e-5, "weight_decay": 1e-4,
          "micro_batch": 8, "accumulation": 2, "max_epochs": 5, "patience": 2,
          "num_workers": 0,
          "precision": "fp16_amp", "training_scope": "head_only; frozen backbone and BN buffers",
          "val_role": "reused exploratory development set", "train_scope": "official complete Train"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def audit():
    for path, expected in ((SOURCE, SOURCE_SHA), (TEACHER, TEACHER_SHA)):
        if sha(path) != expected:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    manifest = json.loads((FRIEND / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["files"]["best_model.pth"]["sha256"] != SOURCE_SHA:
        raise ValueError("H17 manifest source weight differs")
    original = (FRIEND / "resolved_config.yaml").read_text(encoding="utf-8")
    if not all(x in original for x in ("efficientnet_v2_s", "hard_fraction: 0.5", "soft_fraction: 0.5",
                                        "inverse_images_per_study", "conservative_cxr_v1", "input_size: 384")):
        raise ValueError("Source distillation protocol differs")
    train_rows = read_split_rows(ARCHIVE, "train")
    val_rows = read_split_rows(ARCHIVE, "val")
    studies = {(r["Subject_id"], r["Study_id"]) for r in train_rows}
    val_studies = {(r["Subject_id"], r["Study_id"]) for r in val_rows}
    if (len(train_rows), len(studies), len({r["Subject_id"] for r in train_rows})) != (30344, 18513, 11055):
        raise ValueError("Train counts differ from H17")
    if (len(val_rows), len(val_studies), len({r["Subject_id"] for r in val_rows})) != (1422, 859, 228):
        raise ValueError("Val counts differ from H17")
    if {r["Subject_id"] for r in train_rows} & {r["Subject_id"] for r in val_rows}:
        raise ValueError("Train/Val patient overlap")
    provenance = json.loads(TEACHER.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    if provenance["csv_sha256"] != TEACHER_SHA or provenance["role"] != "training soft targets; unseen patient folds only":
        raise ValueError("Teacher provenance differs")
    with TEACHER.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)]:
            raise ValueError("Teacher class order differs")
        teacher = {}
        for row in reader:
            key = (row["Subject_id"], row["Study_id"])
            vector = np.array([float(row[f"Prob_{i}"]) for i in range(1, 11)], dtype=np.float32)
            if key in teacher or not np.isfinite(vector).all() or np.any((vector < 0) | (vector > 1)):
                raise ValueError("Invalid teacher Study or probability")
            teacher[key] = vector
    if set(teacher) != studies:
        raise ValueError("Teacher OOF coverage differs from official Train")
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in train_rows)
    return train_rows, val_rows, teacher, counts, {"train_images": len(train_rows), "train_studies": len(studies),
        "train_patients": 11055, "val_images": len(val_rows), "val_studies": 859,
        "teacher_sha256": TEACHER_SHA, "teacher_role": provenance["role"],
        "multi_image_studies": sum(v > 1 for v in counts.values()), "study_soft_weight_sum": sum(v * (1 / v) for v in counts.values()),
        "source_checkpoint_sha256": SOURCE_SHA, "source_training_config_sha256": sha(FRIEND / "resolved_config.yaml"),
        "original_optimizer_state_local": False, "resume_kind": "new head-only optimizer; not H17 optimizer continuation"}


class TeacherDataset(CXRArchiveDataset):
    def __init__(self, archive, rows, teacher, counts):
        super().__init__(archive, "train", rows=rows, unlisted_policy="negative", input_size=384,
                         preprocessing="black_pad_v1", augmentation={"enabled": True, "name": "conservative_cxr_v1",
                         "affine_probability": 0.5, "rotation_degrees": 3.0, "translation_fraction": 0.02,
                         "photometric_probability": 0.5, "brightness_contrast_fraction": 0.05})
        self.teacher, self.counts = teacher, counts

    def __getitem__(self, i):
        item = super().__getitem__(i)
        key = item["subject_id"], item["study_id"]
        item["soft_target"] = torch.from_numpy(self.teacher[key])
        item["soft_weight"] = torch.tensor(1 / self.counts[key], dtype=torch.float32)
        return item


def evaluate(model, loader, targets):
    model.eval()
    images = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                probability = torch.sigmoid(model(batch["image"].to("cuda", non_blocking=True)))
            for i, row in enumerate(probability.float().cpu().tolist()):
                images.append(ImagePrediction(batch["subject_id"][i], batch["study_id"][i],
                                              batch["image_id"][i], tuple(row)))
    metrics = study_multilabel_metrics(images, targets)
    study = aggregate_mean_probabilities(images)
    if len(study) != 859:
        raise ValueError("Val Study coverage changed")
    return metrics, study


def save_predictions(path, study):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)])
        for item in study:
            writer.writerow([item.subject_id, item.study_id] + list(item.probabilities))


def loss_for(batch, model):
    images = batch["image"].to("cuda", non_blocking=True)
    hard = batch["target"].to("cuda", non_blocking=True)
    soft = batch["soft_target"].to("cuda", non_blocking=True)
    weight = batch["soft_weight"].to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(images).float()
        hard_sum = F.binary_cross_entropy_with_logits(logits, hard, reduction="sum")
        soft_sum = (F.binary_cross_entropy_with_logits(logits, soft, reduction="none") * weight[:, None]).sum()
    return hard_sum, soft_sum, hard.numel(), (weight.sum() * 10).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error("Smoke and resume are mutually exclusive")
    run = HERE / CONFIG["id"]
    if not args.resume and not args.smoke and run.exists():
        parser.error("Pilot directory exists; use --resume only for incomplete run")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    train_rows, val_rows, teacher, counts, audit_info = audit()
    train_loader = DataLoader(TeacherDataset(ARCHIVE, train_rows, teacher, counts), batch_size=8, shuffle=True,
                              num_workers=0, pin_memory=True, persistent_workers=False,
                              generator=torch.Generator().manual_seed(42))
    val_loader = make_dataloader(ARCHIVE, "val", rows=val_rows, batch_size=8, num_workers=0,
                                 persistent_workers=False, preprocessing="black_pad_v1", input_size=384)
    targets = study_targets_from_rows(val_rows, unlisted_policy="negative")
    model = create_model("efficientnet_v2_s", pretrained=False).to("cuda")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    model.load_state_dict(source, strict=True)
    head = model.classifier[-1]
    if not isinstance(head, torch.nn.Linear) or head.out_features != 10:
        raise ValueError("H17 classifier structure changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    if args.smoke:
        model.eval(); head.train()
        batch = next(iter(train_loader))
        hard, soft, n, w = loss_for(batch, model)
        loss = 0.5 * hard / n + 0.5 * soft / w
        if not bool(torch.isfinite(loss)):
            raise ValueError("Smoke loss is non-finite")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if not all(torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("Smoke head parameters are non-finite")
        smoke = HERE / "smoke"
        smoke.mkdir(exist_ok=True)
        path = smoke / "resume_test.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, path)
        again = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(again["model"]); optimizer.load_state_dict(again["optimizer"]); scaler.load_state_dict(again["scaler"])
        write_json(smoke / "smoke_result.json", {"state": "PASS", "loss": float(loss), "hard_bce": float(hard / n),
                   "soft_bce": float(soft / w), "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                   "resume_sha256": sha(path), **audit_info})
        return
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        shutil.copy2(SOURCE, run / "source_model.pth")
        write_json(run / "resolved_config.json", CONFIG)
        write_json(run / "source_audit.json", audit_info)
        base_metrics, base_study = evaluate(model, val_loader, targets)
        save_predictions(run / "baseline_val_predictions.csv", base_study)
        write_json(run / "baseline_metrics.json", base_metrics)
        save_predictions(run / "val_predictions.csv", base_study)
        write_json(run / "best_metrics.json", base_metrics)
        best_auc, best_epoch, stale, start_epoch = base_metrics["macro_auc"], 0, 0, 1
        torch.save(model.state_dict(), run / "best_student.pth")
    else:
        if json.loads((run / "resolved_config.json").read_text(encoding="utf-8")) != CONFIG:
            raise ValueError("Resume configuration differs")
        state = torch.load(run / "last_training.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        restore_rng_state(state, train_loader, val_loader)
        best_auc, best_epoch, stale, start_epoch = state["best_auc"], state["best_epoch"], state["stale"], state["epoch"] + 1
    history = run / "train_history.csv"
    with history.open("a" if args.resume else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "hard_bce", "soft_bce", "val_macro_auc", "best_epoch", "stale", "seconds", "peak_mib", "amp_skips"])
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, 6):
            started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            model.eval(); head.train()
            total_hard = total_soft = total_n = total_w = 0.0; skips = 0
            optimizer.zero_grad(set_to_none=True)
            iterator = iter(train_loader)
            for _ in range(math.ceil(len(train_loader) / 2)):
                microbatches = list(itertools.islice(iterator, 2))
                group_n = sum(batch["target"].numel() for batch in microbatches)
                group_w = sum(float(batch["soft_weight"].sum()) * 10 for batch in microbatches)
                if not group_n or group_w <= 0:
                    raise ValueError("Empty accumulation group")
                for batch in microbatches:
                    hard, soft, n, w = loss_for(batch, model)
                    loss = 0.5 * hard / group_n + 0.5 * soft / group_w
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    total_hard += float(hard); total_soft += float(soft); total_n += n; total_w += w
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skips += scaler.get_scale() < before
                optimizer.zero_grad(set_to_none=True)
            metrics, study = evaluate(model, val_loader, targets)
            if metrics["macro_auc"] > best_auc:
                best_auc, best_epoch, stale = metrics["macro_auc"], epoch, 0
                temporary = run / "best_student.tmp"
                torch.save(model.state_dict(), temporary); temporary.replace(run / "best_student.pth")
                save_predictions(run / "val_predictions.csv", study)
                write_json(run / "best_metrics.json", metrics)
            else:
                stale += 1
            torch.save(model.state_dict(), run / f"epoch_{epoch:02d}.pth")
            record = {"epoch": epoch, "train_loss": 0.5 * total_hard / total_n + 0.5 * total_soft / total_w,
                      "hard_bce": total_hard / total_n, "soft_bce": total_soft / total_w,
                      "val_macro_auc": metrics["macro_auc"], "best_epoch": best_epoch, "stale": stale,
                      "seconds": round(time.perf_counter() - started, 2),
                      "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20), "amp_skips": skips}
            writer.writerow(record); stream.flush()
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), **capture_rng_state(train_loader, val_loader),
                     "best_auc": best_auc, "best_epoch": best_epoch, "stale": stale}
            temp = run / "last_training.tmp"
            torch.save(state, temp); temp.replace(run / "last_training.pt")
            write_json(run / "status.json", {"state": "RUNNING", **record})
            print(json.dumps(record), flush=True)
            if stale >= 2:
                break
    write_json(run / "status.json", {"state": "TRAINING_COMPLETE", "best_epoch": best_epoch,
                                      "best_auc": best_auc, "source_auc": json.loads((run / "baseline_metrics.json").read_text())["macro_auc"]})
    files = {p.name: sha(p) for p in run.iterdir() if p.is_file() and p.name not in ("last_training.pt", "source_model.pth")}
    write_json(run / "manifest.json", {"model_id": CONFIG["id"], "source_sha256": SOURCE_SHA,
                                         "teacher_sha256": TEACHER_SHA, "files_sha256": files,
                                         "oof_scope": "N/A: full Train student", "val_status": "EXPLORATORY"})


if __name__ == "__main__":
    main()
