"""Read-only Phase50 provenance audit plus isolated CPU/CUDA synthetic smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.models.factory import create_model
from cxr.data.archive import read_split_rows, image_member, split_archive_path
from cxr.data.dataset import preprocess_image

FOLDS = ROOT / "outputs/analysis/phase17_patient_folds"
OOF = ROOT / "outputs/analysis/phase17_final_handoff_20261023/e17_oof_study_probabilities.csv"
INDEX = ROOT / "outputs/analysis/phase17_final_handoff_20261023/e17_oof_source_index.csv"
WEIGHT = ROOT / ".wheels/torch_cache/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth"
FROZEN = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"
EXPECTED_FOLDS = {"patient_folds.csv": "04ed3b715910bacc016c3b489c193b24267c00c119035ecefef4e1fda025158c",
                  "study_folds.csv": "d92fac966dece9f797e0cf323100be1bfc35da698473c1d36016df8350b62edf",
                  "image_folds.csv": "f043e61739d5e6a7384b2ef62e5568f2c6a574c963f14b32ee68a3eba16c164d"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit():
    checks = {k: digest(FOLDS / k) == v for k, v in EXPECTED_FOLDS.items()}
    checks["imagenet_sha256"] = digest(WEIGHT) == "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
    frozen_config = json.loads((FROZEN / "config/deployment_config.json").read_text(encoding="utf-8"))
    checks["frozen_h17_weight"] = digest(FROZEN / frozen_config["model_file"]) == frozen_config["model_sha256"]
    patients = read(FOLDS / "patient_folds.csv")
    fold_col = next(k for k in patients[0] if k.lower() == "fold")
    patient_col = next(k for k in patients[0] if k.lower() == "subject_id")
    assignments = {r[patient_col]: int(r[fold_col]) for r in patients}
    checks["patient_assignments_unique"] = len(assignments) == len(patients) == 11055
    studies = read(OOF)
    sources = read(INDEX)
    source_by_key = {(r["Subject_id"], r["Study_id"]): r for r in sources}
    keys = {(r["Subject_id"], r["Study_id"]) for r in studies}
    checks["teacher_study_coverage"] = len(studies) == len(keys) == len(sources) == len(source_by_key) == 18513
    checks["teacher_oof_source_fold"] = all(int(r["Fold"]) == assignments[r["Subject_id"]] and
                                            all(r[f"{b}_Model_ID"] == f"{b}_F{int(r['Fold']):02d}" for b in ("D17", "C17", "S17"))
                                            for r in sources)
    checks["teacher_probability_range"] = all(all(0 <= float(r[f"Prob_{j}"]) <= 1 for j in range(1, 11)) for r in studies)
    archive = Path("F:/数据集/cxr_first.zip")
    image_rows = read_split_rows(archive, "train")
    checks["official_train_image_rows"] = len(image_rows) == 30344
    chosen = next(r for r in image_rows if assignments[r["Subject_id"]] != 1)
    from PIL import Image
    with zipfile.ZipFile(split_archive_path(archive, "train")) as source:
        with Image.open(io.BytesIO(source.read(image_member(chosen, "train")))) as image:
            sample = preprocess_image(image, 384, preprocessing="black_pad_v1")
    checks["real_image_preprocess"] = tuple(sample.shape) == (3, 384, 384) and bool(torch.isfinite(sample).all())
    # Every source model for a Fold2-5 student-training patient was trained on
    # the Fold1 validation patients. This is a nested-validation violation.
    training_source_folds = sorted({assignments[r["Subject_id"]] for r in studies if assignments[r["Subject_id"]] != 1})
    checks["teacher_isolated_from_fold1_validation"] = all(fold == 1 for fold in training_source_folds)
    release = ROOT / "outputs/experiments/phase33_h17_finetune/P33_H17_HEAD_01"
    return dict(checks=checks, counts=dict(patients=len(assignments), studies=len(keys),
                                          fold1_validation_patients=sum(v == 1 for v in assignments.values()),
                                          student_training_teacher_source_folds=training_source_folds,
                                          official_train_images=len(image_rows)),
                baseline=dict(frozen_h17_training_scope="full Train, distilled; no independent Fold1 OOF",
                              full_train_checkpoint_cannot_initialize_fold1=True,
                              same_protocol_head_only_experiment_exists=release.is_dir(),
                              imagenet_sha256=digest(WEIGHT), frozen_weight_sha256=digest(FROZEN / frozen_config["model_file"])),
                decision="KD_BLOCKED: Fold2-5 OOF teachers trained on Fold1 validation patients; nested teachers required")


def smoke(device):
    torch.manual_seed(42)
    model = create_model("efficientnet_v2_s", num_classes=10, pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=device == "cuda")
    ema = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)
    x = torch.randn(2, 3, 384, 384, device=device)
    y = torch.randint(0, 2, (2, 10), device=device).float()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise FloatingPointError("nonfinite logits or loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite unscaled gradient")
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before
    if skipped:
        raise FloatingPointError("AMP skipped smoke optimizer step")
    ema.update_parameters(model)
    result = dict(state="PASS", device=device, input_shape=list(x.shape), output_shape=list(logits.shape),
                  loss=float(loss.detach()), gradient_norm=float(norm.detach()), amp_skips=0, ema_updates=int(ema.n_averaged),
                  peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    report = audit()
    report["cpu_smoke"] = smoke("cpu")
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["cuda_smoke"] = smoke("cuda")
    (HERE / "preflight_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
