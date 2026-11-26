"""Report V03 validation, subgroup and local Study inference measurements."""

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from cxr.data.archive import archive_path
from cxr.data.dataset import CXRArchiveDataset
from cxr.engine.b01_analysis import read_study_predictions
from cxr.models.study_residual import MultiViewResidual
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config
from train_v03_residual import load_previous
from train_study_heads import Studies


def timed(call, repeats=20):
    values = []
    for i in range(repeats + 5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        if i >= 5:
            values.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(values),
            "p95_ms": sorted(values)[int(.95 * len(values)) - 1], "repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = root / "outputs/experiments/v03_m02_multiview_residual"
    data = json.loads((result / "comparison.json").read_text(encoding="utf-8"))
    protocol = root / "outputs/analysis/phase13_internal_protocol"
    audit = json.loads((protocol / "audit.json").read_text(encoding="utf-8"))
    source = root / cfg["source"]["m02_run"]
    m02_cfg = load_config(source / "config.yaml")
    val = Studies(root / cfg["source"]["feature_cache"], "val")
    original, original_targets, subjects, studies = read_study_predictions(
        source / "validation_analysis/study_predictions.csv")
    lookup = {key: i for i, key in enumerate(zip(subjects, studies))}
    order = [lookup[key] for key in val.keys]
    m02 = original[order]
    if not np.array_equal(original_targets[order], val.labels):
        raise ValueError("M02 validation labels differ")
    v03 = load_previous(result / "study_predictions.csv", val.keys, val.labels)
    single = np.array([len(group) == 1 for group in val.indices])
    if not np.array_equal(m02[single], v03[single]):
        raise ValueError("Single-view predictions differ from M02")
    multi_delta = np.abs(v03[~single] - m02[~single])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for deployment measurement")
    backbone = load_model_state(source / "best_model.pt", device="cuda")
    head_checkpoint = torch.load(result / "best_head.pt", map_location="cpu", weights_only=True)
    head = MultiViewResidual(768, 0.1, 0.25).cuda().eval()
    head.load_state_dict(head_checkpoint["model_state_dict"], strict=True)
    dataset = CXRArchiveDataset(archive_path(m02_cfg), "val", augmentation=None,
                                preprocessing="black_pad_v1", input_size=384)
    by_study = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        by_study[(row["Subject_id"], row["Study_id"])].append(i)
    one_index = next(indices for indices in by_study.values() if len(indices) == 1)
    two_indices = next(indices[:2] for indices in by_study.values() if len(indices) >= 2)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        def study_forward(indices):
            images = torch.stack([dataset[index]["image"] for index in indices]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                pooled = backbone.avgpool(backbone.features(images))
                features = backbone.classifier[1](backbone.classifier[0](pooled))
                image_logits = backbone.classifier[2](features)
                baseline_probability = image_logits.sigmoid().float().mean(dim=0, keepdim=True)
            baseline_logit = torch.logit(baseline_probability.clamp(1e-6, 1 - 1e-6))
            return head(features.float().unsqueeze(0),
                        torch.ones(1, len(indices), dtype=torch.bool, device="cuda"),
                        baseline_logit).sigmoid()
        one_time = timed(lambda: study_forward(one_index))
        two_time = timed(lambda: study_forward(two_indices))
        features = torch.from_numpy(np.array(val.features[val.indices[np.flatnonzero(~single)[0]]], dtype=np.float32)).cuda()
        baseline = torch.zeros(1, 10, device="cuda")
        extra_time = timed(lambda: head(features.unsqueeze(0),
                                       torch.ones(1, len(features), dtype=torch.bool, device="cuda"), baseline), repeats=50)
    dataset.close()
    checkpoint_paths = [root / "outputs/experiments" / directory / "best_model.pt" for directory in
                        ("b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated",
                         "b03_densenet121_lr3e5_20261016", "m02_convnext_tiny_384_aug_20261016_elevated")]
    total_bytes = sum(path.stat().st_size for path in checkpoint_paths) + (result / "best_head.pt").stat().st_size
    deployment = {"gpu": torch.cuda.get_device_name(), "single_image_study": one_time,
                  "two_image_study": two_time, "head_only_multi_view": extra_time,
                  "loaded_backbone_and_head_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "v03_head_bytes": data["head_bytes"], "e03_four_models_plus_v03_head_bytes": total_bytes,
                  "max_model_bytes": 500_000_000,
                  "latency_scope": "Local RTX 4060, one representative validation Study per view count; includes ZIP read/decode, deterministic preprocessing, host-to-GPU transfer, frozen M02 and V03. Other E03 backbones excluded. Official T4 100ms compliance unverified."}
    (result / "deployment.json").write_text(json.dumps(deployment, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase 13: V03 conservative multi-view residual", "",
             "V03 uses the frozen M02 768-D per-image features and Mean probability. It computes masked cross-view feature variance, applies dropout 0.1 and one bias-free 768-to-10 projection, then bounds the logit correction to +/-0.25. The projection starts at zero. A one-image Study has exactly zero variance and therefore exactly preserves M02 probabilities.",
             "", "The head was trained only on official Train multi-image Studies, using BCE plus residual L2 0.01, AdamW lr 0.001 / weight decay 0.001, cosine schedule, seed 42 and fixed 30 epochs. Official validation selected the best head; no validation label updated its parameters. The frozen training feature cache uses deterministic preprocessing, unlike M02's randomized training augmentation.",
             "", f"Best epoch {data['best_epoch']}; Train multi-image Studies {data['train_multi_image_studies']}; validation {data['validation_studies']} Studies / {data['validation_patients']} patients (single/multi {data['validation_single_studies']}/{data['validation_multi_studies']}).",
             "", "| Method | Macro-AUC | Macro-AP | Macro-F1@0.5 | Single AUC | Multi AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("M02", "v00", "v01", "v02", "V03"):
        scores = data["scores"][name]
        subgroup = data["subgroups"][name]
        lines.append(f"| {name.upper()} | {scores['macro_auc']:.5f} | {scores['macro_ap']:.5f} | {scores['macro_f1_at_0_5']:.5f} | {subgroup['single']['macro_auc']:.5f} | {subgroup['multi']['macro_auc']:.5f} |")
    lines += ["", f"Single-image Study predictions match frozen M02 exactly for all {single.sum()} Studies. On {len(val) - single.sum()} multi-image Studies, mean absolute probability change is {multi_delta.mean():.6f}, maximum {multi_delta.max():.6f}.",
              "", "## Paired 228-patient bootstrap: V03 minus comparator", ""]
    for name, item in data["paired_patient_bootstrap_v03_minus"].items():
        lines.append(f"- {name.upper()}: {item['point_delta']:+.5f}; 95% CI [{item['ci_95'][0]:+.5f}, {item['ci_95'][1]:+.5f}]; {item['valid']}/{item['requested']} valid draws.")
    lines += ["", "## Ten-class AUC / AP", "", "| Class | M02 AUC | V00 AUC | V01 AUC | V02 AUC | V03 AUC | M02 AP | V00 AP | V01 AP | V02 AP | V03 AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for j in range(10):
        rows = [data["scores"][key]["per_class"][j] for key in ("M02", "v00", "v01", "v02", "V03")]
        lines.append(f"| {j+1} {rows[0]['class_name']} | " + " | ".join(f"{row['auc']:.5f}" for row in rows) +
                     " | " + " | ".join(f"{row['ap']:.5f}" for row in rows) + " |")
    lines += ["", "## Focus: classes 1, 4, 8", "", "| Class | Positive Studies | M02 AUC | V03 AUC | M02 AP | V03 AP | Single AUC M02/V03 | Multi AUC M02/V03 |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for j in (0, 3, 7):
        base = data["scores"]["M02"]["per_class"][j]
        changed = data["scores"]["V03"]["per_class"][j]
        sb = data["subgroups"]["M02"]["single"]["per_class"][j]
        sv = data["subgroups"]["V03"]["single"]["per_class"][j]
        mb = data["subgroups"]["M02"]["multi"]["per_class"][j]
        mv = data["subgroups"]["V03"]["multi"]["per_class"][j]
        lines.append(f"| {j+1} | {base['positive_studies']} | {base['auc']:.5f} | {changed['auc']:.5f} | {base['ap']:.5f} | {changed['ap']:.5f} | {sb['auc']:.5f}/{sv['auc']:.5f} | {mb['auc']:.5f}/{mv['auc']:.5f} |")
    logs = [json.loads(row) for row in (result / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    lines += ["", "## Trend and deployment", "", f"At best epoch {data['best_epoch']}, AUC {logs[data['best_epoch']]['val_macro_auc']:.5f}; final epoch 30 AUC {logs[-1]['val_macro_auc']:.5f}. Multi-Study Train loss fell from {logs[data['best_epoch']]['train_loss']:.5f} to {logs[-1]['train_loss']:.5f}. Head training took {data['elapsed_seconds']:.1f} s and peak head GPU allocation was {data['peak_gpu_allocated_mib']:.1f} MiB.",
              f"Head parameters {data['head_parameters']}, checkpoint {data['head_bytes']} bytes. Four E03 checkpoints plus V03 head total {total_bytes / 1e6:.2f} MB (<500 MB). Local single-image Study median {one_time['median_ms']:.1f} ms; two-image Study median {two_time['median_ms']:.1f} ms; see `deployment.json` for p95 and scope. These are not official T4 measurements.",
              "", "## Independent-patient protocol", "", f"Frozen split: internal Train {audit['counts']['internal_train']['patients']} patients / {audit['counts']['internal_train']['studies']} Studies / {audit['counts']['internal_train']['images']} images; Holdout {audit['counts']['internal_val']['patients']} / {audit['counts']['internal_val']['studies']} / {audit['counts']['internal_val']['images']}. No patient crosses. See `outputs/analysis/phase13_internal_protocol/report.md` for support counts and the retrain plan. Frozen M02 has seen Holdout patients and is invalid for that comparison. No internal retraining was started.",
              "", "V03's +0.00068 point AUC change relative to M02 is small and its patient-bootstrap CI crosses zero. The multi-image gain is minimal despite exact single-image preservation. Do not replace the current E03 candidate on this evidence alone."]
    (result / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(result / "report.md"), "single_ms": one_time["median_ms"],
                      "two_image_ms": two_time["median_ms"], "combined_bytes": total_bytes}))


if __name__ == "__main__":
    main()
