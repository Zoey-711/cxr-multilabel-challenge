"""Summarize frozen-feature architecture results and local deployment measurements."""

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from cxr.data.archive import archive_path
from cxr.data.dataset import CXRArchiveDataset
from cxr.models.study_pool import StudyPool
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config
from train_study_heads import Studies, collate, predict
from torch.utils.data import DataLoader
import numpy as np


def timed(call, repeats=50):
    samples = []
    for i in range(repeats + 5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        if i >= 5:
            samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "p95_ms": sorted(samples)[int(.95 * len(samples)) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result_dir = project / "outputs/experiments/v00_v01_v02_m02_frozen_features"
    data = json.loads((result_dir / "comparison.json").read_text(encoding="utf-8"))
    source = project / cfg["source"]["m02_run"]
    m02_cfg = load_config(source / "config.yaml")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for latency measurement")
    model = load_model_state(source / "best_model.pt", device="cuda")
    torch.cuda.reset_peak_memory_stats()
    dataset = CXRArchiveDataset(archive_path(m02_cfg), "val", augmentation=None,
                                preprocessing="black_pad_v1", input_size=384)
    image = dataset[0]["image"].unsqueeze(0).cuda()
    heads = {}
    validation = Studies(project / cfg["source"]["feature_cache"], "val")
    validation_loader = DataLoader(validation, batch_size=128, shuffle=False, collate_fn=collate)
    with torch.inference_mode():
        def feature():
            with torch.autocast("cuda", dtype=torch.float16):
                return model.classifier[1](model.classifier[0](model.avgpool(model.features(image))))
        backbone_latency = timed(feature, repeats=30)
        one = feature().float()
        for kind in ("v00", "v01", "v02"):
            checkpoint = torch.load(result_dir / kind / "best_head.pt", map_location="cpu", weights_only=True)
            head = StudyPool(kind).cuda().eval()
            head.load_state_dict(checkpoint["model_state_dict"])
            recomputed = predict(head, validation_loader, len(validation))
            with (result_dir / kind / "study_predictions.csv").open(encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
            recorded = np.array([[float(row[f"probability_{j}"]) for j in range(1, 11)] for row in saved])
            if len(saved) != len(validation) or not np.allclose(recomputed, recorded, atol=1e-7, rtol=0):
                raise ValueError(f"{kind} checkpoint does not reproduce all validation Study probabilities")
            mask = torch.ones(1, 1, dtype=torch.bool, device="cuda")
            double = torch.cat((one, one), dim=0).unsqueeze(0)
            double_mask = torch.ones(1, 2, dtype=torch.bool, device="cuda")
            heads[kind] = {"single_study_head_latency": timed(lambda: head(one.unsqueeze(0), mask)),
                           "two_image_study_head_latency": timed(lambda: head(double, double_mask)),
                           "single_image_backbone_plus_head_latency": timed(
                               lambda: head(feature().float().unsqueeze(0), mask), repeats=30)}
            def full_single_image():
                pixels = dataset[0]["image"].unsqueeze(0).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = model.classifier[1](model.classifier[0](model.avgpool(model.features(pixels))))
                return head(hidden.float().unsqueeze(0), mask)
            heads[kind]["local_one_image_end_to_end_latency"] = timed(full_single_image, repeats=20)
    dataset.close()
    directories = ["b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated",
                   "b03_densenet121_lr3e5_20261016", "m02_convnext_tiny_384_aug_20261016_elevated"]
    weights = {name: (project / "outputs/experiments" / name / "best_model.pt").stat().st_size for name in directories}
    for kind in ("v00", "v01", "v02"):
        weights[kind] = (result_dir / kind / "best_head.pt").stat().st_size
    deployment = {"gpu": torch.cuda.get_device_name(), "measurement": "local CUDA, one representative validation image; dedicated end-to-end field includes ZIP read, decode, preprocessing, host-to-GPU transfer, frozen M02 feature forward and head. Other E03 models excluded; not a T4 certification",
                  "backbone_latency": backbone_latency, "heads": heads, "weight_file_bytes": weights,
                  "local_loaded_model_plus_head_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "e03_plus_largest_head_bytes": sum(weights[name] for name in directories) + max(weights[k] for k in ("v00", "v01", "v02")),
                  "limit_bytes": 500_000_000}
    (result_dir / "deployment.json").write_text(json.dumps(deployment, indent=2, ensure_ascii=False), encoding="utf-8")
    with (project / cfg["source"]["feature_cache"] / "train_images.csv").open(encoding="utf-8") as handle:
        train_counts = Counter((row["Subject_id"], row["Study_id"]) for row in csv.DictReader(handle))
    lines = ["# Phase 12: frozen M02 multi-view Study heads", "",
             "M02 checkpoint was SHA-256 verified before feature extraction. Only official Train features and labels updated the heads; validation was used for best checkpoint selection. Cached Train features use deterministic validation preprocessing, unlike M02's random training augmentation.",
             "", f"Train {data['train_studies']} Studies (single/multi-image {sum(v == 1 for v in train_counts.values())}/{sum(v > 1 for v in train_counts.values())}); validation {data['val_studies']} Studies / {data['val_patients']} patients (single/multi-image {data['single_studies']}/{data['multi_studies']}).",
             "", "## Architecture control", "",
             "All three heads take the same frozen 768-D per-image M02 features and ten-class Train Study labels. They use BCE, AdamW, lr 0.001, weight decay 0.0001, Study batch 128, cosine scheduler, seed 42, and exactly 30 epochs. Best validation Study Macro-AUC selects each checkpoint.",
             "", "- V00: masked mean of image features, then a 768-to-10 linear head.",
             "- V01: tanh/sigmoid gated attention with masked softmax, then the same linear head.",
             "- V02: one permutation-equivariant Transformer encoder layer (128-D, four heads, no positional embeddings), masked mean, 768-D projection and the same linear head. Padding is masked from attention.",
             "", "| Method | Best epoch | Macro-AUC | Macro-AP | Macro-F1@0.5 | Head parameters |", "|---|---:|---:|---:|---:|---:|"]
    base = data["scores"]["M02_mean_probability"]
    lines.append(f"| M02 Mean probability | — | {base['macro_auc']:.5f} | {base['macro_ap']:.5f} | {base['macro_f1_at_0_5']:.5f} | 0 |")
    for kind in ("v00", "v01", "v02"):
        item = data["scores"][kind]
        met = item["metrics"]
        lines.append(f"| {kind.upper()} | {item['best_epoch']} | {met['macro_auc']:.5f} | {met['macro_ap']:.5f} | {met['macro_f1_at_0_5']:.5f} | {item['parameters']} |")
    lines += ["", "## Paired patient bootstrap AUC differences", ""]
    for name, item in data["comparisons"].items():
        lines.append(f"- {name}: {item['point_delta']:+.5f}, 95% CI [{item['ci_95'][0]:+.5f}, {item['ci_95'][1]:+.5f}], {item['valid']}/{item['requested']} valid draws.")
    lines += ["", "## Per-class AUC / AP", "", "| Class | M02 AUC | V00 AUC | V01 AUC | V02 AUC | M02 AP | V00 AP | V01 AP | V02 AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for j in range(10):
        values = [data["scores"]["M02_mean_probability"]["per_class"][j]] + [data["scores"][k]["metrics"]["per_class"][j] for k in ("v00", "v01", "v02")]
        lines.append(f"| {j+1} {values[0]['class_name']} | " + " | ".join(f"{v['auc']:.5f}" for v in values) + " | " + " | ".join(f"{v['ap']:.5f}" for v in values) + " |")
    lines += ["", "## Single/multi-image subgroups", "", "Subgroup class composition differs; subgroup AUC comparisons are descriptive.", ""]
    m02_single = data["scores"]["M02_single_image"]
    m02_multi = data["scores"]["M02_multi_image"]
    lines.append(f"- M02 Mean probability: single-image Macro-AUC {m02_single['macro_auc']:.5f} ({m02_single['classes_with_auc']}/10 evaluable classes); multi-image {m02_multi['macro_auc']:.5f} ({m02_multi['classes_with_auc']}/10).")
    for kind in ("v00", "v01", "v02"):
        item = data["scores"][kind]
        lines.append(f"- {kind.upper()}: single-image Macro-AUC {item['single_image']['macro_auc']:.5f} ({item['single_image']['classes_with_auc']}/10 evaluable classes); multi-image {item['multi_image']['macro_auc']:.5f} ({item['multi_image']['classes_with_auc']}/10).")
    lines += ["", "Focused classes 1, 4, 8 by view count (positive/negative Study support; AUC):", "",
              "| Class | Group | Positive | Negative | M02 | V00 | V01 | V02 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for j in (0, 3, 7):
        for group, baseline_key in (("single", "M02_single_image"), ("multi", "M02_multi_image")):
            key = "single_image" if group == "single" else "multi_image"
            baseline_row = data["scores"][baseline_key]["per_class"][j]
            head_rows = [data["scores"][kind][key]["per_class"][j] for kind in ("v00", "v01", "v02")]
            lines.append(f"| {j+1} | {group} | {baseline_row['positive_studies']} | {baseline_row['negative_studies']} | " +
                         " | ".join(f"{row['auc']:.5f}" if row['auc'] is not None else "NA" for row in [baseline_row, *head_rows]) + " |")
    lines += ["", "## Training trend", ""]
    for kind in ("v00", "v01", "v02"):
        with (result_dir / kind / "metrics.jsonl").open(encoding="utf-8") as handle:
            log = [json.loads(line) for line in handle]
        best = data["scores"][kind]["best_epoch"]
        lines.append(f"- {kind.upper()}: best epoch {best}, final epoch {log[-1]['epoch']}; AUC {log[best-1]['val_macro_auc']:.5f} to {log[-1]['val_macro_auc']:.5f} while Train BCE {log[best-1]['train_loss']:.5f} to {log[-1]['train_loss']:.5f}; head training {data['scores'][kind]['elapsed_seconds']:.1f} s, peak GPU allocation {data['scores'][kind]['gpu_peak_allocated_mib']:.1f} MiB.")
    lines += ["", "## Deployment", "", f"Local GPU: {deployment['gpu']}; M02 backbone median {backbone_latency['median_ms']:.2f} ms. E03's four unique checkpoints plus the largest V head total {deployment['e03_plus_largest_head_bytes'] / 1e6:.2f} MB, below 500 MB.",
              "The separate local one-image end-to-end timings in `deployment.json` include ZIP decode and preprocessing; they exclude other E03 models. Official T4 end-to-end 100 ms compliance remains unverified.",
              "", "## Interpretation and next study", "", "V00 has the highest head AUC, but its patient-bootstrap improvement over M02 crosses zero. V01 provides no clear gain over V00; V02 falls below the non-learning baseline. Every head peaks at epoch 1 while Train BCE keeps falling, consistent with early overfitting of learned aggregation on these frozen features. Class 1 V00 AP improves but has only 24 positive validation Studies, while classes 4 and 8 show mixed AUC/AP changes. No E03+head fusion was tested; E03 remains the frozen submission candidate.",
              "", "A controlled residual correction could add a small bounded correction to the frozen M02 Study logits, initialized at zero, with the same Train-only supervision and a predeclared strength. Design only; no residual model was trained here. For a stronger generalization check, a true internal patient holdout requires retraining the backbone without those patients; the frozen M02 backbone has already seen all Train patients. No backbone fine-tuning, FPN or disease-specific query decoder was run."]
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(result_dir / "report.md"), "weight_bytes": deployment["e03_plus_largest_head_bytes"]}))


if __name__ == "__main__":
    main()
