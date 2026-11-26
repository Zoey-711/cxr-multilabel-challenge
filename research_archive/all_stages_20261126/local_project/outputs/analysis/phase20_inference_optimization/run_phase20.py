"""Read-only frozen-model inference audit; writes only beside this script.

Modes: benchmark (paired sample, no full-Val claims) and validate (all 859 Val
Studies, exact original per-model batch sizes). No training or Test access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from cxr.data.archive import image_member, parse_positive_classes, read_split_rows
from cxr.data.dataset import CXRArchiveDataset, IMAGENET_MEAN, IMAGENET_STD, resize_pad_image, preprocess_image
from cxr.engine.b01_analysis import read_study_predictions, binary_auc
from cxr.utils.checkpoint import load_model_state


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/submissions/E08_EMA_20261020/manifest.json"
NAMES = ("B01", "B02", "B03", "M02", "N01_EMA")
WEIGHT_E03 = (1 / 6, 1 / 6, 1 / 6, 1 / 2)
WEIGHT_E08 = (1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 2)
MODEL_PROB_ATOL = 1e-4
FUSION_PROB_ATOL = 1e-4
AUC_ATOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(values: list[float]) -> dict:
    return {"mean_ms": float(statistics.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def frozen_fusion(values: dict, names: tuple[str, ...]) -> np.ndarray:
    """Preserve frozen nesting and float64 arithmetic, including near ties."""
    e01 = (values["B01"] + values["B02"] + values["B03"]) / 3
    e03 = 0.5 * e01 + 0.5 * values["M02"]
    return e03 if len(names) == 4 else 0.5 * e03 + 0.5 * values["N01_EMA"]


def load_sources() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {}
    for name in NAMES:
        item = manifest["models"][name]
        if item["preprocessing"] != "black_pad_v1" or item["input_size"] != 384:
            raise ValueError(f"Different preprocessing requires a separate branch: {name}")
        path = ROOT / "outputs/experiments" / item["run_id"] / item["checkpoint"]
        if sha256(path) != item["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {name}")
        sources[name] = {"model": load_model_state(path, device="cuda"), "checkpoint": path,
                         "checkpoint_sha256": item["checkpoint_sha256"]}
    return manifest, sources


def choose_images(rows: list[dict]) -> list[dict]:
    """Deterministic spread over observed view and aspect-ratio strata."""
    picked = {}
    for row in rows:
        ratio = int(row["Columns"]) / int(row["Rows"])
        aspect = "narrow" if ratio < 0.8 else "wide" if ratio > 1.2 else "middle"
        view = row["ViewPosition"].upper()
        key = (view, aspect)
        if view in {"AP", "PA", "LATERAL"} and key not in picked:
            picked[key] = row
    chosen = list(picked.values())
    if len(chosen) < 6:
        for row in rows:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= 6:
                break
    return chosen[:9]


def decode_and_transform(zf: zipfile.ZipFile, row: dict, stages: dict | None = None) -> torch.Tensor:
    t = time.perf_counter()
    raw = zf.read(image_member(row, "val"))
    if stages is not None:
        stages["zip_read"].append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        image.load()
    if stages is not None:
        stages["decode_rgb"].append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
    if stages is not None:
        stages["resize_pad"].append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    tensor = TF.normalize(TF.to_tensor(padded), IMAGENET_MEAN, IMAGENET_STD)
    if stages is not None:
        stages["normalize"].append((time.perf_counter() - t) * 1000)
    return tensor.unsqueeze(0)


def timed_single(zf, row, sources, names, *, shared: bool, pinned: bool = False):
    stages = defaultdict(list)
    probabilities = {}
    torch.cuda.synchronize()
    begin = time.perf_counter()
    common = decode_and_transform(zf, row, stages) if shared else None
    for name in names:
        tensor = common if shared else decode_and_transform(zf, row, stages)
        t = time.perf_counter()
        gpu = tensor.pin_memory().to("cuda", non_blocking=True) if pinned else tensor.to("cuda")
        torch.cuda.synchronize()
        stages["host_to_device"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pred = torch.sigmoid(sources[name]["model"](gpu)).float().cpu().numpy()[0]
        torch.cuda.synchronize()
        stages[f"forward_{name}"].append((time.perf_counter() - t) * 1000)
        probabilities[name] = pred
    t = time.perf_counter()
    fused = frozen_fusion({name: value.astype(np.float64) for name, value in probabilities.items()}, names)
    study_mean = np.mean(fused[None, :], axis=0)
    stages["fusion_and_study_mean"].append((time.perf_counter() - t) * 1000)
    torch.cuda.synchronize()
    stages["full_end_to_end"].append((time.perf_counter() - begin) * 1000)
    if not np.isfinite(study_mean).all():
        raise ValueError("Nonfinite prediction")
    return probabilities, study_mean, stages


def benchmark(archive: Path, sources: dict) -> None:
    rows = choose_images(read_split_rows(archive, "val"))
    # Same images and GPU, alternating order to limit warm-cache and clock drift.
    runs = []
    stage_values = defaultdict(list)
    max_input_diff = 0.0
    max_probability_diff = 0.0
    with zipfile.ZipFile(archive) as zf:
        # Verify transform once per image, outside all timed regions.
        for row in rows:
            with Image.open(io.BytesIO(zf.read(image_member(row, "val")))) as opened:
                original = opened.convert("RGB")
            canonical = preprocess_image(original, 384, preprocessing="black_pad_v1").unsqueeze(0)
            candidate = decode_and_transform(zf, row)
            difference = float(torch.max(torch.abs(canonical - candidate)))
            max_input_diff = max(max_input_diff, difference)
            if not torch.equal(canonical, candidate):
                raise ValueError(f"Optimized model input differs for {row['Dicom_id']}")
        for candidate, names in (("E03", NAMES[:4]), ("E08_EMA", NAMES)):
            for i in range(3):
                for row in rows:
                    modes = ("baseline_repeated", "shared_once") if i % 2 == 0 else ("shared_once", "baseline_repeated")
                    outputs = {}
                    for mode in modes:
                        probs, fused, stages = timed_single(zf, row, sources, names, shared=mode == "shared_once")
                        outputs[mode] = (probs, fused)
                        if i:
                            runs.append({"candidate": candidate, "mode": mode, "image_id": row["Dicom_id"],
                                         "view": row["ViewPosition"], "width": int(row["Columns"]),
                                         "height": int(row["Rows"]), "full_ms": stages["full_end_to_end"][0]})
                            for stage, values in stages.items():
                                stage_values[(candidate, mode, stage)].extend(values)
                    for name in names:
                        max_probability_diff = max(max_probability_diff,
                            float(np.max(np.abs(outputs["baseline_repeated"][0][name] - outputs["shared_once"][0][name]))))
                    max_probability_diff = max(max_probability_diff,
                        float(np.max(np.abs(outputs["baseline_repeated"][1] - outputs["shared_once"][1]))))
    summary = []
    for candidate in ("E03", "E08_EMA"):
        for mode in ("baseline_repeated", "shared_once"):
            values = [r["full_ms"] for r in runs if r["candidate"] == candidate and r["mode"] == mode]
            summary.append({"candidate": candidate, "mode": mode, **stat(values),
                            "images": len(rows), "measured_runs": len(values), "device": torch.cuda.get_device_name(0),
                            "batch_size": 1, "precision": "FP16 autocast", "max_probability_abs_diff": max_probability_diff,
                            "status": "sample_benchmark_only"})
    with (OUT / "optimization_benchmark.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    breakdown = [{"candidate": candidate, "mode": mode, "stage": stage, **stat(values), "samples": len(values),
                  "device": torch.cuda.get_device_name(0), "stage_timings_additive": False}
                 for (candidate, mode, stage), values in sorted(stage_values.items())]
    with (OUT / "latency_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=breakdown[0]); w.writeheader(); w.writerows(breakdown)
    (OUT / "baseline_benchmark.json").write_text(json.dumps({"fixed_images": [
        {"image_id": r["Dicom_id"], "view": r["ViewPosition"], "rows": int(r["Rows"]),
         "columns": int(r["Columns"])} for r in rows], "warmup_rounds": 1,
        "measured_rounds": 2, "timing_scope": "ZIP read through fused single-image Study probability; models preloaded",
        "cuda_synchronize": "before and after full path and GPU stages", "paired_summary": summary,
        "max_input_abs_diff": max_input_diff, "max_probability_abs_diff": max_probability_diff, "raw_measurements": runs,
        "model_load_in_timing": False, "target_hardware_verified": False},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def reference_matrix(name: str, manifest: dict):
    run = ROOT / "outputs/experiments" / manifest["models"][name]["run_id"]
    rel = "ema_validation_analysis/study_predictions.csv" if name == "N01_EMA" else "validation_analysis/study_predictions.csv"
    prob, labels, subjects, studies = read_study_predictions(run / rel)
    return {(sub, study): prob[i] for i, (sub, study) in enumerate(zip(subjects, studies))}, labels


def validate(archive: Path, manifest: dict, sources: dict, *, channels_last: bool = False) -> None:
    rows = read_split_rows(archive, "val")
    if channels_last:
        for source in sources.values():
            source["model"].to(memory_format=torch.channels_last)
    loader = DataLoader(CXRArchiveDataset(archive, "val", rows=rows, input_size=384,
                        preprocessing="black_pad_v1"), batch_size=16, num_workers=0, shuffle=False)
    by_image = {name: [] for name in NAMES}
    keys = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to("cuda", non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            keys.extend(zip(batch["subject_id"], batch["study_id"]))
            for name in NAMES:
                outputs = []
                step = 8 if name in {"M02", "N01_EMA"} else 16
                for chunk in images.split(step):
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs.append(torch.sigmoid(sources[name]["model"](chunk)).float().cpu().numpy())
                by_image[name].append(np.concatenate(outputs, axis=0))
    loader.dataset.close()
    if len(keys) != len(rows) or len(keys) != 1422:
        raise ValueError("Validation image coverage differs")
    study_keys = sorted(set(keys))
    if len(study_keys) != 859:
        raise ValueError("Validation Study coverage differs")
    indices = defaultdict(list)
    for i, key in enumerate(keys): indices[key].append(i)
    model_study = {}
    per_model = {}
    reference_labels = None
    for name in NAMES:
        image_probs = np.concatenate(by_image[name], axis=0)
        model_study[name] = np.stack([image_probs[indices[key]].astype(np.float64).mean(axis=0) for key in study_keys])
        original, labels = reference_matrix(name, manifest)
        if set(original) != set(study_keys): raise ValueError(f"Frozen Val ID mismatch: {name}")
        if reference_labels is None:
            reference_labels = {key: labels[i] for i, key in enumerate(original)}
        ref = np.stack([original[key] for key in study_keys])
        delta = np.abs(model_study[name] - ref)
        y = np.stack([reference_labels[key] for key in study_keys])
        auc_old = [float(binary_auc(y[:, c], ref[:, c])) for c in range(10)]
        auc_new = [float(binary_auc(y[:, c], model_study[name][:, c])) for c in range(10)]
        per_model[name] = {"max_abs_probability_diff": float(delta.max()),
            "studies_above_1e-4": int((delta.max(axis=1) > MODEL_PROB_ATOL).sum()),
            "old_macro_auc": float(np.mean(auc_old)), "new_macro_auc": float(np.mean(auc_new)),
            "max_class_auc_diff": float(np.max(np.abs(np.array(auc_old) - auc_new)))}
    y = np.stack([reference_labels[key] for key in study_keys])
    results = {}
    for candidate, names, ref_path in (
        ("E03", NAMES[:4], ROOT / "outputs/submissions/E03_20261017/val_predictions.csv"),
        ("E08_EMA", NAMES, ROOT / "outputs/submissions/E08_EMA_20261020/val_predictions.csv")):
        combined = frozen_fusion(model_study, names)
        if not ref_path.exists():
            if candidate == "E03":
                originals = {name: np.stack([reference_matrix(name, manifest)[0][key] for key in study_keys])
                             for name in names}
                reference = frozen_fusion(originals, names)
            else: raise FileNotFoundError(ref_path)
        else:
            with ref_path.open(encoding="utf-8-sig", newline="") as f: records = list(csv.DictReader(f))
            lookup = {(r["Subject_id"], r["Study_id"]): np.array([float(r[f"probability_{c}"]) for c in range(1,11)]) for r in records}
            if set(lookup) != set(study_keys): raise ValueError(f"Frozen {candidate} Study ID mismatch")
            reference = np.stack([lookup[key] for key in study_keys])
        auc_old = [float(binary_auc(y[:, c], reference[:, c])) for c in range(10)]
        auc_new = [float(binary_auc(y[:, c], combined[:, c])) for c in range(10)]
        delta = np.abs(combined - reference)
        results[candidate] = {"study_count": len(study_keys), "max_abs_probability_diff": float(delta.max()),
            "studies_above_1e-4": int((delta.max(axis=1)>FUSION_PROB_ATOL).sum()),
            "old_macro_auc": float(np.mean(auc_old)), "new_macro_auc": float(np.mean(auc_new)),
            "old_per_class_auc": auc_old, "new_per_class_auc": auc_new,
            "max_class_auc_diff": float(np.max(np.abs(np.array(auc_old)-auc_new)))}
    prob_tol = 1e-3 if channels_last else MODEL_PROB_ATOL
    auc_tol = 1e-4 if channels_last else AUC_ATOL
    accepted = (all(x["max_abs_probability_diff"] <= prob_tol and
                    x["max_class_auc_diff"] <= auc_tol for x in per_model.values()) and
                all(x["max_abs_probability_diff"] <= prob_tol and
                    x["max_class_auc_diff"] <= auc_tol for x in results.values()))
    report = {"accepted": bool(accepted), "predeclared_tolerances": {
        "per_model_probability_abs": prob_tol, "fusion_probability_abs": prob_tol,
        "per_class_auc_abs": auc_tol}, "validation_images": len(keys), "validation_studies": len(study_keys),
        "channels_last": channels_last,
        "model_batch_sizes": {name: (8 if name in {"M02","N01_EMA"} else 16) for name in NAMES},
        "per_model": per_model, "ensembles": results,
        "checkpoint_sha256": {name: sources[name]["checkpoint_sha256"] for name in NAMES},
        "comparison_source": "frozen independent validation_analysis/study_predictions.csv and frozen E08 Val matrix",
        "method": "Each image is decoded/preprocessed once, then each model uses its original validation batch size; Study Mean per model before fixed fusion"}
    filename = "prediction_equivalence_channels_last.json" if channels_last else "prediction_equivalence.json"
    (OUT / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": accepted, "per_model": per_model,
                      "ensembles": {k: {a:b for a,b in v.items() if 'auc' not in a or a in ('old_macro_auc','new_macro_auc')}
                                    for k,v in results.items()}}, ensure_ascii=False))


def probe_options(archive: Path, sources: dict) -> None:
    """Cheap gates only; an option is not deployable without full-Val validation."""
    selected = choose_images(read_split_rows(archive, "val"))
    with zipfile.ZipFile(archive) as zf:
        inputs = [decode_and_transform(zf, row) for row in selected]
    base = []
    pin = []
    for i in range(48):
        tensor = inputs[i % len(inputs)]
        for mode, dest in (("pageable", base), ("pin_plus_copy", pin)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            moved = tensor.to("cuda") if mode == "pageable" else tensor.pin_memory().to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            if i >= 8: dest.append((time.perf_counter() - start) * 1000)
            del moved
    standard = defaultdict(list)
    optimized = defaultdict(list)
    standard_probs = {}
    optimized_probs = {}
    with torch.inference_mode():
        for name in NAMES:
            model = sources[name]["model"]
            for format_name, timings, saved in (
                ("standard_before", standard[name], standard_probs),
                ("channels_last", optimized[name], optimized_probs),
                ("standard_after", standard[name], standard_probs)):
                channels_last = format_name == "channels_last"
                model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
                for turn in range(3):
                    for i, tensor in enumerate(inputs):
                        gpu = tensor.to("cuda")
                        if channels_last:
                            gpu = gpu.contiguous(memory_format=torch.channels_last)
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16):
                            pred = torch.sigmoid(model(gpu)).float().cpu().numpy()
                        torch.cuda.synchronize()
                        if turn > 0: timings.append((time.perf_counter()-t)*1000)
                        saved[(name, i)] = pred
    max_diff = max(float(np.max(np.abs(standard_probs[key]-optimized_probs[key]))) for key in standard_probs)
    result = {"fixed_images": len(inputs), "device": torch.cuda.get_device_name(0),
              "pin_memory_including_pin_call": {"pageable": stat(base), "pinned": stat(pin)},
              "channels_last": {"standard_forward": {k:stat(v) for k,v in standard.items()},
                                "channels_last_forward": {k:stat(v) for k,v in optimized.items()},
                                "total_standard_mean_ms": sum(stat(v)["mean_ms"] for v in standard.values()),
                                "total_channels_last_mean_ms": sum(stat(v)["mean_ms"] for v in optimized.values()),
                                "max_probability_abs_diff": max_diff,
                                "status": "SAMPLE_ONLY_NOT_DEPLOYABLE"},
              "inference_mode_fp16": "Already used by frozen baseline; no new change",
              "full_validation_required_before_adoption": True}
    (OUT / "option_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pin":result["pin_memory_including_pin_call"],
                      "channels_last_total":(result["channels_last"]["total_standard_mean_ms"],
                                             result["channels_last"]["total_channels_last_mean_ms"]),
                      "max_probability_abs_diff":max_diff}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "validate", "probe", "validate_cl"))
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    archive = Path(os.environ["CXR_ARCHIVE"])
    if not archive.is_file(): raise FileNotFoundError(archive)
    manifest, sources = load_sources()
    if args.mode == "benchmark": benchmark(archive, sources)
    elif args.mode == "validate": validate(archive, manifest, sources)
    elif args.mode == "validate_cl": validate(archive, manifest, sources, channels_last=True)
    else: probe_options(archive, sources)


if __name__ == "__main__": main()
