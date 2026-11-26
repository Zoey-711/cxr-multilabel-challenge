"""Shared T4 wall-clock benchmark. Candidate cost.py imports its own predict.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_files(folder: Path, expected: int) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = sorted(path for path in folder.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} images, found {len(paths)}: {folder}")
    return paths


def read_submission(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Subject_id", "Study_id", "Predict_class", "Probability"]:
            raise ValueError("Submission columns/order invalid")
        rows = list(reader)
    studies = {}
    keys = set()
    for row in rows:
        subject, study = row["Subject_id"], row["Study_id"]
        class_id = int(row["Predict_class"])
        value = float(row["Probability"])
        key = (subject, study, class_id)
        if not subject or not study or class_id not in range(1, 11) or key in keys:
            raise ValueError("Invalid or duplicate submission key")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid submission probability")
        keys.add(key)
        studies.setdefault((subject, study), np.full(10, np.nan))[class_id - 1] = value
    if len(rows) != 14140 or len(studies) != 1414 or any(not np.isfinite(v).all() for v in studies.values()):
        raise ValueError("Submission must contain 1414 Studies x 10 classes")
    return studies


def consistency(index_path: Path | None, measured: list[Path], predictions: dict,
                submission: dict | None) -> dict:
    if submission is None:
        return {"status": "SUBMISSION_CONSISTENCY_UNVERIFIED", "reason": "matching submitted CSV absent"}
    if index_path is None:
        return {"status": "SUBMISSION_CONSISTENCY_UNVERIFIED", "reason": "image-to-Study index absent"}
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image", "Subject_id", "Study_id", "study_image_count"]:
            raise ValueError("Index columns must be image,Subject_id,Study_id,study_image_count")
        rows = list(reader)
    expected = {str(path.name) for path in measured}
    if len(rows) != 100 or {row["image"] for row in rows} != expected:
        raise ValueError("Image index must cover the exact 100 timed images")
    grouped = {}
    counts = {}
    for row in rows:
        key = (row["Subject_id"], row["Study_id"])
        total = int(row["study_image_count"])
        if total < 1 or (key in counts and counts[key] != total):
            raise ValueError("Inconsistent Study image count")
        counts[key] = total
        grouped.setdefault(key, []).append(predictions[row["image"]])
    complete = {key:np.mean(values, axis=0) for key, values in grouped.items()
                if len(values) == counts[key]}
    if not complete:
        return {"status": "SUBMISSION_CONSISTENCY_UNVERIFIED", "reason": "no complete Study among timed images",
                "partial_studies": len(grouped)}
    differences = []
    gate_mismatch = 0
    order_mismatch = 0
    for key, value in complete.items():
        if key not in submission:
            raise ValueError(f"Study absent from submission: {key}")
        truth = submission[key]
        differences.append(np.abs(value - truth))
        gate_mismatch += int(np.any((value >= 0.5) != (truth >= 0.5)))
        order_mismatch += int(not np.array_equal(np.argsort(value, kind="stable"),
                                                np.argsort(truth, kind="stable")))
    diffs = np.concatenate(differences)
    passed = gate_mismatch == 0 and order_mismatch == 0 and float(diffs.max()) <= 1e-3
    return {"status": "PASS" if passed else "FAIL", "complete_studies": len(complete),
            "partial_studies": len(grouped) - len(complete), "max_abs_probability_diff": float(diffs.max()),
            "mean_abs_probability_diff": float(diffs.mean()),
            "gate_mismatch_studies": gate_mismatch, "class_order_mismatch_studies": order_mismatch,
            "normal_float_tolerance": 1e-3}


def run(predict_module, candidate_id: str):
    parser = argparse.ArgumentParser(description="T4 official-style image benchmark")
    parser.add_argument("--pictures-100", type=Path, required=True)
    parser.add_argument("--pictures-5", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--image-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-sample", action="store_true")
    args = parser.parse_args()
    measured = image_files(args.pictures_100, 100)
    warm = image_files(args.pictures_5, 5)
    if {p.resolve() for p in warm} & {p.resolve() for p in measured}:
        raise ValueError("Warmup and measured images overlap")
    if len({p.name for p in measured}) != 100:
        raise ValueError("Timed image basenames must be unique for index matching")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU timing is prohibited")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = predict_module.load_models()
    for path in warm:
        _ = predict_module.predict_image(path, model)
        torch.cuda.synchronize()
    raw_path = output / "timings_300.csv"
    pred_path = output / "predictions_300.csv"
    timings = {p.name: [] for p in measured}
    final = {}
    with raw_path.open("x", encoding="utf-8", newline="") as raw_stream, \
         pred_path.open("x", encoding="utf-8", newline="") as pred_stream:
        raw_writer = csv.writer(raw_stream)
        pred_writer = csv.writer(pred_stream)
        raw_writer.writerow(["round", "image", "elapsed_ms"])
        pred_writer.writerow(["round", "image", *[f"Prob_{i}" for i in range(1, 11)]])
        for round_id in range(1, 4):
            for path in measured:
                torch.cuda.synchronize()
                start = time.perf_counter()
                probability = np.asarray(predict_module.predict_image(path, model), dtype=np.float64)
                if probability.shape != (10,) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                    raise ValueError(f"Invalid ten-class output: {path}")
                pred_writer.writerow([round_id, path.name, *[format(float(v), ".17g") for v in probability]])
                pred_stream.flush()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                raw_writer.writerow([round_id, path.name, format(elapsed_ms, ".9f")])
                raw_stream.flush()
                timings[path.name].append(elapsed_ms)
                final[path.name] = probability
    per_image = [statistics.mean(timings[p.name]) for p in measured]
    submission = read_submission(args.submission) if args.submission else None
    match = consistency(args.image_index, measured, final, submission)
    gpu = torch.cuda.get_device_name()
    report = {"candidate": candidate_id, "gpu": gpu, "batch": 1,
              "warmup_images": 5, "timed_images": 100, "rounds": 3, "records": 300,
              "timer_scope": "file read through prediction CSV write and CUDA synchronization",
              "per_image_then_overall_mean_ms": float(statistics.mean(per_image)),
              "p50_ms": float(np.percentile(per_image, 50)), "p95_ms": float(np.percentile(per_image, 95)),
              "max_ms": float(max(per_image)), "threshold_ms": 100,
              "official_sample_attested": bool(args.official_sample),
              "prediction_consistency": match, "precision": predict_module.PRECISION,
              "timings_sha256": sha256(raw_path), "predictions_sha256": sha256(pred_path),
              "t4_status": "T4_UNVERIFIED" if "T4" not in gpu else "T4_MEASURED",
              "official_compliance": "PENDING_ORGANIZER_RETEST"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

