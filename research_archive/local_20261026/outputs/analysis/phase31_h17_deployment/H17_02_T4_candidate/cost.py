"""H17-02 official-style timing and same-entry Submission consistency checks."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from predict import HERE, load_config, load_models, predict_image, sha256
from cxr.data.archive import image_member, read_split_rows, split_archive_path


def validate_submission(path: Path) -> dict:
    matrix: dict[tuple[str, str], np.ndarray] = {}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Subject_id", "Study_id", "Predict_class", "Probability"]:
            raise ValueError("Submission needs the exact four official columns")
        for row in reader:
            key = (row["Subject_id"], row["Study_id"])
            cls = int(row["Predict_class"])
            prob = float(row["Probability"])
            if not key[0] or not key[1] or cls not in range(1, 11) or not math.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError("Invalid Submission row")
            if (key, cls) in seen:
                raise ValueError("Duplicate Study/class")
            seen.add((key, cls))
            matrix.setdefault(key, np.full(10, np.nan))[cls - 1] = prob
    if len(matrix) != 1414 or len(seen) != 14140 or any(not np.isfinite(x).all() for x in matrix.values()):
        raise ValueError("Submission lacks 1414 Studies × 10 classes")
    return matrix


def validate_wide(path: Path, submission: dict) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)]:
            raise ValueError("Wide Test probability class order mismatch")
        rows = list(reader)
    if len(rows) != 1414:
        raise ValueError("Wide Test probability Study count mismatch")
    keys = set()
    max_diff = 0.0
    for row in rows:
        key = (row["Subject_id"], row["Study_id"])
        if key in keys or key not in submission:
            raise ValueError("Wide Test probability ID duplicate/missing")
        keys.add(key)
        p = np.array([float(row[f"Prob_{i}"]) for i in range(1, 11)])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid wide Test probability")
        max_diff = max(max_diff, float(np.max(np.abs(p - submission[key]))))
    if keys != set(submission):
        raise ValueError("Wide and Submission IDs differ")
    return {"studies": len(rows), "max_abs_diff": max_diff,
            "status": "PASS" if max_diff == 0 else "FAIL"}


def check_archive(archive: Path, models: dict, config: dict, submission: dict,
                  wide: Path) -> dict:
    rows = read_split_rows(archive, "test")
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in rows)
    selected = sorted((r for r in rows if counts[(r["Subject_id"], r["Study_id"])] == 1),
                      key=lambda r: (r["Subject_id"], r["Study_id"]))[:100]
    if len(rows) != 2218 or len(counts) != 1414 or len(selected) != 100:
        raise ValueError("Test archive coverage or single-image sample invalid")
    differences = []
    with zipfile.ZipFile(split_archive_path(archive, "test")) as source:
        for row in selected:
            key = (row["Subject_id"], row["Study_id"])
            with Image.open(io.BytesIO(source.read(image_member(row, "test")))) as image:
                predicted = predict_image(image, models, config)
            differences.append(np.abs(predicted - submission[key]))
    diff = np.stack(differences)
    return {"source": "official unlabeled Test; first 100 sorted single-image Studies",
            "official_fixed_timing_sample": False, "test_auc": None,
            "sample_studies": 100, "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()), "p99_abs_diff": float(np.percentile(diff, 99)),
            "values_above_1e-4": int(np.sum(diff > 1e-4)),
            "status_1e-4": "PASS" if float(diff.max()) <= 1e-4 else "FAIL",
            "wide_vs_submission": validate_wide(wide, submission)}


def list_images(folder: Path, expected: int) -> list[Path]:
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != expected:
        raise ValueError(f"Expected {expected} images in {folder}, found {len(images)}")
    return images


def time_images(config: dict, models: dict, pictures_100: Path, pictures_5: Path,
                output_dir: Path, official_sample: bool, submission: dict,
                study_index: Path | None) -> dict:
    measured = list_images(pictures_100, 100)
    warmup = list_images(pictures_5, 5)
    for path in warmup:
        predict_image(path, models, config)
        torch.cuda.synchronize()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "cost_predictions.csv"
    if predictions_path.exists() or (output_dir / "cost_report.json").exists():
        raise FileExistsError("Refusing to overwrite existing timing output")
    elapsed = {str(path.relative_to(pictures_100)): [] for path in measured}
    final = {}
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "image"] + [f"Prob_{i}" for i in range(1, 11)])
        for round_id in range(1, 4):
            for path in measured:
                torch.cuda.synchronize()
                start = time.perf_counter()
                probability = predict_image(path, models, config)
                relative = str(path.relative_to(pictures_100))
                writer.writerow([round_id, relative] + [format(float(x), ".17g") for x in probability])
                handle.flush()
                torch.cuda.synchronize()
                elapsed[relative].append((time.perf_counter() - start) * 1000)
                final[relative] = probability
    per_image = [statistics.mean(values) for values in elapsed.values()]
    gpu = torch.cuda.get_device_name()
    report = {"gpu": gpu, "precision": "FP16 autocast", "batch_size": 1,
              "warmup_images": 5, "measured_images": 100, "rounds": 3,
              "scope": "wall clock from image file read through final CSV write; model load/warmup excluded; CUDA synchronized",
              "mean_ms": statistics.mean(per_image), "p50_ms": float(np.percentile(per_image, 50)),
              "p95_ms": float(np.percentile(per_image, 95)), "max_ms": max(per_image),
              "limit_ms": 100.0, "official_sample_attested": official_sample,
              "prediction_consistency": "UNVERIFIED_NO_STUDY_INDEX",
              "latency_status": "T4_UNVERIFIED" if "T4" not in gpu else "TARGET_MEASURED",
              "cost_predictions_sha256": sha256(predictions_path)}
    if study_index:
        with study_index.open(encoding="utf-8-sig", newline="") as handle:
            index = list(csv.DictReader(handle))
        if len(index) != 100 or {row["image"] for row in index} != set(final):
            raise ValueError("Study index does not match 100 timed images")
        grouped = {}
        counts = {}
        for row in index:
            key = (row["Subject_id"], row["Study_id"])
            grouped.setdefault(key, []).append(final[row["image"]])
            counts[key] = int(row["study_image_count"])
        complete = {key: np.mean(values, axis=0) for key, values in grouped.items()
                    if len(values) == counts[key]}
        if not complete or not set(complete).issubset(submission):
            raise ValueError("No complete Study for timing/submission comparison")
        delta = max(float(np.max(np.abs(prob - submission[key]))) for key, prob in complete.items())
        report["complete_studies_compared"] = len(complete)
        report["complete_sample_images_compared"] = sum(len(grouped[key]) for key in complete)
        report["submission_max_abs_diff"] = delta
        report["prediction_consistency"] = (
            "FAIL" if delta > 1e-4 else
            "PASS" if report["complete_sample_images_compared"] == 100 else "PARTIAL_SAMPLE")
    report["deployment_status"] = (
        "T4_UNVERIFIED" if "T4" not in gpu else
        "OFFICIAL_SAMPLE_UNVERIFIED" if not official_sample else
        "LATENCY_FAIL" if report["mean_ms"] > 100 else
        "PREDICTION_UNVERIFIED" if report["prediction_consistency"] != "PASS" else
        "T4_TIMING_AND_SAMPLE_CONSISTENCY_PASS")
    (output_dir / "cost_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    parser.add_argument("--submission", type=Path, default=HERE / "results/submission.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/cost")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check-archive", type=Path)
    mode.add_argument("--pictures-100", type=Path)
    parser.add_argument("--pictures-5", type=Path)
    parser.add_argument("--study-index", type=Path)
    parser.add_argument("--official-sample", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    matrix = validate_submission(args.submission)
    models = load_models(config)
    if args.self_check_archive:
        if args.pictures_5 or args.study_index or args.official_sample:
            parser.error("Timing flags are not valid with --self-check-archive")
        result = check_archive(args.self_check_archive, models, config, matrix,
                               args.submission.parent / "test_predictions.csv")
        output = args.submission.parent / "internal_consistency.json"
        if output.exists():
            parser.error("Refusing to overwrite internal consistency report")
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if not args.pictures_5:
            parser.error("--pictures-5 is required for timing")
        result = time_images(config, models, args.pictures_100, args.pictures_5,
                             args.output_dir, args.official_sample, matrix, args.study_index)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
