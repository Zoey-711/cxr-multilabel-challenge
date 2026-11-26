#!/usr/bin/env python3
"""
Phase65 frozen five-model random-sample benchmark.

This is a reproducible RANDOM_TEST_BENCHMARK, not an organizer-provided
official sample test. It calls the frozen Phase65 predict.py unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE / "benchmark"
DEFAULT_LOCK = BASE / "phase65_source"
DEFAULT_SUBMISSION = BASE / "submission_original_E08_EMA.csv"

EXPECTED_SUBMISSION_SHA256 = (
    "9a9846947a544e759cc19c79f6cc8c7fae39508058ee0e3d98d3deb24e8b2a07"
)
EXPECTED_ENGINE_SHA256 = {
    "B02": "286cd4527cf4a69bec719b6679d35ed9de1d696ccccb8f62b754cb72df7be70c",
    "M02": "54cb461deb38d4e744450f5c33fa0e84396b8b36659a3138e5823bd1df84a94e",
}
PROBABILITY_TOLERANCE = 1e-3
LATENCY_THRESHOLD_MS = 100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def read_reference_submission(path: Path) -> dict[tuple[str, str], np.ndarray]:
    fields, rows = read_csv_rows(path)

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    normalized = {norm(name): name for name in fields}

    def choose(label: str, aliases: set[str]) -> str:
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        raise RuntimeError(
            f"Cannot identify {label} column in reference submission. "
            f"Available columns: {fields}"
        )

    subject_col = choose("Subject_id", {"subjectid", "subject"})
    study_col = choose("Study_id", {"studyid", "study"})
    class_col = choose("Predict_class", {"predictclass", "classid", "class", "labelid"})
    probability_col = choose(
        "Probability",
        {"probability", "predictprobability", "score", "prob", "prediction"},
    )

    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (row[subject_col], row[study_col])
        class_id = int(row[class_col])
        value = float(row[probability_col])
        require(1 <= class_id <= 10, f"Invalid class id in reference: {key}, {class_id}")
        require(math.isfinite(value) and 0.0 <= value <= 1.0,
                f"Invalid reference probability: {key}, class {class_id}")
        bucket = grouped.setdefault(key, {})
        require(class_id not in bucket, f"Duplicate reference class: {key}, {class_id}")
        bucket[class_id] = value

    result: dict[tuple[str, str], np.ndarray] = {}
    for key, values in grouped.items():
        require(set(values) == set(range(1, 11)),
                f"Reference Study does not have exactly classes 1..10: {key}")
        result[key] = np.asarray([values[i] for i in range(1, 11)], dtype=np.float64)

    require(len(rows) == 14140, f"Expected 14140 reference rows, found {len(rows)}")
    require(len(result) == 1414, f"Expected 1414 reference Studies, found {len(result)}")
    return result


def verify_samples(
    sample_root: Path,
    pictures_100: Path,
    pictures_5: Path,
    image_index: Path,
) -> tuple[list[Path], list[Path], list[dict[str, str]], dict[str, Any]]:
    manifest_path = sample_root / "sample_manifest.json"
    require(manifest_path.is_file(), f"Missing sample manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(manifest.get("benchmark_type") == "RANDOM_TEST_BENCHMARK",
            "Sample manifest is not RANDOM_TEST_BENCHMARK")
    require(manifest.get("official_sample_attested") is False,
            "Random sample must not be marked as organizer-official")
    require(manifest.get("source_split") == "test", "Unexpected sample source split")
    require(manifest.get("random_seed") == 20261102, "Unexpected sample random seed")
    require(manifest.get("timed_images") == 100, "Sample manifest must specify 100 timed images")
    require(manifest.get("warmup_images") == 5, "Sample manifest must specify 5 warmup images")
    require(manifest.get("batch_size") == 1 and manifest.get("rounds") == 3,
            "Sample manifest benchmark protocol mismatch")

    def image_files(folder: Path, expected: int) -> list[Path]:
        require(folder.is_dir(), f"Missing image folder: {folder}")
        paths = sorted(
            x for x in folder.rglob("*")
            if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        require(len(paths) == expected, f"Expected {expected} images in {folder}, found {len(paths)}")
        require(len({x.name for x in paths}) == expected,
                f"Duplicate image basenames in {folder}")
        return paths

    measured = image_files(pictures_100, 100)
    warm = image_files(pictures_5, 5)
    measured_resolved = {p.resolve() for p in measured}
    warm_resolved = {p.resolve() for p in warm}
    require(not (measured_resolved & warm_resolved), "Warmup and timed samples overlap")

    fields, index_rows = read_csv_rows(image_index)
    expected_fields = ["image", "Subject_id", "Study_id", "study_image_count"]
    require(fields == expected_fields, f"Image index columns differ: {fields}")
    require(len(index_rows) == 100, f"Expected 100 image index rows, found {len(index_rows)}")

    indexed_names = [r["image"] for r in index_rows]
    require(len(set(indexed_names)) == 100, "Duplicate image names in timed index")
    require(set(indexed_names) == {p.name for p in measured},
            "Timed image index does not exactly match pictures_100")

    counts = Counter((r["Subject_id"], r["Study_id"]) for r in index_rows)
    expected_counts: dict[tuple[str, str], int] = {}
    for row in index_rows:
        key = (row["Subject_id"], row["Study_id"])
        count = int(row["study_image_count"])
        require(count >= 1, f"Invalid study_image_count: {key}")
        if key in expected_counts:
            require(expected_counts[key] == count, f"Inconsistent study_image_count: {key}")
        expected_counts[key] = count

    for key, observed_count in counts.items():
        require(observed_count == expected_counts[key],
                f"Selected Study is incomplete: {key}, observed={observed_count}, expected={expected_counts[key]}")

    require(len(counts) == 64, f"Expected 64 complete Studies, found {len(counts)}")

    manifest_files = manifest.get("files", [])
    require(len(manifest_files) == 105, "Sample manifest must hash all 105 images")
    manifest_map = {item["image"]: item for item in manifest_files}
    require(len(manifest_map) == 105, "Duplicate image entries in sample manifest")

    for path in measured + warm:
        item = manifest_map.get(path.name)
        require(item is not None, f"Image absent from sample manifest: {path.name}")
        expected_role = "timed" if path.resolve() in measured_resolved else "warmup"
        require(item.get("role") == expected_role,
                f"Sample role mismatch for {path.name}: {item.get('role')} != {expected_role}")
        actual_hash = sha256(path)
        require(actual_hash == item["sha256"], f"Sample image SHA256 mismatch: {path.name}")

    return measured, warm, index_rows, manifest


def environment_info(torch: Any) -> dict[str, Any]:
    import numpy as np
    import PIL
    import torchvision
    import tensorrt as trt

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    nvidia_smi = None
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except Exception as exc:
        nvidia_smi = f"unavailable: {exc}"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "tensorrt": trt.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "nvidia_smi": nvidia_smi,
    }


def verify_frozen_assets(lock: Path, submission: Path) -> dict[str, Any]:
    manifest_path = lock / "frozen_e08_manifest.json"
    require(manifest_path.is_file(), f"Missing frozen manifest: {manifest_path}")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_records = {}
    for name, meta in frozen["models"].items():
        path = lock / "model" / f"{name}.pt"
        require(path.is_file(), f"Missing model weight: {path}")
        actual = sha256(path)
        require(actual == meta["checkpoint_sha256"], f"Model SHA256 mismatch: {name}")
        require(path.stat().st_size == meta["checkpoint_bytes"], f"Model byte-size mismatch: {name}")
        model_records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "status": "PASS",
        }

    engine_records = {}
    for name, expected_hash in EXPECTED_ENGINE_SHA256.items():
        path = BASE / "tensorrt_engines" / f"{name}_fp32.engine"
        require(path.is_file(), f"Missing TensorRT engine: {path}")
        actual = sha256(path)
        require(actual == expected_hash, f"TensorRT engine SHA256 mismatch: {name}")
        engine_records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "status": "PASS",
        }

    require(submission.is_file(), f"Missing reference submission: {submission}")
    submission_hash = sha256(submission)
    require(submission_hash == EXPECTED_SUBMISSION_SHA256,
            "Original 77.75-point E08-EMA submission SHA256 mismatch")
    # The original submission is independently hash-locked above. The frozen
    # manifest's optional file-record layout is deliberately not assumed here.

    return {
        "frozen_version": frozen.get("version"),
        "fusion_formula": frozen.get("formula"),
        "fusion_weights": frozen.get("fusion_weights"),
        "models": model_records,
        "tensorrt_engines": engine_records,
        "reference_submission": {
            "path": str(submission),
            "bytes": submission.stat().st_size,
            "sha256": submission_hash,
            "status": "PASS",
        },
        "frozen_manifest_sha256": sha256(manifest_path),
    }


def aggregate_studies(
    index_rows: list[dict[str, str]],
    image_probabilities: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in index_rows:
        key = (row["Subject_id"], row["Study_id"])
        grouped[key].append(image_probabilities[row["image"]])
    return {
        key: np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64)
        for key, values in grouped.items()
    }


def compare_to_submission(
    predicted: dict[tuple[str, str], np.ndarray],
    reference: dict[tuple[str, str], np.ndarray],
) -> dict[str, Any]:
    details = []
    errors = []
    binary_changed = []
    ranking_changed = []
    out_of_tolerance = []

    for key in sorted(predicted):
        require(key in reference, f"Study missing from reference submission: {key}")
        p = predicted[key]
        r = reference[key]
        diff = np.abs(p - r)
        max_diff = float(diff.max())
        errors.extend(diff.tolist())

        binary_same = bool(np.array_equal(p >= 0.5, r >= 0.5))
        # Stable sorting gives deterministic class-ID tie handling.
        ranking_same = bool(np.array_equal(
            np.argsort(-p, kind="stable"), np.argsort(-r, kind="stable")
        ))

        if not binary_same:
            binary_changed.append(key)
        if not ranking_same:
            ranking_changed.append(key)
        if max_diff > PROBABILITY_TOLERANCE:
            out_of_tolerance.append(key)

        if (not binary_same) or (not ranking_same) or max_diff > PROBABILITY_TOLERANCE:
            details.append({
                "subject_id": key[0],
                "study_id": key[1],
                "max_probability_error": max_diff,
                "binary_same_at_0_5": binary_same,
                "ranking_same": ranking_same,
            })

    max_error = max(errors) if errors else 0.0
    passed = (
        max_error <= PROBABILITY_TOLERANCE
        and len(binary_changed) == 0
        and len(ranking_changed) == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "studies_compared": len(predicted),
        "classes_compared": len(errors),
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "maximum_probability_error": max_error,
        "mean_absolute_probability_error": float(np.mean(errors)) if errors else 0.0,
        "p95_absolute_probability_error": float(np.percentile(errors, 95)) if errors else 0.0,
        "probability_out_of_tolerance_studies": len(out_of_tolerance),
        "binary_changed_studies": len(binary_changed),
        "ranking_changed_studies": len(ranking_changed),
        "binary_threshold": 0.5,
        "details_for_nonmatching_studies": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase65 frozen five-model random-sample benchmark (not organizer official samples)"
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pictures-100", type=Path, default=DEFAULT_OUT / "samples/pictures_100")
    parser.add_argument("--pictures-5", type=Path, default=DEFAULT_OUT / "samples/pictures_5")
    parser.add_argument("--image-index", type=Path, default=DEFAULT_OUT / "timed_image_index.csv")
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT / "run_portable_retest")
    args = parser.parse_args()

    lock = args.lock.resolve()
    sample_root = args.sample_root.resolve()
    output = args.output.resolve()
    submission_path = args.submission.resolve()

    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    require(lock.is_dir(), f"Missing frozen Phase65 directory: {lock}")

    print("===== Phase65 random benchmark preflight =====", flush=True)
    measured, warm, index_rows, sample_manifest = verify_samples(
        sample_root,
        args.pictures_100.resolve(),
        args.pictures_5.resolve(),
        args.image_index.resolve(),
    )
    frozen_assets = verify_frozen_assets(lock, submission_path)
    reference = read_reference_submission(submission_path)

    sample_keys = {(r["Subject_id"], r["Study_id"]) for r in index_rows}
    require(sample_keys.issubset(reference), "At least one selected Study is missing from reference submission")
    print(f"Samples: {len(measured)} timed + {len(warm)} warmup; {len(sample_keys)} complete Studies", flush=True)
    print("Frozen weights, TensorRT engines, original submission and sample hashes: PASS", flush=True)

    # Import the frozen Phase65 implementation without editing it.
    sys.path.insert(0, str(lock / "src"))
    sys.path.insert(0, str(lock))
    import predict  # type: ignore

    actual_predict_path = Path(predict.__file__).resolve()
    expected_predict_path = (lock / "predict.py").resolve()
    require(actual_predict_path == expected_predict_path,
            f"Wrong predict.py imported: {actual_predict_path}")
    require(getattr(predict, "PRECISION", None) == "PyTorch FP16 autocast + CUDA Graph",
            f"Unexpected precision declaration: {getattr(predict, 'PRECISION', None)}")

    import torch

    require(torch.cuda.is_available(), "CUDA is required; CPU timing is prohibited")
    gpu_name = torch.cuda.get_device_name(0)
    require("T4" in gpu_name.upper(), f"Expected Tesla T4, detected: {gpu_name}")

    output.mkdir(parents=True)
    # Preserve the exact reference file inside the deliverable bundle.
    reference_copy = output / "reference_submission_original_E08_EMA.csv"
    shutil.copy2(submission_path, reference_copy)
    require(sha256(reference_copy) == EXPECTED_SUBMISSION_SHA256,
            "Copied reference submission SHA256 mismatch")

    environment = environment_info(torch)
    write_json(output / "environment.json", environment)

    # Record the frozen implementation and assets used by this run.
    source_records = {}
    for name, path in {
        "cost.py": Path(__file__).resolve(),
        "predict.py": expected_predict_path,
        "frozen_predict.py": (lock / "frozen_predict.py").resolve(),
        "frozen_e08_manifest.json": (lock / "frozen_e08_manifest.json").resolve(),
    }.items():
        source_records[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    frozen_assets["total_checkpoint_bytes"] = sum(
        item["bytes"] for item in frozen_assets["models"].values()
    )
    frozen_assets["total_tensorrt_engine_bytes"] = sum(
        item["bytes"] for item in frozen_assets["tensorrt_engines"].values()
    )
    frozen_assets["total_weights_plus_engines_bytes"] = (
        frozen_assets["total_checkpoint_bytes"]
        + frozen_assets["total_tensorrt_engine_bytes"]
    )
    write_json(output / "model_manifest.json", {
        **frozen_assets,
        "phase65_source_files": source_records,
        "precision": predict.PRECISION,
        "graph_models": ["B01", "B03", "N01_EMA"],
        "tensorrt_models": ["B02", "M02"],
        "benchmark_type": "RANDOM_TEST_BENCHMARK",
        "official_sample_attested": False,
    })

    print("===== Loading frozen Phase65 five-model pipeline =====", flush=True)
    load_start = time.perf_counter()
    state = predict.load_models()
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1000.0

    require(set(state["models"]) == {"B01", "B03", "N01_EMA"},
            f"Unexpected PyTorch model set: {sorted(state['models'])}")
    require(set(state["trt_engines"]) == {"B02", "M02"},
            f"Unexpected TensorRT model set: {sorted(state['trt_engines'])}")
    require(set(state["graphs"]) == {"B01", "B03", "N01_EMA"},
            f"Unexpected CUDA Graph set: {sorted(state['graphs'])}")

    print("===== Five-image warmup (excluded from timing) =====", flush=True)
    warmup_start = time.perf_counter()
    for path in warm:
        probability = np.asarray(predict.predict_image(path, state), dtype=np.float64)
        require(probability.shape == (10,) and np.isfinite(probability).all(),
                f"Invalid warmup prediction: {path.name}")
        require(np.all((probability >= 0.0) & (probability <= 1.0)),
                f"Warmup probability outside [0,1]: {path.name}")
        torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    timings_path = output / "timings_300.csv"
    predictions_path = output / "predictions_300.csv"
    index_by_name = {row["image"]: row for row in index_rows}
    timings: dict[str, list[float]] = {p.name: [] for p in measured}
    round_predictions: dict[int, dict[str, np.ndarray]] = {1: {}, 2: {}, 3: {}}

    print("===== Timed inference: 100 images x 3 rounds, batch=1 =====", flush=True)
    with timings_path.open("x", encoding="utf-8", newline="") as timing_stream, \
         predictions_path.open("x", encoding="utf-8", newline="") as prediction_stream:
        timing_writer = csv.writer(timing_stream)
        prediction_writer = csv.writer(prediction_stream)
        timing_writer.writerow(["round", "image", "elapsed_ms"])
        prediction_writer.writerow(["round", "image"] + [f"Prob_{i}" for i in range(1, 11)])

        for round_id in range(1, 4):
            round_start = time.perf_counter()
            for index, path in enumerate(measured, start=1):
                torch.cuda.synchronize()
                start = time.perf_counter()

                probability = np.asarray(
                    predict.predict_image(path, state), dtype=np.float64
                )
                require(
                    probability.shape == (10,)
                    and np.isfinite(probability).all()
                    and np.all((probability >= 0.0) & (probability <= 1.0)),
                    f"Invalid ten-class prediction: {path.name}",
                )
                torch.cuda.synchronize()

                # Prediction CSV row write/flush is inside the per-image timer.
                prediction_writer.writerow([
                    round_id,
                    path.name,
                    *[format(float(v), ".17g") for v in probability],
                ])
                prediction_stream.flush()
                torch.cuda.synchronize()

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                timing_writer.writerow([round_id, path.name, format(elapsed_ms, ".9f")])
                timing_stream.flush()

                timings[path.name].append(elapsed_ms)
                round_predictions[round_id][path.name] = probability

                if index % 25 == 0 or index == 100:
                    print(
                        f"Round {round_id}/3 [{index}/100] "
                        f"elapsed={(time.perf_counter() - round_start):.1f}s",
                        flush=True,
                    )

    # Study Mean aggregation and consistency audit are outside image latency timing.
    study_prediction_path = output / "study_predictions_300.csv"
    consistency_by_round = {}
    with study_prediction_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "round", "Subject_id", "Study_id", "image_count",
            *[f"Prob_{i}" for i in range(1, 11)],
        ])
        for round_id in range(1, 4):
            studies = aggregate_studies(index_rows, round_predictions[round_id])
            consistency_by_round[round_id] = compare_to_submission(studies, reference)
            for key in sorted(studies):
                values = studies[key]
                writer.writerow([
                    round_id, key[0], key[1], len([
                        row for row in index_rows
                        if (row["Subject_id"], row["Study_id"]) == key
                    ]),
                    *[format(float(v), ".17g") for v in values],
                ])

    per_image_mean = [
        statistics.mean(timings[path.name]) for path in measured
    ]
    all_single_run = [v for values in timings.values() for v in values]
    mean_ms = float(statistics.mean(per_image_mean))
    latency_status = "WITHIN_100MS" if mean_ms <= LATENCY_THRESHOLD_MS else "OVER_100MS"

    consistency_passed = all(
        item["status"] == "PASS" for item in consistency_by_round.values()
    )
    consistency_report = {
        "benchmark_type": "RANDOM_TEST_BENCHMARK",
        "official_sample_attested": False,
        "reference_submission": str(submission_path),
        "reference_submission_sha256": sha256(submission_path),
        "aggregation": "arithmetic mean of image probabilities within each complete Study",
        "comparison": "each round's Study Mean probabilities vs original E08-EMA submission",
        "rounds": {str(k): v for k, v in consistency_by_round.items()},
        "overall_status": "PASS" if consistency_passed else "FAIL",
        "overall_passed": consistency_passed,
    }
    write_json(output / "consistency_report.json", consistency_report)

    per_image_rows = [
        {
            "image": path.name,
            "mean_ms": statistics.mean(timings[path.name]),
            "min_ms": min(timings[path.name]),
            "max_ms": max(timings[path.name]),
        }
        for path in measured
    ]
    per_image_rows.sort(key=lambda row: row["mean_ms"], reverse=True)
    with (output / "per_image_latency_summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mean_ms", "min_ms", "max_ms"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Re-hash frozen assets after inference to prove the run did not alter them.
    post_run_assets = verify_frozen_assets(lock, submission_path)
    require(
        post_run_assets["models"] == frozen_assets["models"]
        and post_run_assets["tensorrt_engines"] == frozen_assets["tensorrt_engines"],
        "Frozen assets changed during benchmark",
    )

    report = {
        "candidate": "E08_PHASE65_LOCKED",
        "benchmark_type": "RANDOM_TEST_BENCHMARK",
        "official_sample_attested": False,
        "official_compliance": "NOT_CLAIMED_RANDOM_SAMPLE_ORGANIZER_RETEST_REQUIRED",
        "gpu": gpu_name,
        "batch_size": 1,
        "warmup_images": 5,
        "timed_images": 100,
        "rounds": 3,
        "records": 300,
        "study_count": len(sample_keys),
        "timer_scope": (
            "image file open/decode + preprocessing + all five model forward passes "
            "+ E08 fusion + prediction CSV row write/flush + CUDA synchronization; "
            "model loading and warmup excluded"
        ),
        "aggregation_for_latency": (
            "per-image mean of three runs, then arithmetic mean across 100 images"
        ),
        "model_loading_ms_excluded": model_load_ms,
        "five_image_warmup_ms_excluded": warmup_ms,
        "per_image_then_overall_mean_ms": mean_ms,
        "p50_ms": float(np.percentile(per_image_mean, 50)),
        "p95_ms": float(np.percentile(per_image_mean, 95)),
        "max_per_image_mean_ms": float(max(per_image_mean)),
        "max_single_run_ms": float(max(all_single_run)),
        "min_per_image_mean_ms": float(min(per_image_mean)),
        "threshold_ms": LATENCY_THRESHOLD_MS,
        "latency_status": latency_status,
        "precision": predict.PRECISION,
        "models": {
            "pytorch_cuda_graph_fp16": ["B01", "B03", "N01_EMA"],
            "tensorrt_fp32": ["B02", "M02"],
        },
        "total_checkpoint_bytes": frozen_assets["total_checkpoint_bytes"],
        "total_tensorrt_engine_bytes": frozen_assets["total_tensorrt_engine_bytes"],
        "total_weights_plus_engines_bytes": frozen_assets["total_weights_plus_engines_bytes"],
        "fusion_formula": frozen_assets["fusion_formula"],
        "prediction_consistency": consistency_report["overall_status"],
        "prediction_consistency_is_informational": True,
        "prediction_consistency_by_round": {
            str(k): v["status"] for k, v in consistency_by_round.items()
        },
        "timings_sha256": sha256(timings_path),
        "predictions_sha256": sha256(predictions_path),
        "study_predictions_sha256": sha256(study_prediction_path),
        "sample_manifest_sha256": sha256(sample_root / "sample_manifest.json"),
        "timed_image_index_sha256": sha256(args.image_index.resolve()),
        "reference_submission_sha256": sha256(submission_path),
        "model_assets_unchanged_after_run": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "report.json", report)

    readme = f"""# Phase65 Random-Sample Benchmark

- Benchmark type: `RANDOM_TEST_BENCHMARK`
- Organizer official sample: **No**. These are reproducibly selected images from the competition Test split.
- GPU: `{gpu_name}`
- Timed images: 100
- Warmup images: 5
- Rounds: 3
- Batch size: 1
- Selected complete Studies: {len(sample_keys)}
- Total checkpoint bytes: {frozen_assets["total_checkpoint_bytes"]}
- Total TensorRT engine bytes: {frozen_assets["total_tensorrt_engine_bytes"]}
- Total weights + engine bytes: {frozen_assets["total_weights_plus_engines_bytes"]}
- Seed: 20261102
- Phase65 weights and TensorRT engines: frozen and SHA256-verified before and after the run.

## Results

- Mean latency: {mean_ms:.6f} ms
- P50: {float(np.percentile(per_image_mean, 50)):.6f} ms
- P95: {float(np.percentile(per_image_mean, 95)):.6f} ms
- Maximum per-image three-run mean: {max(per_image_mean):.6f} ms
- Maximum single run: {max(all_single_run):.6f} ms
- 100 ms threshold status: `{latency_status}`
- Submission consistency against the original E08-EMA submission: `{consistency_report['overall_status']}`

## Timing scope

Image file open/decode, preprocessing, all five model forward passes, original E08 fusion,
prediction CSV row write/flush, and CUDA synchronization. Model loading and warmup are excluded.
Latency summary is computed by averaging the three measurements for each image first,
then averaging those 100 per-image means.

## Important limitation

This is a random-sample performance benchmark, not an organizer-provided official sample
test and not an organizer-issued compliance result. Final official timing/compliance remains
subject to organizer retesting.

## Files

- `timings_300.csv`: 300 image-level timing records.
- `predictions_300.csv`: image-level ten-class probabilities for all three rounds.
- `study_predictions_300.csv`: per-round Study Mean predictions.
- `per_image_latency_summary.csv`: per-image three-run latency summary.
- `consistency_report.json`: round-by-round comparison against the original 77.75-point E08-EMA submission.
- `environment.json`: runtime and GPU environment.
- `model_manifest.json`: model, engine, frozen source and reference submission hashes.
- `reference_submission_original_E08_EMA.csv`: exact SHA256-verified reference copy.
- `report.json`: consolidated benchmark report.
- `SHA256SUMS`: file integrity manifest.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    # SHA256SUMS is generated last and excludes itself.
    checksum_path = output / "SHA256SUMS"
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("x", encoding="utf-8", newline="") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")

    print("\n===== PHASE65 RANDOM BENCHMARK COMPLETE =====", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\nDelivery directory: {output}", flush=True)
    print("Note: random sample benchmark only; not an organizer official result.", flush=True)


if __name__ == "__main__":
    main()
