"""Recompute every Test image with the packaged six-model T4 entry.

This is an audit of an already submitted CSV. It never writes a new submission.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase65_source"))
sys.path.insert(0, str(ROOT / "phase65_source/src"))

from phase65_source import predict
from cxr.prediction_consistency import (
    aggregate_complete, compare_studies, probabilities, read_index,
    read_submission, sha256, validate_index,
)


def verify_inputs(image_root: Path, index: Path, inventory: Path, submission: Path, config: Path):
    package = json.loads((ROOT / "manifest_sha256.json").read_text(encoding="utf-8"))
    for name, expected in package.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Deployment file missing or changed: {name}")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if cfg["batch"] != 1 or cfg["formula"] != "0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3":
        raise ValueError("Unexpected deployment config")
    if sha256(submission) != cfg["historical_submission_sha256"]:
        raise ValueError("Historical submission differs")
    rows = read_index(index)
    images, studies = validate_index(rows)
    reference = read_submission(submission)
    if len(images) != 2218 or len(studies) != 1414 or set(studies) != set(reference):
        raise ValueError("Full Test mapping does not match submission")
    records = json.loads(inventory.read_text(encoding="utf-8"))
    if len(records) != 2218 or {r["image"] for r in records} != set(images):
        raise ValueError("Image inventory incomplete")
    paths = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.name in images:
            if path.name in paths:
                raise ValueError(f"Duplicate image filename: {path.name}")
            paths[path.name] = path
    if set(paths) != set(images):
        raise ValueError(f"Missing {len(set(images) - set(paths))} Test image files")
    for record in records:
        name = record["image"]
        meta = images[name]
        if (record["Subject_id"], record["Study_id"]) != (meta["Subject_id"], meta["Study_id"]):
            raise ValueError(f"Image mapping changed: {name}")
        path = paths[name]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Original Test image bytes changed: {name}")
    return rows, paths, reference


def run(args):
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Audit output must be outside the immutable deployment package")
    output.mkdir(parents=True, exist_ok=False)
    rows, paths, reference = verify_inputs(
        args.images_root, args.image_index, args.image_inventory,
        args.submission, args.config,
    )
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("The packaged TensorRT engines require an actual T4")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    state = predict.load_models()
    if any(module.training for model in state["models"].values() for module in model.modules()):
        raise ValueError("A PyTorch model is not in eval mode")
    values = {}
    started = time.perf_counter()
    repeat_positions = {0, 221, 442, 663, 884, 1105, 1326, 1547, 1768, 1989}
    repeat_max = 0.0
    with (output / "image_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "Subject_id", "Study_id", *[f"Prob_{n}" for n in range(1, 11)]])
        for position, row in enumerate(rows):
            name = row["image"]
            vector = probabilities(predict.predict_image(paths[name], state))
            if position in repeat_positions:
                again = probabilities(predict.predict_image(paths[name], state))
                repeat_max = max(repeat_max, float(np.max(np.abs(vector - again))))
            values[name] = vector
            writer.writerow([name, row["Subject_id"], row["Study_id"],
                             *[format(float(x), ".17g") for x in vector]])
            if (position + 1) % 200 == 0:
                print(f"FULL_TEST {position + 1}/2218", flush=True)
    torch.cuda.synchronize()
    actual = aggregate_complete(rows, values)
    comparison, details = compare_studies(actual, reference)
    with (output / "study_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id", *[f"Prob_{n}" for n in range(1, 11)]])
        for key, vector in sorted(actual.items()):
            writer.writerow([*key, *[format(float(x), ".17g") for x in vector]])
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    report = {
        "status": "PASS" if comparison["status"] == "PASS" and repeat_max == 0 else "FAIL",
        "comparison": comparison,
        "repeat_max_abs": repeat_max,
        "images": len(values),
        "studies": len(actual),
        "seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": json.loads(args.config.read_text(encoding="utf-8"))["precision"],
        "historical_submission_sha256": sha256(args.submission),
        "scope": "Full Test consistency, not official 100-image latency",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest_sha256.json").write_text(json.dumps({
        path.name: sha256(path) for path in output.iterdir() if path.is_file()
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, default=ROOT / "test_image_index.csv")
    parser.add_argument("--image-inventory", type=Path, default=ROOT / "test_image_inventory.json")
    parser.add_argument("--submission", type=Path, default=ROOT / "submission_original_78_81.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
