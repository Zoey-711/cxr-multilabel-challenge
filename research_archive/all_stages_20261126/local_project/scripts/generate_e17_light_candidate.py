"""Evaluate one predeclared four-model E17 candidate and infer missing Test probabilities."""

import argparse
import csv
import gc
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_phase17_three_fold_ensemble import paired_bootstrap
from analyze_phase17_fold2_fusion import score
from cxr.data.archive import archive_path, read_split_rows
from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import read_study_predictions
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import (
    CSV_FIELDS, MATRIX_FIELDS, build_candidate_rows, verify_candidate, write_csv,
)
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "C17_F1": "c17_f1_20261019",
    "C17_F2": "c17_f2_20261020",
    "C17_F3": "c17_f3_20261021",
    "S17_F1": "s17_f1_20261019",
}
WEIGHTS = {name: 0.25 for name in SOURCES}
OLD_TEST = ROOT / "outputs/submissions/E17_F1_20261020"
OLD_NAMES = {"C17_F1": "c17_f1_test_predictions.csv",
             "S17_F1": "s17_f1_test_predictions.csv"}
PROB = [f"probability_{i}" for i in range(1, 11)]
TARGET = [f"target_{i}" for i in range(1, 11)]


def read_matrix(path, expected):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
            raise ValueError(f"Probability column order differs: {path}")
        rows = list(reader)
    lookup = {}
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        values = tuple(float(row[field]) for field in PROB)
        if key in lookup or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
            raise ValueError(f"Duplicate Study or invalid probability: {path}: {key}")
        lookup[key] = values
    if set(lookup) != expected:
        raise ValueError(f"Test Study coverage differs: {path}")
    return lookup


def matrix_rows(keys, matrix):
    return [{"Subject_id": subject, "Study_id": study,
             **{f"probability_{i}": repr(float(value)) for i, value in enumerate(vector, 1)}}
            for (subject, study), vector in zip(keys, matrix)]


def source_audit():
    sources, reference, labels, aligned, archive = {}, None, None, {}, None
    for name, run_id in SOURCES.items():
        run = ROOT / "outputs/experiments" / run_id
        frozen = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError(f"Incomplete run: {run_id}")
        for filename in ("best_model.pt", "config.yaml", "validation_analysis/study_predictions.csv"):
            path = run / filename
            record = frozen["files"][filename]
            if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen source changed: {run_id}/{filename}")
        fold = int(name[-1])
        if frozen["fold"] != fold:
            raise ValueError(f"Fold mismatch: {run_id}")
        config = load_config(run / "config.yaml")
        if config["validation"]["study_fusion"] != "mean_probability":
            raise ValueError(f"Study Mean protocol changed: {run_id}")
        current_archive = archive_path(config)
        if archive is None:
            archive = current_archive
        elif archive != current_archive:
            raise ValueError("Source models use different competition archives")
        path = run / "validation_analysis/study_predictions.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        if fields[:2] != ["Subject_id", "Study_id"] or fields[4:14] != PROB or fields[14:24] != TARGET:
            raise ValueError(f"Validation class order changed: {path}")
        p, y, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != len(set(keys)) or len(keys) != 859:
            raise ValueError(f"Invalid validation coverage: {run_id}")
        if reference is None:
            reference, labels, order = keys, y, np.arange(len(keys))
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(reference):
                raise ValueError(f"Validation Study mismatch: {run_id}")
            order = [lookup[key] for key in reference]
            if not np.array_equal(y[order], labels):
                raise ValueError(f"Validation labels mismatch: {run_id}")
        aligned[name] = p[order]
        sources[name] = {
            "run_id": run_id, "fold": fold,
            "manifest_sha256": sha256(run / "frozen_manifest.json"),
            "checkpoint_sha256": frozen["best_checkpoint_sha256"],
            "checkpoint_bytes": (run / "best_model.pt").stat().st_size,
            "config_sha256": frozen["files"]["config.yaml"]["sha256"],
            "val_sha256": frozen["files"]["validation_analysis/study_predictions.csv"]["sha256"],
            "input_size": int(config["data"]["input_size"]),
            "preprocessing": config["data"].get("preprocessing", "black_pad_v1"),
            "batch_size": int(config["training"]["batch_size"]),
            "num_workers": int(config["data"]["num_workers"]),
            "fold_index_sha256": frozen["fold_index_sha256"],
        }
    if len({subject for subject, _ in reference}) != 228:
        raise ValueError("Expected 228 validation patients")
    folds = [source["fold_index_sha256"] for source in sources.values()]
    if any(item != folds[0] for item in folds):
        raise ValueError("Frozen patient-fold index differs")
    return sources, reference, labels, aligned, archive


def test_index(archive):
    metadata = read_split_rows(archive, "test")
    if len(metadata) != 2218 or any("Predict_class" in row for row in metadata):
        raise ValueError("Test metadata differs or contains labels")
    images = {(r["Subject_id"], r["Study_id"], r["Dicom_id"]) for r in metadata}
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in metadata)
    if len(images) != 2218 or len(counts) != 1414:
        raise ValueError("Expected 2218 Test images in 1414 Studies")
    return images, counts


def prepare(target):
    if target.exists():
        raise FileExistsError(f"Use a new candidate version: {target}")
    sources, keys, labels, aligned, archive = source_audit()
    images, counts = test_index(archive)
    total = sum(item["checkpoint_bytes"] for item in sources.values())
    if total > 500_000_000:
        raise ValueError(f"Candidate exceeds 500MB: {total}")
    fused = sum(WEIGHTS[name] * aligned[name] for name in SOURCES)
    metrics = score(fused, labels, [key[0] for key in keys])
    e08_dir = ROOT / "outputs/submissions/E08_EMA_20261020"
    e08_manifest = json.loads((e08_dir / "manifest.json").read_text(encoding="utf-8"))
    e08_path = e08_dir / "val_predictions.csv"
    if sha256(e08_path) != e08_manifest["files"]["val_predictions.csv"]["sha256"]:
        raise ValueError("Frozen E08 validation probabilities changed")
    e08 = read_matrix(e08_path, set(keys))
    e08_aligned = np.asarray([e08[key] for key in keys])
    e08_metrics = score(e08_aligned, labels, [key[0] for key in keys])
    if abs(e08_metrics["macro_auc"] - e08_manifest["validation_macro_auc"]) > 1e-10:
        raise ValueError("E08 baseline AUC changed")
    paired = paired_bootstrap(fused, e08_aligned, labels, keys)
    paired["delta_point"] = metrics["macro_auc"] - e08_metrics["macro_auc"]
    target.mkdir(parents=True)
    write_csv(target / "val_predictions.csv", MATRIX_FIELDS, matrix_rows(keys, fused))
    report = {
        "status": "PENDING_TEST_INFERENCE", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "0.25*C17_F1+0.25*C17_F2+0.25*C17_F3+0.25*S17_F1",
        "weights": WEIGHTS, "sources": sources, "validation": metrics,
        "e08_validation_macro_auc": e08_metrics["macro_auc"],
        "paired_vs_e08": paired, "checkpoint_total_bytes": total,
        "test_images": len(images), "test_studies": len(counts), "test_auc": None,
        "size_status": "PASS", "latency_status": "LATENCY_UNVERIFIED",
        "val_predictions_sha256": sha256(target / "val_predictions.csv"),
        "e08_val_sha256": sha256(e08_path),
    }
    (target / "preparation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "val_macro_auc": metrics["macro_auc"],
                      "checkpoint_bytes": total, "delta_vs_e08": paired}, ensure_ascii=False), flush=True)


def finish(target):
    import torch

    if (target / "manifest.json").exists():
        raise FileExistsError("Candidate already finalized")
    prep = json.loads((target / "preparation_manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "PENDING_TEST_INFERENCE":
        raise ValueError("Unexpected preparation status")
    if sha256(target / "val_predictions.csv") != prep["val_predictions_sha256"]:
        raise ValueError("Prepared validation probabilities changed")
    sources, _, _, _, archive = source_audit()
    if sources != prep["sources"]:
        raise ValueError("Frozen model sources changed since preparation")
    image_keys, counts = test_index(archive)
    expected = set(counts)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for missing Test inference")
    old_manifest = json.loads((OLD_TEST / "manifest.json").read_text(encoding="utf-8"))
    matrices = {}
    for name, item in sources.items():
        destination = target / f"{name.lower()}_test_predictions.csv"
        if destination.exists():
            matrices[name] = read_matrix(destination, expected)
            continue
        if name in OLD_NAMES:
            old_name = OLD_NAMES[name]
            old_file = OLD_TEST / old_name
            old_source = old_manifest["sources"][name.replace("_", "-")]
            if old_source["checkpoint_sha256"] != item["checkpoint_sha256"] or \
                    sha256(old_file) != old_manifest["files"][old_name]["sha256"]:
                raise ValueError(f"Cached Test source changed: {name}")
            shutil.copyfile(old_file, destination)
            matrices[name] = read_matrix(destination, expected)
            print(f"REUSED_TEST {name}", flush=True)
            continue
        run = ROOT / "outputs/experiments" / item["run_id"]
        model = load_model_state(run / "best_model.pt", device="cuda")
        # Windows sandbox may forbid DataLoader worker pipes. A single reader
        # preserves the deterministic preprocessing and input order.
        images = predict_archive_images(model, archive, batch_size=item["batch_size"],
                                        num_workers=0, device="cuda",
                                        input_size=item["input_size"],
                                        preprocessing=item["preprocessing"])
        del model
        torch.cuda.empty_cache()
        gc.collect()
        if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
            raise ValueError(f"Test image coverage differs: {name}")
        studies = aggregate_mean_probabilities(images)
        lookup = {(study.subject_id, study.study_id): study for study in studies}
        if set(lookup) != expected or any(lookup[key].image_count != counts[key] for key in expected):
            raise ValueError(f"Test Study Mean coverage differs: {name}")
        keys = sorted(expected)
        write_csv(destination, MATRIX_FIELDS,
                  matrix_rows(keys, [lookup[key].probabilities for key in keys]))
        matrices[name] = read_matrix(destination, expected)
        print(f"TEST_DONE {name} checkpoint_sha256={item['checkpoint_sha256']}", flush=True)
    keys = sorted(expected)
    vectors = [tuple(sum(WEIGHTS[name] * matrices[name][key][class_index] for name in SOURCES)
                     for class_index in range(10)) for key in keys]
    study_rows = [StudyPrediction(key[0], key[1], counts[key], vector)
                  for key, vector in zip(keys, vectors)]
    matrix, candidate, _ = build_candidate_rows(study_rows, policy="all_classes")
    write_csv(target / "test_predictions.csv", MATRIX_FIELDS, matrix)
    write_csv(target / "submission.csv", CSV_FIELDS, candidate)
    checked = verify_candidate(target / "test_predictions.csv", target / "submission.csv",
                               expected, policy="all_classes")
    if checked["candidate_rows"] != 14140 or checked["candidate_covered_studies"] != 1414:
        raise ValueError("Submission coverage failed")
    report = {**checked, "validation_macro_auc": prep["validation"]["macro_auc"],
              "test_auc": None, "checkpoint_total_bytes": prep["checkpoint_total_bytes"],
              "checkpoint_limit_bytes": 500_000_000, "size_status": "PASS",
              "latency_status": "LATENCY_UNVERIFIED", "deployment_status": "LATENCY_UNVERIFIED",
              "upload_status": "generated_not_uploaded"}
    (target / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = [path for path in target.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {"version": target.name, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "formula": prep["formula"], "weights": WEIGHTS, "sources": sources,
                "test_dataloader_num_workers": 0,
                "validation_macro_auc": prep["validation"]["macro_auc"],
                "test_auc": None, "test_images": 2218, "test_studies": 1414,
                "submission_rows": 14140, "checkpoint_total_bytes": prep["checkpoint_total_bytes"],
                "size_status": "PASS", "latency_status": "LATENCY_UNVERIFIED",
                "upload_status": "generated_not_uploaded",
                "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for path in paths}}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "FORMAT_PASS", "submission": str(target / "submission.csv"),
                      "csv_sha256": checked["csv_sha256"], "rows": checked["candidate_rows"],
                      "studies": checked["candidate_covered_studies"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "finish"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    (prepare if args.stage == "prepare" else finish)(target)


if __name__ == "__main__":
    main()
