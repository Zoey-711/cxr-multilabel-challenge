"""Prepare frozen Fold 1 fusion probabilities; infer Test only after local Fold 2 stops."""

import argparse
import csv
import gc
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.patient_folds import sha256
from cxr.engine.b01_analysis import read_study_predictions
from cxr.engine.metrics import binary_auc
from cxr.inference.aggregate import aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import MATRIX_FIELDS, build_candidate_rows, verify_candidate, write_csv
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


RUNS = {"D17-F1": "d17_f1_20261019", "C17-F1": "c17_f1_20261019",
        "S17-F1": "s17_f1_20261019"}
WEIGHTS = {"D17-F1": 0.25, "C17-F1": 0.25, "S17-F1": 0.50}
EXPECTED_AUC = {"val": 0.806889, "oof": 0.8068045}
FOLD2_STATUS = Path("outputs/analysis/phase17_fold2_pipeline_20261020/status.json")


def source_info(root: Path) -> dict:
    info = {}
    first_archive = None
    first_fold = None
    for model_id, run_id in RUNS.items():
        run = root / "outputs/experiments" / run_id
        frozen = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
        config = load_config(run / "config.yaml")
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete" or frozen["fold"] != 1:
            raise ValueError(f"Source experiment is not a completed Fold 1 run: {model_id}")
        for name in ("config.yaml", "best_model.pt", "validation_analysis/study_predictions.csv",
                     "oof_analysis/study_predictions.csv"):
            path = run / name
            if sha256(path) != frozen["files"][name]["sha256"]:
                raise ValueError(f"Frozen file changed: {model_id}/{name}")
        archive = archive_path(config)
        fold = config["data"]["train_subset"]["sha256"]
        if first_archive is None:
            first_archive, first_fold = archive, fold
        elif archive != first_archive or fold != first_fold:
            raise ValueError("Source models use different archives or patient folds")
        if config["validation"]["study_fusion"] != "mean_probability":
            raise ValueError(f"Study Mean protocol differs: {model_id}")
        info[model_id] = {"run_id": run_id, "checkpoint_sha256": frozen["best_checkpoint_sha256"],
                          "config_sha256": frozen["files"]["config.yaml"]["sha256"],
                          "checkpoint_bytes": (run / "best_model.pt").stat().st_size,
                          "preprocessing": config["data"].get("preprocessing", "black_pad_v1"),
                          "input_size": int(config["data"]["input_size"]),
                          "archive": archive}
    return info


def source_probabilities(root: Path, folder: str, expected_count: int) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    reference_keys = reference_labels = None
    aligned = {}
    for model_id, run_id in RUNS.items():
        path = root / "outputs/experiments" / run_id / folder / "study_predictions.csv"
        probs, labels, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate Study key: {model_id}/{folder}")
        if reference_keys is None:
            reference_keys, reference_labels = keys, labels
            order = np.arange(len(keys))
        else:
            lookup = {key: index for index, key in enumerate(keys)}
            if set(lookup) != set(reference_keys):
                raise ValueError(f"Study alignment differs: {model_id}/{folder}")
            order = [lookup[key] for key in reference_keys]
            if not np.array_equal(labels[order], reference_labels):
                raise ValueError(f"Target alignment differs: {model_id}/{folder}")
        aligned[model_id] = probs[order]
    if len(reference_keys) != expected_count or reference_labels.shape != (expected_count, 10):
        raise ValueError(f"Unexpected Study/class count in {folder}")
    fused = sum(WEIGHTS[model_id] * aligned[model_id] for model_id in RUNS)
    if not np.isfinite(fused).all() or np.any((fused < 0) | (fused > 1)):
        raise ValueError(f"Invalid fused probabilities in {folder}")
    return reference_keys, fused, reference_labels


def macro_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if any(np.unique(labels[:, index]).size != 2 for index in range(10)):
        raise ValueError("All ten classes require positive and negative support")
    return float(np.mean([binary_auc(labels[:, i], probabilities[:, i]) for i in range(10)]))


def probability_rows(keys, matrix):
    return [{"Subject_id": subject, "Study_id": study,
             **{f"probability_{index}": repr(float(value)) for index, value in enumerate(vector, 1)}}
            for (subject, study), vector in zip(keys, matrix)]


def prepare(root: Path, target: Path, config_path: Path) -> None:
    if config_path.resolve() != (root / "configs/f17_densenet121_fold1_384.yaml").resolve():
        raise ValueError("Expected frozen D17 Fold 1 config")
    if target.exists():
        raise FileExistsError(f"Candidate directory already exists: {target}")
    sources = source_info(root)
    metadata = read_split_rows(next(iter(sources.values()))["archive"], "test")
    if len(metadata) != 2218 or any("Predict_class" in row for row in metadata):
        raise ValueError("Unexpected Test metadata or hidden labels")
    counts = Counter((row["Subject_id"], row["Study_id"]) for row in metadata)
    if len(counts) != 1414 or len({(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in metadata}) != 2218:
        raise ValueError("Test does not contain exactly 2218 images in 1414 Studies")
    splits = {}
    for split, folder, expected in (("val", "validation_analysis", 859), ("oof", "oof_analysis", 3687)):
        keys, probabilities, labels = source_probabilities(root, folder, expected)
        auc = macro_auc(probabilities, labels)
        if abs(auc - EXPECTED_AUC[split]) > 1e-5:
            raise ValueError(f"Frozen {split} AUC changed: {auc}")
        splits[split] = {"keys": keys, "probabilities": probabilities, "macro_auc": auc,
                         "patients": len({subject for subject, _ in keys})}
    target.mkdir(parents=True)
    try:
        for split in ("val", "oof"):
            write_csv(target / f"{split}_predictions.csv", MATRIX_FIELDS,
                      probability_rows(splits[split]["keys"], splits[split]["probabilities"]))
        prep = {"status": "PENDING_TEST_INFERENCE", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "formula": "0.25*D17-F1+0.25*C17-F1+0.50*S17-F1", "weights": WEIGHTS,
                "sources": {name: {key: value for key, value in item.items() if key != "archive"}
                            for name, item in sources.items()},
                "fold_index_sha256": load_config(config_path)["data"]["train_subset"]["sha256"],
                "val": {"macro_auc": splits["val"]["macro_auc"], "studies": 859, "patients": splits["val"]["patients"]},
                "oof": {"macro_auc": splits["oof"]["macro_auc"], "studies": 3687,
                        "patients": splits["oof"]["patients"], "scope": "Fold 1 only; not five-fold OOF"},
                "test": {"images": 2218, "studies": 1414, "labels_used": False},
                "file_sha256": {name: sha256(target / name) for name in ("val_predictions.csv", "oof_predictions.csv")}}
        (target / "preparation_manifest.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Keep the partial directory for inspection; do not overwrite it on retry.
        raise
    print(json.dumps({"status": prep["status"], "val_auc": prep["val"]["macro_auc"],
                      "fold1_oof_auc": prep["oof"]["macro_auc"], "target": str(target)}, ensure_ascii=False))


def read_matrix(path: Path, expected_keys: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
            raise ValueError(f"Invalid matrix header: {path}")
        rows = list(reader)
    lookup = {}
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        values = tuple(float(row[f"probability_{i}"]) for i in range(1, 11))
        if key in lookup or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
            raise ValueError(f"Invalid or duplicate probability: {path}: {key}")
        lookup[key] = values
    if set(lookup) != expected_keys:
        raise ValueError(f"Study coverage differs: {path}")
    return lookup


def finish(root: Path, target: Path, config_path: Path) -> None:
    import torch

    if config_path.resolve() != (root / "configs/f17_densenet121_fold1_384.yaml").resolve():
        raise ValueError("Expected frozen D17 Fold 1 config")
    if (target / "manifest.json").exists():
        raise FileExistsError("Candidate already finalized")
    fold2 = json.loads((root / FOLD2_STATUS).read_text(encoding="utf-8"))
    if fold2["state"] != "complete":
        raise RuntimeError("Fold 2 is still running; refusing concurrent GPU inference")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for missing Test probabilities")
    prep = json.loads((target / "preparation_manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "PENDING_TEST_INFERENCE":
        raise ValueError("Unexpected preparation status")
    for filename, digest in prep["file_sha256"].items():
        if sha256(target / filename) != digest:
            raise ValueError(f"Prepared probability file changed: {filename}")
    sources = source_info(root)
    for name, item in sources.items():
        if prep["sources"][name]["checkpoint_sha256"] != item["checkpoint_sha256"]:
            raise ValueError(f"Source checkpoint changed: {name}")
    archive = next(iter(sources.values()))["archive"]
    metadata = read_split_rows(archive, "test")
    counts = Counter((row["Subject_id"], row["Study_id"]) for row in metadata)
    expected_keys = set(counts)
    image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in metadata}
    if len(metadata) != 2218 or len(expected_keys) != 1414 or len(image_keys) != 2218:
        raise ValueError("Test metadata changed")
    aligned = {}
    for model_id, item in sources.items():
        path = target / f"{model_id.lower().replace('-', '_')}_test_predictions.csv"
        if path.exists():
            aligned[model_id] = read_matrix(path, expected_keys)
            continue
        config = load_config(root / "outputs/experiments" / item["run_id"] / "config.yaml")
        model = load_model_state(root / "outputs/experiments" / item["run_id"] / "best_model.pt", device="cuda")
        images = predict_archive_images(model, archive, batch_size=int(config["training"]["batch_size"]),
                                        num_workers=int(config["data"]["num_workers"]), device="cuda",
                                        input_size=item["input_size"], preprocessing=item["preprocessing"])
        del model
        torch.cuda.empty_cache()
        gc.collect()
        if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
            raise ValueError(f"Test image coverage differs: {model_id}")
        studies = aggregate_mean_probabilities(images)
        lookup = {(p.subject_id, p.study_id): p for p in studies}
        if set(lookup) != expected_keys or any(lookup[key].image_count != counts[key] for key in expected_keys):
            raise ValueError(f"Test Study Mean coverage differs: {model_id}")
        rows = probability_rows(sorted(expected_keys), [lookup[key].probabilities for key in sorted(expected_keys)])
        write_csv(path, MATRIX_FIELDS, rows)
        aligned[model_id] = read_matrix(path, expected_keys)
        print(f"TEST_DONE {model_id} checkpoint_sha256={item['checkpoint_sha256']}", flush=True)
    fused = []
    for key in sorted(expected_keys):
        vector = tuple(sum(WEIGHTS[name] * aligned[name][key][index] for name in RUNS)
                       for index in range(10))
        fused.append((key, vector))
    from cxr.inference.aggregate import StudyPrediction
    studies = [StudyPrediction(key[0], key[1], counts[key], vector) for key, vector in fused]
    matrix, candidate, _ = build_candidate_rows(studies, policy="all_classes")
    if len(matrix) != 1414 or len(candidate) != 14140:
        raise ValueError("Incomplete fused Test probabilities")
    write_csv(target / "test_predictions.csv", MATRIX_FIELDS, matrix)
    from cxr.inference.versioned_submission import CSV_FIELDS
    write_csv(target / "submission.csv", CSV_FIELDS, candidate)
    checked = verify_candidate(target / "test_predictions.csv", target / "submission.csv", expected_keys,
                               policy="all_classes")
    if checked["candidate_rows"] != 14140 or checked["candidate_covered_studies"] != 1414:
        raise ValueError("Official full-ten format validation failed")
    total_bytes = sum(item["checkpoint_bytes"] for item in sources.values())
    report = {**checked, "val_macro_auc": prep["val"]["macro_auc"],
              "fold1_oof_macro_auc": prep["oof"]["macro_auc"], "test_auc": None,
              "checkpoint_total_bytes": total_bytes, "checkpoint_limit_bytes": 500_000_000,
              "size_status": "PASS" if total_bytes <= 500_000_000 else "SIZE_FAIL",
              "end_to_end_latency_ms": None, "latency_status": "LATENCY_UNVERIFIED",
              "deployment_status": "LATENCY_UNVERIFIED", "upload_status": "generated_not_uploaded"}
    (target / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": target.name, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "fusion_weights": WEIGHTS, "sources": prep["sources"],
                "fold_index_sha256": prep["fold_index_sha256"],
                "val_macro_auc": prep["val"]["macro_auc"],
                "fold1_oof_macro_auc": prep["oof"]["macro_auc"], "test_auc": None,
                "test_images": 2218, "test_studies": 1414, "submission_rows": 14140,
                "total_checkpoint_bytes": total_bytes, "latency_status": "LATENCY_UNVERIFIED",
                "upload_status": "generated_not_uploaded",
                "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for path in target.iterdir() if path.is_file() and path.name not in
                          {"manifest.json", "finish_stdout.log", "finish_stderr.log",
                           "priority_finish_stdout.log", "priority_finish_stderr.log"}}}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target": str(target), "submission_sha256": checked["csv_sha256"],
                      "rows": checked["candidate_rows"], "studies": checked["candidate_covered_studies"],
                      "checkpoint_bytes": total_bytes, "latency_status": "LATENCY_UNVERIFIED"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "finish"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/submissions/E17_F1_20261020"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    (prepare if args.stage == "prepare" else finish)(root, target, args.config)


if __name__ == "__main__":
    main()
