"""Create the frozen E08-EMA candidate, reusing E03 Test Study probabilities."""

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
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import CSV_FIELDS, MATRIX_FIELDS, build_candidate_rows, verify_candidate, write_csv
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


RUNS = {
    "B01": "b01_densenet121_384_20261016",
    "B02": "b02_densenet121_wd1e3_20261016_elevated",
    "B03": "b03_densenet121_lr3e5_20261016",
    "M02": "m02_convnext_tiny_384_aug_20261016_elevated",
    "N01_EMA": "n01_efficientnet_v2_s_384_amp_retry_20261018",
}
FUSION_WEIGHTS = {"B01": 1 / 12, "B02": 1 / 12, "B03": 1 / 12, "M02": 0.25, "N01_EMA": 0.5}
EXPECTED_AUC = 0.8116686461316818
E03 = Path("outputs/submissions/E03_20261017")
FOLD2_STATUS = Path("outputs/analysis/phase17_fold2_pipeline_20261020/status.json")


def source_info(root: Path) -> tuple[dict, Path]:
    trio = json.loads((root / "outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "outputs/experiments" / RUNS["N01_EMA"] / "comparison/comparison.json").read_text(encoding="utf-8"))
    if comparison["fusion"]["E08_EMA"] != "0.5*E03+0.5*N01_EMA" or \
            abs(comparison["scores"]["E08_EMA"]["macro_auc"] - EXPECTED_AUC) > 1e-12:
        raise ValueError("Frozen E08-EMA fusion or validation AUC differs")
    info = {}
    common_archive = None
    for name, run_id in RUNS.items():
        run = root / "outputs/experiments" / run_id
        config = load_config(run / "config.yaml")
        frozen = (json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
                  if name in {"M02", "N01_EMA"} else trio[name])
        checkpoint = "best_ema_model.pt" if name == "N01_EMA" else "best_model.pt"
        val_path = ("ema_validation_analysis/study_predictions.csv" if name == "N01_EMA" else
                    "validation_analysis/study_predictions.csv")
        for path in ("config.yaml", checkpoint, val_path):
            if sha256(run / path) != frozen[path]["sha256"]:
                raise ValueError(f"Frozen source changed: {name}/{path}")
        if comparison["source_sha256"][name]["probabilities_sha256"] != sha256(run / val_path):
            raise ValueError(f"Comparison source prediction changed: {name}")
        if name == "N01_EMA" and comparison["source_sha256"][name]["checkpoint_sha256"] != sha256(run / checkpoint):
            raise ValueError("Comparison EMA checkpoint changed")
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError(f"Source run incomplete: {name}")
        if config["validation"]["study_fusion"] != "mean_probability":
            raise ValueError(f"Study Mean aggregation differs: {name}")
        archive = archive_path(config)
        if common_archive is None:
            common_archive = archive
        elif archive != common_archive:
            raise ValueError("Models refer to different official archives")
        info[name] = {"run_id": run_id, "checkpoint": checkpoint,
                      "checkpoint_sha256": frozen[checkpoint]["sha256"],
                      "config_sha256": frozen["config.yaml"]["sha256"],
                      "checkpoint_bytes": (run / checkpoint).stat().st_size,
                      "val_predictions_sha256": frozen[val_path]["sha256"],
                      "preprocessing": config["data"].get("preprocessing", "black_pad_v1"),
                      "input_size": int(config["data"]["input_size"])}
    return info, common_archive


def read_matrix(path: Path, expected: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
            raise ValueError(f"Unexpected matrix columns: {path}")
        rows = list(reader)
    lookup = {}
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        values = tuple(float(row[f"probability_{i}"]) for i in range(1, 11))
        if key in lookup or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
            raise ValueError(f"Duplicate Study or invalid probabilities: {path}: {key}")
        lookup[key] = values
    if set(lookup) != expected:
        raise ValueError(f"Study coverage differs: {path}")
    return lookup


def table(keys, vectors):
    return [{"Subject_id": key[0], "Study_id": key[1],
             **{f"probability_{i}": repr(float(p)) for i, p in enumerate(vector, 1)}}
            for key, vector in zip(keys, vectors)]


def prepare(root: Path, target: Path, config_path: Path) -> None:
    n01 = root / "outputs/experiments" / RUNS["N01_EMA"]
    if config_path.resolve() != (n01 / "config.yaml").resolve():
        raise ValueError("Expected exact frozen N01 runtime config")
    if target.exists():
        raise FileExistsError(f"Candidate directory already exists: {target}")
    info, archive = source_info(root)
    test_rows = read_split_rows(archive, "test")
    if len(test_rows) != 2218 or any("Predict_class" in row for row in test_rows):
        raise ValueError("Official Test metadata differs or includes a hidden label")
    test_keys = {(row["Subject_id"], row["Study_id"]) for row in test_rows}
    if len(test_keys) != 1414:
        raise ValueError("Expected 1414 Test Studies")
    e03 = root / E03
    e03_manifest = json.loads((e03 / "manifest.json").read_text(encoding="utf-8"))
    if e03_manifest["submission_policy"] != "all_classes" or \
            e03_manifest["model_versions"] != ["B01", "B02", "B03", "M02"]:
        raise ValueError("E03 source identity differs")
    for name in ("B01", "B02", "B03", "M02"):
        if e03_manifest["checkpoint_sha256"][name] != info[name]["checkpoint_sha256"]:
            raise ValueError(f"E03 checkpoint provenance differs: {name}")
    if sha256(e03 / "probability_matrix.csv") != e03_manifest["files"]["probability_matrix.csv"]["sha256"] or \
            sha256(e03 / "submission.csv") != e03_manifest["csv_sha256"]:
        raise ValueError("Frozen E03 Test files changed")
    verify_candidate(e03 / "probability_matrix.csv", e03 / "submission.csv", test_keys, policy="all_classes")
    e03_test = read_matrix(e03 / "probability_matrix.csv", test_keys)
    reference = None
    aligned = {}
    for name, item in info.items():
        run = root / "outputs/experiments" / item["run_id"]
        path = run / ("ema_validation_analysis/study_predictions.csv" if name == "N01_EMA" else
                      "validation_analysis/study_predictions.csv")
        probabilities, labels, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != 859 or len(set(keys)) != 859:
            raise ValueError(f"Invalid Val Study coverage: {name}")
        if reference is None:
            reference = (keys, labels)
            order = np.arange(len(keys))
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(reference[0]):
                raise ValueError(f"Val Study mismatch: {name}")
            order = [lookup[key] for key in reference[0]]
            if not np.array_equal(labels[order], reference[1]):
                raise ValueError(f"Val labels mismatch: {name}")
        aligned[name] = probabilities[order]
    val_keys, val_labels = reference
    e01 = (aligned["B01"] + aligned["B02"] + aligned["B03"]) / 3
    e03_val = 0.5 * e01 + 0.5 * aligned["M02"]
    fused = 0.5 * e03_val + 0.5 * aligned["N01_EMA"]
    auc = float(np.mean([binary_auc(val_labels[:, i], fused[:, i]) for i in range(10)]))
    if abs(auc - EXPECTED_AUC) > 1e-10:
        raise ValueError(f"E08-EMA frozen validation AUC differs: {auc}")
    target.mkdir(parents=True)
    write_csv(target / "val_predictions.csv", MATRIX_FIELDS, table(val_keys, fused))
    prep = {"status": "PENDING_TEST_INFERENCE", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "formula": "0.5*E03+0.5*N01_EMA; E03=0.5*E01+0.5*M02; E01=(B01+B02+B03)/3",
            "fusion_weights": FUSION_WEIGHTS, "validation_macro_auc": auc,
            "validation_studies": 859, "test_studies": 1414, "test_images": 2218,
            "test_auc": None, "sources": info,
            "e03_test_matrix_sha256": sha256(e03 / "probability_matrix.csv"),
            "e03_test_manifest_sha256": sha256(e03 / "manifest.json"),
            "val_probabilities_sha256": sha256(target / "val_predictions.csv")}
    (target / "preparation_manifest.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": prep["status"], "val_auc": auc, "target": str(target)}, ensure_ascii=False))


def finish(root: Path, target: Path, config_path: Path) -> None:
    import torch

    n01 = root / "outputs/experiments" / RUNS["N01_EMA"]
    if config_path.resolve() != (n01 / "config.yaml").resolve():
        raise ValueError("Expected exact frozen N01 runtime config")
    if (target / "manifest.json").exists():
        raise FileExistsError("E08-EMA candidate is already finalized")
    fold2 = json.loads((root / FOLD2_STATUS).read_text(encoding="utf-8"))
    if fold2["state"] != "complete":
        raise RuntimeError("Fold 2 training still uses the only GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for missing N01 EMA Test probabilities")
    prep = json.loads((target / "preparation_manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "PENDING_TEST_INFERENCE" or \
            sha256(target / "val_predictions.csv") != prep["val_probabilities_sha256"]:
        raise ValueError("Prepared E08-EMA validation probabilities changed")
    info, archive = source_info(root)
    if info != prep["sources"]:
        raise ValueError("Frozen model sources differ from the prepared manifest")
    e03 = root / E03
    if sha256(e03 / "probability_matrix.csv") != prep["e03_test_matrix_sha256"] or \
            sha256(e03 / "manifest.json") != prep["e03_test_manifest_sha256"]:
        raise ValueError("E03 Test source changed")
    metadata = read_split_rows(archive, "test")
    counts = Counter((row["Subject_id"], row["Study_id"]) for row in metadata)
    expected = set(counts)
    image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in metadata}
    if len(metadata) != 2218 or len(expected) != 1414 or len(image_keys) != 2218:
        raise ValueError("Test metadata coverage differs")
    e03_probs = read_matrix(e03 / "probability_matrix.csv", expected)
    ema_path = target / "n01_ema_test_predictions.csv"
    if ema_path.exists():
        n01_probs = read_matrix(ema_path, expected)
    else:
        config = load_config(n01 / "config.yaml")
        model = load_model_state(n01 / "best_ema_model.pt", device="cuda")
        images = predict_archive_images(model, archive, batch_size=int(config["training"]["batch_size"]),
                                        num_workers=int(config["data"]["num_workers"]), device="cuda",
                                        input_size=info["N01_EMA"]["input_size"],
                                        preprocessing=info["N01_EMA"]["preprocessing"])
        del model
        torch.cuda.empty_cache()
        gc.collect()
        if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
            raise ValueError("N01 EMA Test image coverage differs")
        studies = aggregate_mean_probabilities(images)
        lookup = {(p.subject_id, p.study_id): p for p in studies}
        if set(lookup) != expected or any(lookup[key].image_count != counts[key] for key in expected):
            raise ValueError("N01 EMA Study Mean coverage differs")
        write_csv(ema_path, MATRIX_FIELDS,
                  table(sorted(expected), [lookup[key].probabilities for key in sorted(expected)]))
        n01_probs = read_matrix(ema_path, expected)
    fused = [StudyPrediction(key[0], key[1], counts[key],
                             tuple(0.5 * e03_probs[key][i] + 0.5 * n01_probs[key][i] for i in range(10)))
             for key in sorted(expected)]
    matrix, candidate, _ = build_candidate_rows(fused, policy="all_classes")
    write_csv(target / "test_predictions.csv", MATRIX_FIELDS, matrix)
    write_csv(target / "submission.csv", CSV_FIELDS, candidate)
    check = verify_candidate(target / "test_predictions.csv", target / "submission.csv",
                             expected, policy="all_classes")
    if check["candidate_rows"] != 14140 or check["candidate_covered_studies"] != 1414:
        raise ValueError("E08-EMA official format is incomplete")
    total_bytes = sum(item["checkpoint_bytes"] for item in info.values())
    report = {**check, "validation_macro_auc": prep["validation_macro_auc"], "test_auc": None,
              "total_checkpoint_bytes": total_bytes, "size_limit_bytes": 500_000_000,
              "size_status": "PASS" if total_bytes <= 500_000_000 else "SIZE_FAIL",
              "end_to_end_latency_ms": None, "latency_status": "LATENCY_UNVERIFIED",
              "deployment_status": "LATENCY_UNVERIFIED", "upload_status": "generated_not_uploaded"}
    (target / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": target.name, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "formula": prep["formula"], "fusion_weights": FUSION_WEIGHTS,
                "models": info, "validation_macro_auc": prep["validation_macro_auc"],
                "test_auc": None, "test_images": 2218, "test_studies": 1414,
                "submission_rows": 14140, "total_checkpoint_bytes": total_bytes,
                "latency_status": "LATENCY_UNVERIFIED", "upload_status": "generated_not_uploaded",
                "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for path in target.iterdir() if path.is_file() and path.name not in
                          {"manifest.json", "finish_stdout.log", "finish_stderr.log",
                           "priority_finish_stdout.log", "priority_finish_stderr.log"}}}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": str(target / "submission.csv"), "csv_sha256": check["csv_sha256"],
                      "bytes": check["csv_bytes"], "val_auc": prep["validation_macro_auc"],
                      "size_bytes": total_bytes, "deployment_status": "LATENCY_UNVERIFIED"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "finish"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submissions/E08_EMA_20261020"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    (prepare if args.stage == "prepare" else finish)(root, target, args.config)


if __name__ == "__main__":
    main()
