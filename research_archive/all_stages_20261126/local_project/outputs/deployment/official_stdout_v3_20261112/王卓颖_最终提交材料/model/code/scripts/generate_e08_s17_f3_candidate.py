"""Generate the fixed 0.75 frozen E08 + 0.25 local S17-F3 Test candidate."""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_phase17_fold2_fusion import score
from cxr.data.archive import read_split_rows
from cxr.engine.b01_analysis import read_study_predictions
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import CSV_FIELDS, MATRIX_FIELDS, build_candidate_rows, verify_candidate, write_csv
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
E08 = ROOT / "outputs/submissions/E08_EMA_20261020"
S17 = ROOT / "outputs/experiments/s17_f3_20261021"
PROB = [f"probability_{i}" for i in range(1, 11)]
TARGET = [f"target_{i}" for i in range(1, 11)]
IMAGE_FIELDS = ("Subject_id", "Study_id", "Image_id", *PROB)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked(path, sha, size=None):
    if not path.is_file() or (size is not None and path.stat().st_size != size) or digest(path) != sha:
        raise ValueError(f"Frozen source missing or changed: {path}")


def matrix(path, expected=None):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
            raise ValueError(f"Invalid probability columns: {path}")
        rows = list(reader)
    result = {}
    for row in rows:
        key = (row["Subject_id"], row["Study_id"])
        p = tuple(float(row[name]) for name in PROB)
        if key in result or not all(math.isfinite(v) and 0 <= v <= 1 for v in p):
            raise ValueError(f"Invalid or repeated Study: {path} {key}")
        result[key] = p
    if expected is not None and set(result) != expected:
        raise ValueError(f"Study coverage differs: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Candidate directory already exists: {out}")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    e08_manifest = json.loads((E08 / "manifest.json").read_text(encoding="utf-8"))
    s17_manifest = json.loads((S17 / "frozen_manifest.json").read_text(encoding="utf-8"))
    for name in ("submission.csv", "test_predictions.csv", "val_predictions.csv"):
        record = e08_manifest["files"][name]
        checked(E08 / name, record["sha256"], record["bytes"])
    for name in ("best_model.pt", "config.yaml", "validation_analysis/study_predictions.csv"):
        record = s17_manifest["files"][name]
        checked(S17 / name, record["sha256"], record["size_bytes"])
    if s17_manifest["fold"] != 3 or s17_manifest["model"] != "swin_t":
        raise ValueError("S17-F3 source identity mismatch")
    config = load_config(S17 / "config.yaml")
    if config["validation"]["study_fusion"] != "mean_probability" or config["model"]["name"] != "swin_t":
        raise ValueError("S17-F3 inference protocol mismatch")
    metadata = read_split_rows(args.archive, "test")
    image_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"]) for r in metadata}
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in metadata)
    if len(metadata) != 2218 or len(image_keys) != 2218 or len(counts) != 1414:
        raise ValueError("Official Test metadata is incomplete or duplicated")
    test_keys = set(counts)
    e08_test = matrix(E08 / "test_predictions.csv", test_keys)
    with (E08 / "submission.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("Frozen E08 submission header differs")
        old_rows = list(reader)
    old_key_order = [(r["Subject_id"], r["Study_id"], int(r["Predict_class"])) for r in old_rows]
    if len(old_rows) != 14140 or len(set(old_key_order)) != 14140 or set(old_key_order) != {
        (*key, c) for key in test_keys for c in range(1, 11)
    }:
        raise ValueError("Frozen E08 submission does not match official Test keys")
    for row in old_rows:
        key = (row["Subject_id"], row["Study_id"])
        if float(row["Probability"]) != e08_test[key][int(row["Predict_class"]) - 1]:
            raise ValueError("Frozen E08 submission differs from its Test probabilities")
    val_path = S17 / "validation_analysis/study_predictions.csv"
    with val_path.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames or []
    if fields[4:14] != PROB or fields[14:24] != TARGET:
        raise ValueError("S17 validation class order differs")
    s17_val, labels, subjects, studies = read_study_predictions(val_path)
    val_keys = list(zip(subjects, studies))
    if len(val_keys) != 859 or len(set(val_keys)) != 859 or len(set(subjects)) != 228:
        raise ValueError("S17 Val coverage differs")
    e08_val = matrix(E08 / "val_predictions.csv", set(val_keys))
    e08_val_array = np.asarray([e08_val[key] for key in val_keys])
    base_auc = score(e08_val_array, labels, subjects)["macro_auc"]
    if abs(base_auc - e08_manifest["validation_macro_auc"]) > 1e-10:
        raise ValueError("E08 Val AUC differs from frozen manifest")
    fused_val = 0.75 * e08_val_array + 0.25 * s17_val
    val_auc = score(fused_val, labels, subjects)["macro_auc"]
    if abs(val_auc - 0.8145689350883447) > 1e-9:
        raise ValueError(f"Preselected Val formula changed: {val_auc}")
    total_bytes = e08_manifest["total_checkpoint_bytes"] + (S17 / "best_model.pt").stat().st_size + (S17 / "config.yaml").stat().st_size
    if total_bytes > 500_000_000:
        raise ValueError("Model and config exceed 500MB")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    print("PREFLIGHT_PASS: 2218 Test images, 1414 Studies; starting S17-F3 inference", flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; Test inference was not started")
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        model = load_model_state(S17 / "best_model.pt", device="cuda")
        images = predict_archive_images(model, args.archive, batch_size=int(config["training"]["batch_size"]),
                                        num_workers=0, device="cuda", input_size=int(config["data"]["input_size"]),
                                        preprocessing=config["data"].get("preprocessing", "black_pad_v1"))
        torch.cuda.synchronize()
        peak_gpu_bytes = torch.cuda.max_memory_allocated()
        del model
        if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
            raise ValueError("S17 image predictions do not cover official Test images")
        s17_studies = aggregate_mean_probabilities(images)
        s17_lookup = {(s.subject_id, s.study_id): s for s in s17_studies}
        if set(s17_lookup) != test_keys or any(s17_lookup[key].image_count != counts[key] for key in test_keys):
            raise ValueError("S17 Study Mean aggregation differs from official Test metadata")
        image_rows = [{"Subject_id": p.subject_id, "Study_id": p.study_id, "Image_id": p.image_id,
                       **{field: repr(float(v)) for field, v in zip(PROB, p.probabilities)}} for p in images]
        write_csv(out / "s17_f3_test_image_predictions.csv", IMAGE_FIELDS, image_rows)
        s17_rows = [{"Subject_id": key[0], "Study_id": key[1],
                     **{field: repr(float(v)) for field, v in zip(PROB, s17_lookup[key].probabilities)}} for key in sorted(test_keys)]
        write_csv(out / "s17_f3_test_predictions.csv", MATRIX_FIELDS, s17_rows)
        fused_studies = [StudyPrediction(key[0], key[1], counts[key], tuple(
            0.75 * a + 0.25 * b for a, b in zip(e08_test[key], s17_lookup[key].probabilities))) for key in sorted(test_keys)]
        fused_matrix, _, _ = build_candidate_rows(fused_studies, policy="all_classes")
        write_csv(out / "test_predictions.csv", MATRIX_FIELDS, fused_matrix)
        fused_lookup = matrix(out / "test_predictions.csv", test_keys)
        new_rows = [{"Subject_id": subject, "Study_id": study, "Predict_class": class_id,
                     "Probability": repr(float(fused_lookup[(subject, study)][class_id - 1]))}
                    for subject, study, class_id in old_key_order]
        write_csv(out / "submission.csv", CSV_FIELDS, new_rows)
        validation = verify_candidate(out / "test_predictions.csv", out / "submission.csv", test_keys, policy="all_classes")
        with (out / "submission.csv").open("r", encoding="utf-8", newline="") as handle:
            check_rows = list(csv.DictReader(handle))
        if len(check_rows) != 14140 or [
            (r["Subject_id"], r["Study_id"], int(r["Predict_class"])) for r in check_rows
        ] != old_key_order:
            raise ValueError("New CSV differs from official E08 key and row order")
        runtime_seconds = time.perf_counter() - start
        validation.update({"format_status": "PASS", "test_images": 2218, "test_studies": 1414,
                           "test_auc": None, "val_macro_auc": val_auc, "latency_status": "T4_UNVERIFIED",
                           "gpu": device_name, "gpu_peak_allocated_bytes": peak_gpu_bytes,
                           "inference_and_generation_seconds": runtime_seconds})
        (out / "submission_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {"version": out.name, "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "formula": "0.75*frozen_E08_EMA_20261020+0.25*local_S17_F3_best_epoch10",
                    "weights": {"E08_EMA": 0.75, "S17_F3": 0.25},
                    "source_files": {"E08_submission_sha256": digest(E08 / "submission.csv"),
                                     "E08_test_probabilities_sha256": digest(E08 / "test_predictions.csv"),
                                     "E08_val_probabilities_sha256": digest(E08 / "val_predictions.csv"),
                                     "S17_checkpoint_sha256": digest(S17 / "best_model.pt"),
                                     "S17_config_sha256": digest(S17 / "config.yaml"),
                                     "S17_val_probabilities_sha256": digest(val_path),
                                     "script_sha256": digest(Path(__file__))},
                    "checkpoint_total_bytes": e08_manifest["total_checkpoint_bytes"] + (S17 / "best_model.pt").stat().st_size,
                    "checkpoint_and_s17_config_bytes": total_bytes,
                    "checkpoint_limit_bytes": 500_000_000, "size_status": "PASS",
                    "val_macro_auc": val_auc, "e08_val_macro_auc": base_auc,
                    "selection_caution": "Exploratory Val-screened candidate; bootstrap CI crosses zero and multiple-comparison correction did not pass.",
                    "preprocessing": config["data"].get("preprocessing", "black_pad_v1"),
                    "input_size": config["data"]["input_size"], "precision": "PyTorch FP16 AMP batch=8 for S17 Test; frozen E08 Study Test probabilities reused",
                    "study_aggregation": "mean_probability", "test_dataloader_workers": 0,
                    "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": device_name,
                                "peak_gpu_allocated_bytes": peak_gpu_bytes, "seconds": runtime_seconds},
                    "test_auc": None, "latency_status": "T4_UNVERIFIED", "upload_status": "generated_not_uploaded",
                    "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in out.iterdir() if p.is_file()}}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "PASS", "submission": str(out / "submission.csv"),
                          "sha256": validation["csv_sha256"], "rows": validation["candidate_rows"],
                          "studies": validation["candidate_covered_studies"], "val_auc": val_auc,
                          "seconds": runtime_seconds}, ensure_ascii=False), flush=True)
    except Exception:
        (out / "FAILED.txt").write_text("Generation incomplete. Do not submit any CSV in this directory.\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
