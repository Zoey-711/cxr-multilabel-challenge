"""Reuse frozen E03 test probabilities and infer frozen T01 for the fixed E07 candidate."""

import argparse
import csv
import gc
import json
import math
from collections import Counter
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import sha256, verify_candidate, write_bundle
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


E03 = Path("outputs/submissions/E03_20261017")
T01 = Path("outputs/experiments/t01_swin_tiny_384_20261018")
FIELDS = tuple(f"probability_{i}" for i in range(1, 11))


def read_matrix(path: Path, expected: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("Subject_id", "Study_id", *FIELDS):
            raise ValueError("E03 probability matrix header differs")
        rows = list(reader)
    lookup = {}
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        values = tuple(float(row[field]) for field in FIELDS)
        if key in lookup or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Duplicate Study or invalid E03 probability")
        lookup[key] = values
    if len(lookup) != 1414 or set(lookup) != expected:
        raise ValueError("E03 matrix does not cover the expected 1414 Studies")
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Exact frozen T01 runtime config")
    parser.add_argument("--version", default="E07_20261018")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    destination = args.output_root / args.version
    if destination.exists():
        raise FileExistsError(f"Submission version exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("E07 test inference requires CUDA")
    config = load_config(args.config)
    if config != load_config(T01 / "config.yaml") or config["model"]["name"] != "swin_t":
        raise ValueError("T01 config differs from frozen runtime copy")
    if json.loads((T01 / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("T01 training not complete")
    frozen = json.loads((T01 / "frozen_manifest.json").read_text(encoding="utf-8"))
    for name in ("best_model.pt", "config.yaml", "validation_analysis/study_predictions.csv"):
        if sha256(T01 / name) != frozen[name]["sha256"]:
            raise ValueError(f"Frozen T01 file hash mismatch: {name}")
    comparison = json.loads((T01 / "comparison/comparison.json").read_text(encoding="utf-8"))
    val_auc = comparison["scores"]["E07"]["macro_auc"]
    if comparison["fusion"]["E07"] != "0.5*E03+0.5*T01" or abs(val_auc - 0.8115193037842005) > 1e-12:
        raise ValueError("Frozen E07 validation scheme or AUC differs")
    archive = archive_path(config)
    rows = read_split_rows(archive, "test")
    if len(rows) != 2218 or any("Predict_class" in row for row in rows):
        raise ValueError("Test metadata or hidden-label boundary differs")
    image_keys = {(r["Subject_id"], r["Study_id"], r["Dicom_id"]) for r in rows}
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in rows)
    expected = set(counts)
    if len(image_keys) != 2218 or len(expected) != 1414:
        raise ValueError("Expected 2218 distinct images and 1414 distinct Studies")
    source_manifest = json.loads((E03 / "manifest.json").read_text(encoding="utf-8"))
    if (source_manifest["version"] != "E03_20261017" or
            source_manifest["model_versions"] != ["B01", "B02", "B03", "M02"] or
            source_manifest["submission_policy"] != "all_classes"):
        raise ValueError("E03 frozen source identity differs")
    for filename, record in source_manifest["files"].items():
        path = E03 / filename
        if sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Frozen E03 file differs: {filename}")
    if sha256(E03 / "submission.csv") != source_manifest["csv_sha256"]:
        raise ValueError("E03 submission hash differs")
    verify_candidate(E03 / "probability_matrix.csv", E03 / "submission.csv", expected,
                     policy="all_classes")
    e03 = read_matrix(E03 / "probability_matrix.csv", expected)
    for name, sha in source_manifest["checkpoint_sha256"].items():
        run_ids = {"B01": "b01_densenet121_384_20261016",
                   "B02": "b02_densenet121_wd1e3_20261016_elevated",
                   "B03": "b03_densenet121_lr3e5_20261016",
                   "M02": "m02_convnext_tiny_384_aug_20261016_elevated"}
        run = Path("outputs/experiments") / run_ids[name]
        if sha256(run / "best_model.pt") != sha or archive_path(load_config(run / "config.yaml")) != archive:
            raise ValueError(f"E03 source checkpoint/archive mismatch: {name}")
    if config["validation"]["study_fusion"] != "mean_probability" or \
            config["data"].get("preprocessing", "black_pad_v1") != "black_pad_v1":
        raise ValueError("T01 inference preprocessing or aggregation differs")
    print("PREFLIGHT_OK " + json.dumps({"E03_matrix_sha256": sha256(E03 / "probability_matrix.csv"),
                                        "T01_checkpoint_sha256": frozen["best_model.pt"]["sha256"]}), flush=True)
    model = load_model_state(T01 / "best_model.pt", device="cuda")
    images = predict_archive_images(model, archive, batch_size=int(config["training"]["batch_size"]),
                                    num_workers=int(config["data"]["num_workers"]), device="cuda",
                                    preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                    input_size=int(config["data"]["input_size"]))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
        raise ValueError("T01 image predictions do not cover all test metadata")
    studies = aggregate_mean_probabilities(images)
    t01 = {(p.subject_id, p.study_id): p for p in studies}
    if len(t01) != 1414 or set(t01) != expected or \
            any(item.image_count != counts[key] for key, item in t01.items()):
        raise ValueError("T01 Study Mean mapping differs")
    fused = [StudyPrediction(subject, study, counts[(subject, study)],
                             tuple(0.5 * e03[(subject, study)][j] +
                                   0.5 * t01[(subject, study)].probabilities[j] for j in range(10)))
             for subject, study in sorted(expected)]
    model_info = {}
    run_ids = {"B01": "b01_densenet121_384_20261016",
               "B02": "b02_densenet121_wd1e3_20261016_elevated",
               "B03": "b03_densenet121_lr3e5_20261016",
               "M02": "m02_convnext_tiny_384_aug_20261016_elevated",
               "T01": T01.name}
    for name, run_id in run_ids.items():
        run = Path("outputs/experiments") / run_id
        model_info[name] = {"version": run_id,
                            "model_name": load_config(run / "config.yaml")["model"]["name"],
                            "checkpoint_sha256": sha256(run / "best_model.pt"),
                            "checkpoint_size_bytes": (run / "best_model.pt").stat().st_size,
                            "preprocessing": "black_pad_v1", "study_aggregation": "mean_probability",
                            "test_probability_source": "reused sealed E03 matrix" if name != "T01" else "fresh T01 inference"}
    total_bytes = sum(item["checkpoint_size_bytes"] for item in model_info.values())
    if total_bytes > 500_000_000:
        raise ValueError("E07 inference weights exceed 500 MB")
    fusion = {"scheme": "E07", "formula": "0.5*E03+0.5*T01",
              "fusion_weights": {"B01": 1/12, "B02": 1/12, "B03": 1/12, "M02": 1/4, "T01": 1/2},
              "source_E03_probability_matrix_sha256": sha256(E03 / "probability_matrix.csv"),
              "source_E03_manifest_sha256": sha256(E03 / "manifest.json"),
              "image_to_study": "T01 image Sigmoid probabilities aggregated with arithmetic Mean",
              "submission_policy": "all ten original class IDs per Study; no threshold filter; no Gate column",
              "validation_macro_auc": val_auc, "models": model_info,
              "total_checkpoint_bytes": total_bytes, "test_archive": str(archive),
              "test_image_count": len(rows), "test_study_count": len(expected)}
    image_rows = [{"Subject_id": p.subject_id, "Study_id": p.study_id, "Image_id": p.image_id,
                   **{field: repr(value) for field, value in zip(FIELDS, p.probabilities)}} for p in images]
    study_rows = [{"Subject_id": p.subject_id, "Study_id": p.study_id,
                   **{field: repr(value) for field, value in zip(FIELDS, p.probabilities)}} for p in studies]
    with (E03 / "probability_matrix.csv").open(encoding="utf-8", newline="") as handle:
        e03_rows = list(csv.DictReader(handle))
    directory, check = write_bundle(args.output_root, args.version, fused, expected, fusion,
                                    {"T01": image_rows}, {"E03": e03_rows, "T01": study_rows},
                                    policy="all_classes", upload_status="generated")
    if check["candidate_rows"] != 14140 or check["candidate_covered_studies"] != 1414:
        raise ValueError("E07 submission coverage failed")
    print(json.dumps({"directory": str(directory.resolve()), "image_count": len(images),
                      "checkpoint_total_bytes": total_bytes, **check}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
