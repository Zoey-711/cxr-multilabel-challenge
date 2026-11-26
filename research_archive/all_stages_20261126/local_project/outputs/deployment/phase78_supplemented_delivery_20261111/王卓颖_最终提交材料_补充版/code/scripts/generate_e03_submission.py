"""Reuse sealed E01 probabilities, infer frozen M02, and seal all-class E03."""

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import sha256, verify_candidate, write_bundle
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}
VALIDATION_AUC = 0.8099309507263751
FIELDS = [f"probability_{index}" for index in range(1, 11)]


def table(predictions, *, image: bool):
    return [{"Subject_id": item.subject_id, "Study_id": item.study_id,
             **({"Image_id": item.image_id} if image else {}),
             **{field: repr(value) for field, value in zip(FIELDS, item.probabilities)}}
            for item in predictions]


def read_e01_matrix(path: Path, expected_keys: set[tuple[str, str]]):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("Subject_id", "Study_id", *FIELDS):
            raise ValueError("E01 matrix has unexpected columns")
        rows = list(reader)
    lookup = {}
    for row in rows:
        key = row["Subject_id"], row["Study_id"]
        if key in lookup:
            raise ValueError(f"Repeated E01 Study: {key}")
        vector = tuple(float(row[field]) for field in FIELDS)
        if any(not 0 <= value <= 1 for value in vector):
            raise ValueError(f"Invalid E01 probability: {key}")
        lookup[key] = vector
    if len(rows) != 1414 or set(lookup) != expected_keys:
        raise ValueError("E01 matrix is incomplete or Study IDs differ")
    return lookup, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen M02 runtime config")
    parser.add_argument("--version", default="E03_20261017")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    destination = args.output_root / args.version
    if destination.exists():
        raise FileExistsError(f"Submission version exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("E03 inference requires CUDA GPU")
    m02 = load_config(args.config)
    if m02 != load_config(RUNS["M02"] / "config.yaml"):
        raise ValueError("M02 runtime configuration mismatch")
    archive = archive_path(m02)
    rows = read_split_rows(archive, "test")
    if len(rows) != 2218 or any("Predict_class" in row for row in rows):
        raise ValueError("Test metadata count or label protocol differs")
    image_keys = {(row["Subject_id"], row["Study_id"], row["Dicom_id"]) for row in rows}
    counts = Counter((row["Subject_id"], row["Study_id"]) for row in rows)
    expected_keys = set(counts)
    if len(image_keys) != 2218 or len(expected_keys) != 1414:
        raise ValueError("Expected 2,218 unique images and 1,414 unique Studies")
    frozen_trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_m02 = json.loads((RUNS["M02"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    model_info = {}
    for name, run in RUNS.items():
        frozen = frozen_m02 if name == "M02" else frozen_trio[name]
        config = load_config(run / "config.yaml")
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError(f"Frozen run incomplete: {name}")
        if archive_path(config) != archive or config["validation"]["study_fusion"] != "mean_probability" or \
                config["data"].get("preprocessing", "black_pad_v1") != "black_pad_v1":
            raise ValueError(f"Archive, Mean aggregation, or deterministic preprocessing mismatch: {name}")
        for artifact in ("best_model.pt", "config.yaml", "best_metrics.json",
                         "validation_analysis/study_predictions.csv"):
            if sha256(run / artifact) != frozen[artifact]["sha256"]:
                raise ValueError(f"Frozen SHA-256 mismatch: {name}/{artifact}")
        model_info[name] = {"version": run.name, "model_name": config["model"]["name"],
                            "checkpoint_sha256": frozen["best_model.pt"]["sha256"],
                            "checkpoint_size_bytes": (run / "best_model.pt").stat().st_size,
                            "preprocessing": "black_pad_v1", "study_aggregation": "mean_probability",
                            "test_probability_source": "reused sealed E01 matrix" if name != "M02" else "fresh M02 inference"}
    source = args.output_root / "E01_20261016"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["version"] != "E01_20261016" or \
            source_manifest["model_versions"] != ["B01", "B02", "B03"] or \
            source_manifest["submission_policy"] != "all_classes":
        raise ValueError("Sealed E01 provenance differs")
    for name in ("B01", "B02", "B03"):
        if source_manifest["checkpoint_sha256"][name] != model_info[name]["checkpoint_sha256"]:
            raise ValueError(f"E01 checkpoint provenance differs: {name}")
    for filename, item in source_manifest["files"].items():
        path = source / filename
        if sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"E01 sealed file changed: {filename}")
    if source_manifest["csv_sha256"] != sha256(source / "submission.csv"):
        raise ValueError("E01 CSV hash changed")
    e01_checked = verify_candidate(source / "probability_matrix.csv", source / "submission.csv",
                                   expected_keys, policy="all_classes")
    if e01_checked["candidate_rows"] != 14140:
        raise ValueError("Sealed E01 submission is incomplete")
    e01, e01_rows = read_e01_matrix(source / "probability_matrix.csv", expected_keys)
    print("PREFLIGHT_OK " + json.dumps({"checkpoints": {n: x["checkpoint_sha256"] for n, x in model_info.items()},
                                        "E01_matrix_sha256": sha256(source / "probability_matrix.csv")}), flush=True)
    model = load_model_state(RUNS["M02"] / "best_model.pt", device="cuda")
    images = predict_archive_images(model, archive,
                                    batch_size=int(m02["training"]["batch_size"]),
                                    num_workers=int(m02["data"]["num_workers"]), device="cuda",
                                    preprocessing=m02["data"].get("preprocessing", "black_pad_v1"))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    if len(images) != 2218 or {(p.subject_id, p.study_id, p.image_id) for p in images} != image_keys:
        raise ValueError("M02 image inference differs from test metadata")
    studies = aggregate_mean_probabilities(images)
    m02_lookup = {(p.subject_id, p.study_id): p for p in studies}
    if len(studies) != 1414 or set(m02_lookup) != expected_keys or \
            any(study.image_count != counts[key] for key, study in m02_lookup.items()):
        raise ValueError("M02 Study Mean aggregation differs from test metadata")
    fused = [StudyPrediction(subject, study, counts[(subject, study)],
                             tuple(0.5 * e01[(subject, study)][j] +
                                   0.5 * m02_lookup[(subject, study)].probabilities[j] for j in range(10)))
             for subject, study in sorted(expected_keys)]
    fusion = {"scheme": "E03", "formula": "0.5*((B01+B02+B03)/3)+0.5*M02",
              "fusion_weights": {"B01": 1 / 6, "B02": 1 / 6, "B03": 1 / 6, "M02": 0.5},
              "source_E01_probability_matrix_sha256": sha256(source / "probability_matrix.csv"),
              "source_E01_manifest_sha256": sha256(source / "manifest.json"),
              "image_to_study": "M02 image Sigmoid probabilities aggregated with arithmetic Mean",
              "submission_policy": "all ten original class IDs per Study; no 0.5 filter; no Gate column",
              "validation_macro_auc": VALIDATION_AUC, "models": model_info,
              "test_archive": str(archive), "test_image_count": len(rows),
              "test_study_count": len(expected_keys)}
    directory, check = write_bundle(args.output_root, args.version, fused, expected_keys, fusion,
                                    {"M02": table(images, image=True)},
                                    {"E01": e01_rows, "M02": table(studies, image=False)},
                                    policy="all_classes", upload_status="generated")
    if check["candidate_rows"] != 14140 or check["candidate_covered_studies"] != 1414:
        raise ValueError("E03 all-class submission coverage failed")
    print(json.dumps({"directory": str(directory.resolve()), "image_count": len(images), **check},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
