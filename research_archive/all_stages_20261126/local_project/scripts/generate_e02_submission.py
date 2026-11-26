"""Infer frozen E02 on all test images and create one immutable candidate bundle."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.inference.aggregate import StudyPrediction, aggregate_mean_probabilities
from cxr.inference.predict import predict_archive_images
from cxr.inference.versioned_submission import write_bundle
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M01": Path("outputs/experiments/m01_convnext_tiny_384_20261016"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def preflight(config_path: Path, version: str, root: Path):
    if (root / version).exists():
        raise FileExistsError(f"Submission version already exists: {root / version}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires CUDA")
    base = load_config(config_path)
    archive = archive_path(base)
    rows = read_split_rows(archive, "test")
    if "Predict_class" in rows[0]:
        raise ValueError("Test data must not have labels")
    metadata = {row["Dicom_id"]: (row["Subject_id"], row["Study_id"]) for row in rows}
    keys = {(row["Subject_id"], row["Study_id"]) for row in rows}
    if len(rows) != 2218 or len(metadata) != 2218 or len(keys) != 1414 or \
            any(not all(key) for key in keys):
        raise ValueError("Expected 2,218 unique test images and 1,414 Studies")
    frozen_trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_m01 = json.loads((RUNS["M01"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    run_info = {}
    for name, run in RUNS.items():
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError(f"Frozen run incomplete: {name}")
        config = load_config(run / "config.yaml")
        if name == "B01" and config != base:
            raise ValueError("B01 source config differs from its frozen runtime config")
        if archive_path(config) != archive or config["data"]["input_size"] != 384 or \
                config["validation"]["study_fusion"] != "mean_probability" or \
                config["data"].get("train_augmentation"):
            raise ValueError(f"Preprocessing/archive/Study fusion differs: {name}")
        expected = frozen_m01 if name == "M01" else frozen_trio[name]
        for relative in ("best_model.pt", "config.yaml", "best_metrics.json",
                         "validation_analysis/study_predictions.csv"):
            if sha256(run / relative) != expected[relative]["sha256"]:
                raise ValueError(f"Frozen artifact SHA-256 mismatch: {name} {relative}")
        checkpoint = run / "best_model.pt"
        best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
        run_info[name] = {"version": run.name, "checkpoint": str(checkpoint.resolve()),
                          "checkpoint_sha256": expected["best_model.pt"]["sha256"],
                          "checkpoint_size_bytes": checkpoint.stat().st_size,
                          "validation_macro_auc": best["val"]["macro_auc"],
                          "best_epoch": best["epoch"],
                          "model_name": config["model"]["name"],
                          "input_size": config["data"]["input_size"],
                          "inference_batch_size": config["training"]["batch_size"],
                          "num_workers": config["data"]["num_workers"],
                          "config": config}
    frozen_score = json.loads((RUNS["M01"] / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    e02_auc = frozen_score["schemes"]["M01_Frozen_equal_half"]["macro_auc"]
    if abs(e02_auc - 0.8034412880236189) > 1e-12:
        raise ValueError("Frozen E02 validation AUC differs")
    return archive, rows, metadata, keys, run_info, e02_auc


def image_table(predictions) -> list[dict]:
    return [{"Subject_id": item.subject_id, "Study_id": item.study_id,
             "Image_id": item.image_id,
             **{f"probability_{i}": repr(value) for i, value in enumerate(item.probabilities, 1)}}
            for item in predictions]


def study_table(predictions: list[StudyPrediction]) -> list[dict]:
    return [{"Subject_id": item.subject_id, "Study_id": item.study_id,
             **{f"probability_{i}": repr(value) for i, value in enumerate(item.probabilities, 1)}}
            for item in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen B01 runtime config")
    parser.add_argument("--version", required=True, help="New submission bundle, e.g. E02_20261016")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    parser.add_argument("--empty-study-policy-confirmed", action="store_true",
                        help="Use only after official permission to omit zero-prediction Studies is confirmed")
    parser.add_argument("--official-rule-reference",
                        help="Required with --empty-study-policy-confirmed; cite organizer notice or rule")
    args = parser.parse_args()
    if args.empty_study_policy_confirmed and not (args.official_rule_reference or "").strip():
        parser.error("Official rule reference is required before marking empty-Study omission as confirmed")
    archive, rows, metadata, expected_keys, info, e02_auc = preflight(
        args.config, args.version, args.output_root)
    predictions_by_model = {}
    image_rows_by_model, study_rows_by_model = {}, {}
    for name, details in info.items():
        print(f"INFER {name} checkpoint_sha256={details['checkpoint_sha256']}", flush=True)
        model = load_model_state(details["checkpoint"], device="cuda")
        predictions = predict_archive_images(
            model, archive, batch_size=int(details["inference_batch_size"]),
            num_workers=int(details["num_workers"]), device="cuda")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        seen = {}
        for item in predictions:
            if item.image_id in seen or metadata.get(item.image_id) != (item.subject_id, item.study_id):
                raise ValueError(f"Wrong or duplicate test image mapping for {name}: {item.image_id}")
            seen[item.image_id] = True
        if len(predictions) != len(rows) or set(seen) != set(metadata):
            raise ValueError(f"Incomplete image inference for {name}")
        study_predictions = aggregate_mean_probabilities(predictions)
        if len(study_predictions) != 1414 or \
                {(item.subject_id, item.study_id) for item in study_predictions} != expected_keys:
            raise ValueError(f"Incomplete Study aggregation for {name}")
        predictions_by_model[name] = {(item.subject_id, item.study_id): item for item in study_predictions}
        image_rows_by_model[name] = image_table(predictions)
        study_rows_by_model[name] = study_table(study_predictions)
        print(f"DONE {name} images={len(predictions)} studies={len(study_predictions)}", flush=True)
    fused = []
    for key in sorted(expected_keys):
        items = [predictions_by_model[name][key] for name in RUNS]
        if len({item.image_count for item in items}) != 1:
            raise ValueError(f"Model image counts differ for Study {key}")
        values = tuple(0.5 * ((items[0].probabilities[j] + items[1].probabilities[j] +
                                items[2].probabilities[j]) / 3) +
                       0.5 * items[3].probabilities[j] for j in range(10))
        fused.append(StudyPrediction(key[0], key[1], items[0].image_count, values))
    fusion_config = {"scheme": "E02", "formula": "0.5*((B01+B02+B03)/3)+0.5*M01",
                     "fusion_weights": {"B01": 1 / 6, "B02": 1 / 6, "B03": 1 / 6, "M01": 0.5},
                     "image_to_study": "arithmetic mean of per-image sigmoid probabilities for each model",
                     "submission_policy": "class probability >= 0.5; zero-class Studies omitted in candidate and flagged",
                     "validation_macro_auc": e02_auc,
                     "models": {name: {key: value for key, value in details.items() if key != "config"}
                                for name, details in info.items()},
                     "empty_study_rule_confirmation": args.official_rule_reference,
                     "test_archive": str(archive), "test_image_count": len(rows),
                     "test_study_count": len(expected_keys)}
    directory, checked = write_bundle(args.output_root, args.version, fused, expected_keys,
                                      fusion_config, image_rows_by_model, study_rows_by_model,
                                      empty_study_policy_confirmed=args.empty_study_policy_confirmed)
    print(json.dumps({"directory": str(directory.resolve()), "image_count": len(rows),
                      "study_count": len(expected_keys), **checked}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
