"""Infer frozen B01/B02/B03 on test images and write the all-class E01 candidate."""

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
}
FROZEN_AUC = 0.8026744715463039


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(predictions, *, image: bool) -> list[dict]:
    return [{"Subject_id": item.subject_id, "Study_id": item.study_id,
             **({"Image_id": item.image_id} if image else {}),
             **{f"probability_{i}": repr(value) for i, value in enumerate(item.probabilities, 1)}}
            for item in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen B01 runtime config")
    parser.add_argument("--version", default="E01_20261016")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    args = parser.parse_args()
    destination = args.output_root / args.version
    if destination.exists():
        raise FileExistsError(f"Submission version already exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("E01 inference requires CUDA")
    base = load_config(args.config)
    archive = archive_path(base)
    rows = read_split_rows(archive, "test")
    if not rows or "Predict_class" in rows[0]:
        raise ValueError("Test metadata unexpectedly contains labels")
    metadata = {row["Dicom_id"]: (row["Subject_id"], row["Study_id"]) for row in rows}
    keys = set(metadata.values())
    if len(rows) != 2218 or len(metadata) != 2218 or len(keys) != 1414:
        raise ValueError("Expected 2,218 distinct images and 1,414 Studies")
    frozen = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    info = {}
    for name, run in RUNS.items():
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError(f"Incomplete frozen experiment: {name}")
        config = load_config(run / "config.yaml")
        if name == "B01" and config != base:
            raise ValueError("B01 runtime config mismatch")
        if archive_path(config) != archive or config["data"]["input_size"] != 384 or \
                config["validation"]["study_fusion"] != "mean_probability" or \
                config["data"].get("train_augmentation"):
            raise ValueError(f"Inconsistent archive or deterministic inference protocol: {name}")
        for artifact in ("best_model.pt", "config.yaml", "best_metrics.json",
                         "validation_analysis/study_predictions.csv"):
            if sha256(run / artifact) != frozen[name][artifact]["sha256"]:
                raise ValueError(f"Frozen SHA-256 mismatch: {name}/{artifact}")
        checkpoint = run / "best_model.pt"
        best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
        info[name] = {"version": run.name, "checkpoint": str(checkpoint.resolve()),
                      "checkpoint_sha256": frozen[name]["best_model.pt"]["sha256"],
                      "checkpoint_size_bytes": checkpoint.stat().st_size,
                      "validation_macro_auc": best["val"]["macro_auc"],
                      "best_epoch": best["epoch"], "model_name": config["model"]["name"],
                      "input_size": 384, "inference_batch_size": config["training"]["batch_size"],
                      "num_workers": config["data"]["num_workers"]}
    print("PREFLIGHT_OK checkpoint_hashes=" + json.dumps(
        {name: details["checkpoint_sha256"] for name, details in info.items()}), flush=True)
    per_model, image_tables, study_tables = {}, {}, {}
    for name, details in info.items():
        print(f"INFER {name}", flush=True)
        model = load_model_state(details["checkpoint"], device="cuda")
        images = predict_archive_images(model, archive, batch_size=int(details["inference_batch_size"]),
                                        num_workers=int(details["num_workers"]), device="cuda")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        image_keys = {item.image_id: (item.subject_id, item.study_id) for item in images}
        if len(images) != 2218 or len(image_keys) != 2218 or image_keys != metadata:
            raise ValueError(f"Incomplete/misaligned image inference: {name}")
        studies = aggregate_mean_probabilities(images)
        lookup = {(item.subject_id, item.study_id): item for item in studies}
        if len(studies) != 1414 or set(lookup) != keys:
            raise ValueError(f"Incomplete Study Mean: {name}")
        per_model[name] = lookup
        image_tables[name] = table(images, image=True)
        study_tables[name] = table(studies, image=False)
        print(f"DONE {name} images={len(images)} studies={len(studies)}", flush=True)
    fused = []
    for key in sorted(keys):
        items = [per_model[name][key] for name in RUNS]
        if len({item.image_count for item in items}) != 1:
            raise ValueError(f"Image count mismatch for {key}")
        probabilities = tuple(sum(item.probabilities[j] for item in items) / 3 for j in range(10))
        fused.append(StudyPrediction(key[0], key[1], items[0].image_count, probabilities))
    fusion = {"scheme": "E01", "formula": "(B01+B02+B03)/3",
              "fusion_weights": {name: 1 / 3 for name in RUNS},
              "image_to_study": "mean of per-image sigmoid probabilities for each model",
              "submission_policy": "all ten original class IDs per Study; no threshold filter; no Gate column",
              "submission_rule_source": "Organizer staff clarification supplied by user on 2026-10-16",
              "validation_macro_auc": FROZEN_AUC, "models": info,
              "test_archive": str(archive), "test_image_count": len(rows), "test_study_count": len(keys)}
    directory, check = write_bundle(args.output_root, args.version, fused, keys, fusion,
                                    image_tables, study_tables, policy="all_classes",
                                    upload_status="generated")
    if check["candidate_rows"] != 14140 or check["candidate_covered_studies"] != 1414:
        raise ValueError("E01 output must contain ten rows for each Study")
    print(json.dumps({"directory": str(directory.resolve()), "image_count": len(rows), **check},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
