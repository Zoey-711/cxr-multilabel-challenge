"""Independently score a completed fold on its held-out Train patients once."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.phase32.dataset import make_dataloader, study_targets_from_rows
from cxr.data.patient_folds import select_fold_rows, sha256
from cxr.engine.analysis import analyze_validation
from cxr.engine.average_precision import average_precision
from cxr.inference.aggregate import ImagePrediction
from cxr.utils.checkpoint import load_model_state
from cxr.phase32.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-kind", choices=("raw", "ema"), default="raw")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    if load_config(run_dir / "config.yaml") != config:
        parser.error("Run config differs from the frozen fold config")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete" or not torch.cuda.is_available():
        parser.error("Completed fold and CUDA GPU are required")
    output = run_dir / ("oof_analysis" if args.checkpoint_kind == "raw" else "ema_oof_analysis")
    checkpoint = run_dir / ("best_model.pt" if args.checkpoint_kind == "raw" else "best_ema_model.pt")
    if output.exists():
        parser.error(f"OOF analysis already exists: {output}")
    archive = archive_path(config)
    rows = select_fold_rows(read_split_rows(archive, "train"), config["data"]["train_subset"], role="oof")
    model = load_model_state(checkpoint, device="cuda")
    loader = make_dataloader(archive, "train", rows=rows, batch_size=int(config["training"]["batch_size"]),
                             num_workers=int(config["data"]["num_workers"]),
                             unlisted_policy=config["data"]["label_encoding"]["unlisted_policy"],
                             preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                             input_size=int(config["data"]["input_size"]))
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                probabilities = torch.sigmoid(model(images)).float().cpu().tolist()
            for index, vector in enumerate(probabilities):
                predictions.append(ImagePrediction(batch["subject_id"][index], batch["study_id"][index],
                                                   batch["image_id"][index], tuple(vector)))
    loader.dataset.close()
    targets = study_targets_from_rows(rows, unlisted_policy="negative")
    analysis, study_rows = analyze_validation(predictions, targets,
                                              threshold=float(config["validation"]["f1_threshold"]))
    probabilities = np.asarray([[row[f"probability_{class_id}"] for class_id in range(1, 11)]
                                for row in study_rows], dtype=float)
    labels = np.asarray([[row[f"target_{class_id}"] for class_id in range(1, 11)]
                         for row in study_rows], dtype=int)
    ap = [average_precision(labels[:, col], probabilities[:, col]) for col in range(10)]
    analysis["macro_ap"] = float(np.mean(ap))
    for item, value in zip(analysis["per_class"], ap):
        item["ap"] = value
    expected = config["data"]["train_subset"]["expected_counts"]["oof"]
    if (analysis["study_count"] != expected["studies"] or len(predictions) != expected["images"] or
            len({row["Subject_id"] for row in study_rows}) != expected["patients"]):
        raise ValueError("OOF Study/patient/image coverage differs from fold index")
    analysis.update({"fold": int(config["data"]["train_subset"]["fold"]),
                     "image_count": len(predictions), "patient_count": expected["patients"],
                     "checkpoint_sha256": sha256(checkpoint),
                     "fold_index_sha256": config["data"]["train_subset"]["sha256"],
                     "role": "held-out Train fold; not used in checkpoint selection"})
    output.mkdir()
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "study_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(study_rows[0]))
        writer.writeheader()
        writer.writerows(study_rows)
    with (output / "image_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["Subject_id", "Study_id", "Image_id"] + [f"probability_{class_id}" for class_id in range(1, 11)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            writer.writerow({"Subject_id": item.subject_id, "Study_id": item.study_id,
                             "Image_id": item.image_id,
                             **{f"probability_{class_id}": value for class_id, value in enumerate(item.probabilities, 1)}})
    print(json.dumps({"fold": analysis["fold"], "studies": analysis["study_count"],
                      "macro_auc": analysis["macro_auc"], "macro_ap": analysis["macro_ap"]}))


if __name__ == "__main__":
    main()
