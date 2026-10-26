"""Independently analyze a completed run's best checkpoint at Study level."""

import argparse
import csv
import json
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.engine.analysis import analyze_validation
from cxr.inference.aggregate import ImagePrediction
from cxr.utils.checkpoint import check_model_file_size, load_model_state
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/validation_analysis")
    parser.add_argument("--checkpoint-kind", choices=("raw", "ema"), default="raw")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    stored = load_config(run_dir / "config.yaml")
    if config != stored:
        parser.error("Analysis config differs from the exact training config copy")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        parser.error("Training is still running or incomplete; analysis must wait")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required for validation image inference")
    if config["data"]["label_encoding"]["unlisted_policy"] != "negative":
        parser.error("This run requires complete competition targets")
    checkpoint = run_dir / ("best_model.pt" if args.checkpoint_kind == "raw" else "best_ema_model.pt")
    model_size = check_model_file_size(checkpoint)
    model = load_model_state(checkpoint, device="cuda")
    archive = archive_path(config)
    rows = read_split_rows(archive, "val")
    loader = make_dataloader(archive, "val", rows=rows,
                             batch_size=int(config["training"]["batch_size"]),
                             num_workers=int(config["data"]["num_workers"]),
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
    best = json.loads((run_dir / ("best_metrics.json" if args.checkpoint_kind == "raw" else "best_ema_metrics.json")).read_text(encoding="utf-8"))
    saved_metrics = best["val"] if args.checkpoint_kind == "raw" else best["ema_val"]
    if analysis["study_count"] != saved_metrics["study_count"] or abs(analysis["macro_auc"] - saved_metrics["macro_auc"]) > 1e-8:
        raise ValueError("Recomputed best checkpoint AUC differs from saved best_metrics.json")
    analysis.update({"best_epoch": best["epoch"], "best_train_loss": best["train_loss"],
                     "best_model_bytes": model_size, "best_metrics_crosscheck_passed": True})
    output = args.output_dir or run_dir / ("validation_analysis" if args.checkpoint_kind == "raw" else "ema_validation_analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    class_rows = []
    for item in analysis["per_class"]:
        class_rows.append({key: value for key, value in item.items() if key != "probability_quantiles"} |
                          item["probability_quantiles"])
    for filename, records in (("class_metrics.csv", class_rows), ("study_predictions.csv", study_rows)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (output / "image_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["Subject_id", "Study_id", "Image_id"] + [f"probability_{i}" for i in range(1, 11)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            writer.writerow({"Subject_id": item.subject_id, "Study_id": item.study_id,
                             "Image_id": item.image_id,
                             **{f"probability_{i}": value for i, value in enumerate(item.probabilities, 1)}})
    print(json.dumps({key: analysis[key] for key in ("best_epoch", "macro_auc", "macro_f1",
                                                   "zero_predicted_class_studies", "best_model_bytes")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
