"""Audit M01 errors, image-weighted sampling, LR schedule, and AMP logs."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cxr.data.archive import archive_path, parse_positive_classes, read_split_rows
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run, output = args.m01_run_dir.resolve(), args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("Analysis cannot write into frozen M01")
    config = load_config(args.config)
    if config != load_config(run / "config.yaml"):
        parser.error("M01 config does not match frozen runtime config")
    train_rows = read_split_rows(archive_path(config), "train")
    val_rows = read_split_rows(archive_path(config), "val")
    probs, labels, subjects, studies = read_study_predictions(
        run / "validation_analysis" / "study_predictions.csv")
    if len(studies) != 859:
        raise ValueError("Expected 859 validation Studies")
    val_meta = defaultdict(list)
    for row in val_rows:
        val_meta[row["Study_id"]].append(row)
    if set(studies) != set(val_meta):
        raise ValueError("Validation Study metadata and predictions differ")
    focus_rows, cases = [], []
    for class_id in (1, 4, 8):
        j = class_id - 1
        for group_name, indices in (("all", np.arange(len(studies))),
                                    ("single_image", np.asarray([i for i, s in enumerate(studies) if len(val_meta[s]) == 1])),
                                    ("multi_image", np.asarray([i for i, s in enumerate(studies) if len(val_meta[s]) > 1]))):
            truth = labels[indices, j] == 1
            prediction = probs[indices, j] >= 0.5
            positive_probs = probs[indices[truth], j]
            negative_probs = probs[indices[~truth], j]
            focus_rows.append({"class_id": class_id, "class_name": CLASS_NAMES[j], "group": group_name,
                               "study_count": len(indices), "positive_studies": int(truth.sum()),
                               "negative_studies": int((~truth).sum()),
                               "auc": binary_auc(labels[indices, j], probs[indices, j]),
                               "tp": int((truth & prediction).sum()), "fp": int((~truth & prediction).sum()),
                               "fn": int((truth & ~prediction).sum()), "tn": int((~truth & ~prediction).sum()),
                               "positive_probability_p50": float(np.median(positive_probs)) if len(positive_probs) else None,
                               "positive_probability_p90": float(np.quantile(positive_probs, 0.9)) if len(positive_probs) else None,
                               "negative_probability_p50": float(np.median(negative_probs)) if len(negative_probs) else None,
                               "negative_probability_p90": float(np.quantile(negative_probs, 0.9)) if len(negative_probs) else None})
        for i, study in enumerate(studies):
            truth, probability = int(labels[i, j]), float(probs[i, j])
            predicted = int(probability >= 0.5)
            if truth != predicted:
                cases.append({"class_id": class_id, "class_name": CLASS_NAMES[j],
                              "Subject_id": subjects[i], "Study_id": study,
                              "image_count": len(val_meta[study]),
                              "ViewPosition": ";".join(row["ViewPosition"].strip() or "<blank>" for row in val_meta[study]),
                              "target": truth, "probability": probability,
                              "error_type": "false_negative" if truth else "false_positive"})

    train_studies = defaultdict(list)
    for row in train_rows:
        train_studies[row["Study_id"]].append(row)
    if len(train_studies) != 18513:
        raise ValueError("Unexpected train Study count")
    study_image_counts = Counter(len(rows) for rows in train_studies.values())
    class_sampling = []
    for class_id, name in enumerate(CLASS_NAMES, 1):
        study_positive = image_positive = 0
        positive_image_counts, negative_image_counts = [], []
        for rows in train_studies.values():
            positives = parse_positive_classes(rows[0]["Predict_class"])
            if any(parse_positive_classes(row["Predict_class"]) != positives for row in rows):
                raise ValueError("Inconsistent train labels within a Study")
            count = len(rows)
            if class_id in positives:
                study_positive += 1
                image_positive += count
                positive_image_counts.append(count)
            else:
                negative_image_counts.append(count)
        study_rate = study_positive / len(train_studies)
        image_rate = image_positive / len(train_rows)
        class_sampling.append({"class_id": class_id, "class_name": name,
                               "positive_studies": study_positive, "positive_images": image_positive,
                               "study_positive_rate": study_rate, "image_positive_rate": image_rate,
                               "image_to_study_positive_rate_ratio": image_rate / study_rate,
                               "mean_images_positive_study": float(np.mean(positive_image_counts)),
                               "mean_images_negative_study": float(np.mean(negative_image_counts))})

    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    microbatch = int(config["training"]["batch_size"])
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    microbatches_per_epoch = math.ceil(len(train_rows) / microbatch)
    steps_per_epoch = math.ceil(microbatches_per_epoch / accumulation)
    warmup = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    total_steps = int(config["training"]["epochs"]) * steps_per_epoch
    cumulative_updates = 0
    lr_rows = []
    for record in records:
        attempted = record["optimizer_steps"] + record["amp_skipped_steps"]
        if attempted != steps_per_epoch:
            raise ValueError(f"Epoch {record['epoch']} has wrong optimizer attempt count")
        cumulative_updates += record["optimizer_steps"]
        factor = ((cumulative_updates + 1) / max(1, warmup) if cumulative_updates < warmup else
                  0.5 * (1 + math.cos(math.pi * min(1.0, (cumulative_updates - warmup) /
                                                    max(1, total_steps - warmup)))))
        expected = [float(config["training"]["backbone_lr"]) * factor,
                    float(config["training"]["classifier_lr"]) * factor]
        actual = record["learning_rates"]
        match = all(abs(a - b) <= 1e-12 for a, b in zip(actual, expected))
        lr_rows.append({"epoch": record["epoch"], "optimizer_steps": record["optimizer_steps"],
                        "amp_skipped_steps": record["amp_skipped_steps"],
                        "cumulative_optimizer_steps": cumulative_updates,
                        "expected_backbone_lr": expected[0], "actual_backbone_lr": actual[0],
                        "expected_classifier_lr": expected[1], "actual_classifier_lr": actual[1],
                        "schedule_match": match, "train_loss_finite": math.isfinite(record["train_loss"])})
    if not all(row["schedule_match"] and row["train_loss_finite"] for row in lr_rows):
        raise ValueError("Learning-rate schedule or training loss audit failed")
    if sum(row["amp_skipped_steps"] for row in lr_rows) != 9:
        raise ValueError("M01 skipped-step total changed from recorded 9")
    result = {"focus_classes": (1, 4, 8), "train_images": len(train_rows),
              "train_studies": len(train_studies), "train_study_image_count_distribution": dict(sorted(study_image_counts.items())),
              "multi_image_study_share": sum(count for n, count in study_image_counts.items() if n > 1) / len(train_studies),
              "multi_image_share_of_training_images": sum(n * count for n, count in study_image_counts.items() if n > 1) / len(train_rows),
              "sampling_unit": "image; each image of a multi-image Study repeats its Study target",
              "training_augmentation": config["data"].get("train_augmentation", None),
              "normalization": "sum of masked BCE over each accumulation group divided by actual valid label count",
              "final_partial_accumulation_group": {"microbatches": microbatches_per_epoch % accumulation or accumulation,
                                                   "images": len(train_rows) % (microbatch * accumulation) or microbatch * accumulation,
                                                   "expected_attempts_per_epoch": steps_per_epoch},
              "lr_schedule_matches_all_epochs": True,
              "amp_skipped_steps_by_epoch": {str(row["epoch"]): row["amp_skipped_steps"] for row in lr_rows},
              "amp_skipped_total": 9,
              "nonfinite_loss_epochs": [row["epoch"] for row in lr_rows if not row["train_loss_finite"]],
              "amp_limitation": "Logs count skipped optimizer steps by epoch, not exact batch or gradient values. Successful updates in every epoch rule out an epoch-long failure but do not prove every gradient finite.",
              "preprocessing_note": "PIL ImageOps.pad used by M01 fits full rectangular images inside the square and adds black padding. No geometric crop is applied."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "generalization_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "focus_class_errors.csv", focus_rows)
    write_csv(output / "error_cases.csv", cases)
    write_csv(output / "train_sampling_by_class.csv", class_sampling)
    write_csv(output / "learning_rate_audit.csv", lr_rows)
    lines = ["# M01 generalization audit", "",
             "M01 used no stochastic train image augmentation. Train and validation both used deterministic ImageOps.pad and ImageNet normalization.",
             f"The train loader sampled {len(train_rows)} images from {len(train_studies)} Studies. Multi-image Studies were "
             f"{result['multi_image_study_share']:.1%} of Studies but {result['multi_image_share_of_training_images']:.1%} of image-level samples.",
             "This gives each multi-image Study a weight proportional to its image count in the image-level BCE objective.",
             f"All {len(lr_rows)} recorded epoch learning rates match actual successful optimizer-step counts; "
             f"there were {result['amp_skipped_total']} AMP skips by epoch: {result['amp_skipped_steps_by_epoch']}.",
             "All logged training losses are finite. Exact skipped batches and gradient values were not saved, so a stronger claim about every gradient is unavailable.",
             "The last accumulation group has one 8-image microbatch; training divides its BCE sum by its actual 80 valid labels.",
             "", "| Class | Studies + | AUC | TP | FP | FN | Positive p50 | Negative p50 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in focus_rows:
        if row["group"] == "all":
            lines.append(f"| {row['class_id']} {row['class_name']} | {row['positive_studies']} | {row['auc']:.5f} | "
                         f"{row['tp']} | {row['fp']} | {row['fn']} | "
                         f"{row['positive_probability_p50']:.4f} | {row['negative_probability_p50']:.4f} |")
    lines += ["", "Class-specific image-versus-Study sampling prevalence is in `train_sampling_by_class.csv`. "
              "View and image-count strata for focused errors are in `focus_class_errors.csv` and `error_cases.csv`.",
              "The current preprocessing fits the complete image with black padding; it remains unchanged for the M01/M02 augmentation-only comparison."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"focus": [row for row in focus_rows if row["group"] == "all"],
                      "multi_study_share": result["multi_image_study_share"],
                      "multi_image_sample_share": result["multi_image_share_of_training_images"],
                      "amp_skips": result["amp_skipped_steps_by_epoch"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
