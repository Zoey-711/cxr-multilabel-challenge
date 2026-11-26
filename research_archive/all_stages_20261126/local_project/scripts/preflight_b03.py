"""Audit B03 against B01's saved runtime configuration before any training."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import study_targets_from_rows
from cxr.utils.config import load_config


ALLOWED_DIFFERENCES = {"experiment.name", "training.backbone_lr", "training.early_stopping_patience"}


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--b01-run-dir", required=True, type=Path)
    parser.add_argument("--planned-run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    b01_dir = args.b01_run_dir.resolve()
    b01, b03 = load_config(b01_dir / "config.yaml"), load_config(args.config)
    first, third = flatten(b01), flatten(b03)
    differences = [{"field": key, "b01": first.get(key), "b03": third.get(key)}
                   for key in sorted(first.keys() | third.keys()) if first.get(key) != third.get(key)]
    if {item["field"] for item in differences} != ALLOWED_DIFFERENCES:
        parser.error(f"Unexpected or missing B01/B03 config differences: {differences}")
    training = b03["training"]
    if (training["backbone_lr"] != 0.00003 or training["classifier_lr"] != 0.0003 or
            training["weight_decay"] != 0.0001 or training["early_stopping_patience"] != 6 or
            training["batch_size"] != 16 or int(training.get("gradient_accumulation_steps", 1)) != 1 or
            training["epochs"] != 30 or b03["experiment"]["seed"] != 42):
        parser.error("B03 requested training settings do not match")
    if b01["training"]["batch_size"] != 16 or int(b01["training"].get("gradient_accumulation_steps", 1)) != 1:
        parser.error("B01 actual batch configuration differs from 16 x 1")
    log = (b01_dir / "train_stdout.log").read_text(encoding="utf-8", errors="replace")
    if "batch=16 train_images=30344 val_studies=859" not in log or "step=1/1897" not in log:
        parser.error("B01 actual runtime log does not confirm batch and step count")
    if b03["data"]["label_encoding"]["unlisted_policy"] != "negative":
        parser.error("Closed-world label policy changed")
    archive = archive_path(b03)
    train_rows, val_rows = read_split_rows(archive, "train"), read_split_rows(archive, "val")
    val_studies = study_targets_from_rows(val_rows, unlisted_policy="negative")
    if len(train_rows) != 30344 or len(val_studies) != 859:
        parser.error("B03 does not use the audited train/validation splits")
    provenance = json.loads((b01_dir / "pretrained_source.json").read_text(encoding="utf-8"))
    weight = project / ".wheels" / "torch_cache" / "hub" / "checkpoints" / "densenet121-a639ec97.pth"
    if not weight.is_file():
        parser.error(f"ImageNet weight file missing: {weight}")
    with weight.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if sha256 != provenance["sha256"]:
        parser.error("ImageNet weight hash differs from B01")
    if b03["model"] != b01["model"] or b03["data"] != b01["data"] or b03["validation"] != b01["validation"]:
        parser.error("Model, data, or Study-level validation protocol changed")
    run_dir = project / "outputs" / "experiments" / args.planned_run_id
    if run_dir.exists() or run_dir.resolve() == b01_dir:
        parser.error(f"B03 run directory is not new and independent: {run_dir}")
    if args.planned_run_id in {"b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated"}:
        parser.error("Run ID collides with an earlier experiment")
    output = args.output_dir.resolve()
    if output == b01_dir or b01_dir in output.parents:
        parser.error("Preflight output must be outside B01")
    result = {"passed": True, "b01_run": str(b01_dir), "b03_config": str(args.config.resolve()),
              "planned_b03_run": str(run_dir), "differences": differences,
              "micro_batch": 16, "gradient_accumulation_steps": 1, "effective_batch": 16,
              "optimizer_steps_per_epoch": math.ceil(len(train_rows) / 16),
              "train_images": len(train_rows), "val_images": len(val_rows), "val_studies": len(val_studies),
              "archive": str(archive), "closed_world_policy": b03["data"]["label_encoding"],
              "imagenet_source": provenance["source"], "imagenet_sha256": sha256,
              "initialization": "Fresh torchvision ImageNet DenseNet-121 and new classifier; train.py loads a prior checkpoint only with --resume",
              "model_selection": b03["validation"],
              "train_augmentation": "No random image augmentation in shared CXRArchiveDataset; same deterministic preprocessing for both runs"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# B03 preflight against actual B01 run", "", "No training started during this audit.",
              "", "| Field | B01 | B03 |", "|---|---:|---:|"]
    report += [f"| `{item['field']}` | {item['b01']} | {item['b03']} |" for item in differences]
    report += ["", "Only the backbone learning rate and permitted early-stopping patience change, besides the experiment name.",
               "Both runs use microbatch 16, accumulation 1, effective batch 16, and 1,897 scheduler steps per epoch.",
               f"Data: `{archive}`; {len(train_rows)} train images, {len(val_rows)} validation images, {len(val_studies)} validation Studies.",
               "The closed-world label encoding, deterministic preprocessing, model, scheduler, seed, and Study-level AUC protocol match B01.",
               f"ImageNet weight SHA-256 `{sha256}` matches B01 provenance.",
               f"Independent planned output: `{run_dir}`; absent at audit.",
               "Start without `--resume`; `train.py` will create a fresh ImageNet-pretrained model and new classifier."]
    (output / "preflight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
