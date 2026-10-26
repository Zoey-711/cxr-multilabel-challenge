"""Read-only B02 readiness checks; never instantiate a model or start training."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import study_targets_from_rows
from cxr.utils.config import load_config


ALLOWED_DIFFERENCES = {
    "experiment.name", "training.weight_decay", "training.early_stopping_patience",
}


def flatten(value, prefix="") -> dict:
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="B02 config")
    parser.add_argument("--b01-run-dir", required=True, type=Path)
    parser.add_argument("--planned-run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    b01_run = args.b01_run_dir.resolve()
    b01, b02 = load_config(b01_run / "config.yaml"), load_config(args.config)
    first, second = flatten(b01), flatten(b02)
    differences = [{"field": key, "b01": first.get(key), "b02": second.get(key)}
                   for key in sorted(first.keys() | second.keys()) if first.get(key) != second.get(key)]
    if set(row["field"] for row in differences) != ALLOWED_DIFFERENCES:
        parser.error(f"Unexpected or missing B01/B02 differences: {differences}")
    training = b02["training"]
    expected = {"batch_size": 16, "weight_decay": 0.001,
                "early_stopping_patience": 6, "backbone_lr": 0.0001, "classifier_lr": 0.0003,
                "loss": "bce_with_logits", "precision": "fp16_amp"}
    if any(training.get(key) != value for key, value in expected.items()):
        parser.error("B02 training settings do not match the requested comparison")
    if int(training.get("gradient_accumulation_steps", 1)) != 1 or int(b01["training"].get("gradient_accumulation_steps", 1)) != 1:
        parser.error("B01/B02 gradient accumulation is not one")
    if b02["data"]["label_encoding"]["unlisted_policy"] != "negative":
        parser.error("Closed-world label policy is not recorded")
    archive = archive_path(b02)
    train_rows, val_rows = read_split_rows(archive, "train"), read_split_rows(archive, "val")
    val_studies = study_targets_from_rows(val_rows, unlisted_policy="negative")
    if len(val_studies) != 859:
        parser.error("Official validation split does not have 859 studies")
    provenance = json.loads((b01_run / "pretrained_source.json").read_text(encoding="utf-8"))
    weight = project / ".wheels" / "torch_cache" / "hub" / "checkpoints" / "densenet121-a639ec97.pth"
    if not weight.is_file():
        parser.error(f"ImageNet weights missing: {weight}")
    with weight.open("rb") as handle:
        weight_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if weight_sha != provenance["sha256"]:
        parser.error("ImageNet weight hash differs from B01 provenance")
    run_dir = project / "outputs" / "experiments" / args.planned_run_id
    if run_dir.exists() or run_dir.resolve() == b01_run:
        parser.error(f"Planned B02 run directory is not independent and empty: {run_dir}")
    if b02["validation"] != b01["validation"] or b02["model"] != b01["model"]:
        parser.error("Model or validation protocol changed")
    b01_steps = math.ceil(len(train_rows) / int(b01["training"]["batch_size"]))
    b02_steps = math.ceil(math.ceil(len(train_rows) / int(training["batch_size"])) /
                          int(training.get("gradient_accumulation_steps", 1)))
    result = {"ready_for_user_approval": True, "b01_run": str(b01_run),
              "b02_config": str(args.config.resolve()), "planned_b02_run": str(run_dir),
              "config_differences": differences, "archive": str(archive),
              "train_images": len(train_rows), "val_images": len(val_rows), "val_studies": len(val_studies),
              "label_policy": b02["data"]["label_encoding"],
              "pretrained_source": provenance["source"], "pretrained_sha256": weight_sha,
              "pretrained_bytes": weight.stat().st_size,
              "b01_optimizer_steps_per_epoch": b01_steps,
              "b02_optimizer_steps_per_epoch": b02_steps,
              "b01_micro_batch": int(b01["training"]["batch_size"]),
              "b01_accumulation_steps": int(b01["training"].get("gradient_accumulation_steps", 1)),
              "b01_effective_batch": int(b01["training"]["batch_size"]) * int(b01["training"].get("gradient_accumulation_steps", 1)),
              "b02_micro_batch": int(training["batch_size"]),
              "b02_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
              "b02_effective_batch": int(training["batch_size"]) * int(training.get("gradient_accumulation_steps", 1)),
              "validation_protocol": b02["validation"],
              "initialization": "scripts/train.py creates a fresh torchvision ImageNet-pretrained DenseNet-121; no B01 checkpoint is loaded without --resume",
              "time_estimate": "Based on B01, approximately 7-9 minutes per epoch and 1.3-4.5 hours total for 11-30 epochs; estimate only."}
    output = args.output_dir.resolve()
    if output == b01_run or b01_run in output.parents:
        parser.error("Preflight output cannot be inside B01")
    output.mkdir(parents=True, exist_ok=True)
    (output / "b02_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# B01/B02 configuration and launch preflight", "",
              "B02 has not been started. B01 files were only read.", "", "## Differences", "",
              "| Field | B01 | B02 |", "|---|---:|---:|"]
    report += [f"| `{item['field']}` | {item['b01']} | {item['b02']} |" for item in differences]
    report += ["", f"Data archive: `{archive}`; train images: {len(train_rows)}; "
               f"official validation images: {len(val_rows)}, studies: {len(val_studies)}.",
               "Unlisted classes use the audited closed-world negative policy; validation fusion and ten-class Macro-AUC protocol match B01.",
               f"ImageNet weight SHA-256: `{weight_sha}`; matches B01 provenance.",
               f"Planned independent checkpoint directory: `{run_dir}` (absent at preflight).",
               "B02 starts from the same ImageNet weight file with a new ten-class head. Do not pass `--resume` or a B01 checkpoint.",
               f"Microbatch / accumulation / effective batch: B01 16 / 1 / 16; B02 16 / 1 / 16. Scheduler steps per epoch: {b01_steps} in both runs.",
               "Correction: B02 was previously planned as 4 x 4; its microbatch is now 16 and the accumulation field was removed to match B01 exactly.",
               "Early stopping patience changes from 8 to 6 as authorized.",
               "Estimated time: 7-9 minutes per epoch, roughly 1.3-4.5 hours for 11-30 epochs; no B02 timing has been measured."]
    (output / "b02_diff_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
