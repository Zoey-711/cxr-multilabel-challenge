"""Check the augmentation-only M02 diff and smoke-test real train images."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.utils.config import load_config


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="New M02 config")
    parser.add_argument("--m01-config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    m01 = load_config(args.m01_config)
    m02 = load_config(args.config)
    run = args.m01_run_dir.resolve()
    output = args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("M02 preparation must not write into frozen M01")
    if m01 != load_config(run / "config.yaml"):
        parser.error("M01 source config differs from its frozen runtime config")
    augmentation = m02["data"].get("train_augmentation")
    if not augmentation or augmentation.get("name") != "conservative_cxr_v1":
        parser.error("Expected conservative_cxr_v1 train augmentation")
    baseline = json.loads(json.dumps(m01))
    proposed = json.loads(json.dumps(m02))
    proposed["experiment"]["name"] = baseline["experiment"]["name"]
    proposed["data"].pop("train_augmentation")
    if proposed != baseline:
        parser.error("M02 has differences beyond experiment name and train augmentation")
    archive = archive_path(m02)
    train_rows = read_split_rows(archive, "train")[:2]
    val_rows = read_split_rows(archive, "val")[:2]
    seed = int(m02["experiment"]["seed"])
    policy = m02["data"]["label_encoding"]["unlisted_policy"]
    torch.manual_seed(seed)
    train_loader = make_dataloader(archive, "train", batch_size=2, num_workers=0,
                                   seed=seed, rows=train_rows, unlisted_policy=policy,
                                   augmentation=augmentation)
    val_loader = make_dataloader(archive, "val", batch_size=2, num_workers=0,
                                 seed=seed, rows=val_rows, unlisted_policy=policy)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    for batch in (train_batch, val_batch):
        if tuple(batch["image"].shape) != (2, 3, 384, 384) or \
                tuple(batch["target"].shape) != (2, 10) or \
                not torch.isfinite(batch["image"]).all():
            raise ValueError("M02 DataLoader smoke test failed")
    no_aug_batch = next(iter(make_dataloader(archive, "val", batch_size=2, num_workers=0,
                                             seed=seed, rows=val_rows, unlisted_policy=policy)))
    if not torch.equal(val_batch["image"], no_aug_batch["image"]):
        raise ValueError("Validation preprocessing changed")
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    durations = [row["elapsed_seconds"] for row in records]
    memory = {"m01_max_peak_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in records),
              "m01_max_peak_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in records)}
    with (run / "gpu_monitor.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        monitor = list(csv.DictReader(handle))
    memory["m01_max_board_used_mib"] = max(int(row["gpu_board_used_mib"]) for row in monitor
                                            if row["gpu_board_used_mib"])
    source = json.loads((run / "pretrained_source.json").read_text(encoding="utf-8"))
    report = {"source_m01_config": str(args.m01_config.resolve()),
              "m01_config_sha256": sha256(args.m01_config), "m02_config_sha256": sha256(args.config),
              "only_config_differences": ["experiment.name", "data.train_augmentation"],
              "training_hyperparameters_unchanged": True,
              "m02_augmentation": augmentation,
              "seed": seed, "train_smoke_shape": list(train_batch["image"].shape),
              "validation_smoke_shape": list(val_batch["image"].shape),
              "train_study_ids": train_batch["study_id"], "validation_study_ids": val_batch["study_id"],
              "validation_deterministic": True,
              "pretrained_source": source,
              "m01_epoch_seconds": durations,
              "m01_epoch_seconds_median": float(torch.tensor(durations).median()),
              "m01_total_epoch_hours": sum(durations) / 3600,
              "m01_gpu_memory": memory,
              "m02_gpu_memory_projection": "Similar model/shape; actual peak must be measured in M02 training",
              "m02_time_projection": "approximately 8–11 min/epoch, 1.2–1.7 h for 9 epochs or 4–5.5 h for 30; CPU augmentation overhead and early stop are uncertain",
              "training_started": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "m01_m02_diff.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# M01 → M02 preparation", "",
             "Compared against the frozen M01 runtime config. The only configuration differences are the experiment name and training augmentation.",
             "All model, optimization, data split, seed, and Study-level validation settings match M01.",
             "", "## New train-only augmentation", "",
             "For each image, independently with probability 0.5, rotate within ±3° and translate within ±2% of width/height; independently with probability 0.5, vary brightness and contrast within ±5%.",
             "No flips, random crops, elastic transforms, cutout, or external data. The M01 preprocessing remains unchanged after augmentation.",
             "Validation and test receive no augmentation. A two-image real-data DataLoader smoke test passed with finite 2×3×384×384 tensors and ten targets.",
             "", "## Resource estimate", "",
             f"M01 max allocated GPU memory: {memory['m01_max_peak_allocated_mib']} MiB; max reserved: {memory['m01_max_peak_reserved_mib']} MiB.",
             f"The passive GPU monitor measured peak board use of {memory['m01_max_board_used_mib']} MiB.",
             f"M01 logged epoch time median: {report['m01_epoch_seconds_median']:.0f} s; total epoch time {report['m01_total_epoch_hours']:.2f} h.",
             "Estimate M02 at 8–11 min/epoch, roughly 1.2–1.7 h if nine epochs or 4–5.5 h for all 30. Actual early stopping and augmentation overhead remain unknown.",
             "This preparation did not load a model, run training, or change any M01 artifact."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"differences": report["only_config_differences"], "smoke_shape": report["train_smoke_shape"],
                      "gpu_memory": memory, "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
