"""Check controlled L01 ASL configuration and data without starting training."""

import argparse
import json
from pathlib import Path

import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.engine.losses import per_label_loss
from cxr.inference.versioned_submission import sha256
from cxr.models.factory import imagenet_weight_info
from cxr.utils.config import load_config


M02 = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated")


def flatten(value: dict, prefix: str = "") -> dict:
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        out.update(flatten(item, name) if isinstance(item, dict) else {name: item})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/phase9_asl_preflight_20261017.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline, candidate = load_config(M02 / "config.yaml"), load_config(args.config)
    old, new = flatten(baseline), flatten(candidate)
    changes = {key: {"M02": old.get(key), "L01": new.get(key)} for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)}
    permitted = {"experiment.name", "training.loss", "training.asl.gamma_neg", "training.asl.gamma_pos",
                 "training.asl.clip", "training.asl.eps", "training.asl.detach_focal_weight"}
    if set(changes) != permitted or old["training.loss"] != "bce_with_logits" or new["training.loss"] != "asymmetric_loss":
        raise ValueError(f"Uncontrolled configuration differences: {changes}")
    if candidate["data"].get("train_sampling"):
        raise ValueError("L01 must preserve M02 image-level shuffle sampling")
    if (Path("outputs/experiments") / candidate["experiment"]["name"]).exists():
        raise FileExistsError("L01 experiment output directory exists")
    source = json.loads((M02 / "pretrained_source.json").read_text(encoding="utf-8"))
    _, _, filename = imagenet_weight_info("convnext_tiny")
    weight = Path(".wheels/torch_cache/hub/checkpoints") / filename
    if not weight.is_file() or sha256(weight) != source["sha256"]:
        raise ValueError("Cached ImageNet weight differs from frozen M02 source")
    data = archive_path(candidate)
    rows = {split: read_split_rows(data, split) for split in ("train", "val")}
    samples = {}
    for split in ("train", "val"):
        ds = CXRArchiveDataset(data, split, rows=rows[split][:1], unlisted_policy="negative",
                               augmentation=candidate["data"]["train_augmentation"] if split == "train" else None,
                               preprocessing=candidate["data"].get("preprocessing", "black_pad_v1"))
        item = ds[0]
        ds.close()
        if item["image"].shape != (3, 384, 384) or item["target"].shape != (10,) or not torch.isfinite(item["image"]).all():
            raise ValueError(f"Bad {split} DataLoader sample")
        samples[split] = {"image_shape": list(item["image"].shape), "target_shape": list(item["target"].shape),
                          "subject_id": item["subject_id"], "study_id": item["study_id"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.tensor([[0.2] * 10, [-0.4] * 10], device=device, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0] * 5, [0.0, 1.0] * 5], device=device)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        loss = per_label_loss(logits, targets, candidate["training"]).mean()
    loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits.grad).all()):
        raise ValueError("Non-finite ASL loss/gradient")
    report = {"status": "passed", "training_started": False, "model_loaded": False,
              "controlled_difference": changes, "baseline_config_sha256": sha256(M02 / "config.yaml"),
              "candidate_config_sha256": sha256(args.config), "pretrained_sha256": sha256(weight),
              "train_images": len(rows["train"]), "val_images": len(rows["val"]),
              "real_data_samples": samples, "loss_smoke_device": device, "loss_value": float(loss.detach()),
              "planned_experiment_directory": str(Path("outputs/experiments") / candidate["experiment"]["name"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": device, "changes": list(changes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
