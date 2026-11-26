"""Fail closed on cloud environment, data, fold and GPU smoke checks; never train."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_pack(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cxr_split_zip_v1" or set(manifest["files"]) != {"train.zip", "valid.zip"}:
        raise ValueError("Expected Train/Val-only pack manifest")
    result = {}
    for name, meta in manifest["files"].items():
        file = root / name
        if file.stat().st_size != meta["bytes"] or sha256(file) != meta["sha256"]:
            raise ValueError(f"Cloud ZIP size/SHA-256 mismatch: {file}")
        result[name] = {"bytes": file.stat().st_size, "sha256": meta["sha256"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack-manifest", type=Path, help="Train/Val split ZIP manifest")
    source.add_argument("--data-sha256", help="SHA-256 of the original combined official ZIP")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if "torch" in sys.modules:
        parser.error("GPU visibility must be selected before importing torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torchvision
    from cxr.data.archive import archive_path, image_member, read_split_rows, split_archive_path
    from cxr.data.dataset import CXRArchiveDataset
    from cxr.data.patient_folds import select_fold_rows
    from cxr.models.factory import create_model, imagenet_weight_info
    from cxr.utils.cloud_paths import output_root, pretrained_home
    from cxr.utils.config import load_config

    config = load_config(args.config)
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.10.0" or \
            torchvision.__version__.split("+")[0] != "0.25.0" or torch.version.cuda != "12.8":
        raise RuntimeError("Require Python 3.12, torch 2.10.0, torchvision 0.25.0, CUDA 12.8")
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU unavailable; check driver/image/CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    if gpu.total_memory < 8 * 2**30:
        raise RuntimeError(f"GPU memory below 8 GiB: {gpu.total_memory}")
    data = archive_path(config)
    if args.pack_manifest:
        if not data.is_dir() or data.resolve() != args.pack_manifest.parent.resolve():
            raise ValueError("CXR_DATA_ROOT must be the Train/Val ZIP directory containing the manifest")
        pack = verify_pack(data, args.pack_manifest)
    else:
        if not data.is_file() or len(args.data_sha256) != 64 or sha256(data) != args.data_sha256.lower():
            raise ValueError("Original official ZIP SHA-256 mismatch")
        pack = {"combined_zip": {"bytes": data.stat().st_size, "sha256": args.data_sha256.lower()}}
    rows = {split: read_split_rows(data, split) for split in ("train", "val")}
    expected = {"train": (30344, 18513, 11055), "val": (1422, 859, 228)}
    for split, (images, studies, patients) in expected.items():
        split_rows = rows[split]
        counts = (len(split_rows), len({x["Study_id"] for x in split_rows}),
                  len({x["Subject_id"] for x in split_rows}))
        if counts != (images, studies, patients):
            raise ValueError(f"{split} row/Study/patient counts differ: {counts}")
        with zipfile.ZipFile(split_archive_path(data, split)) as archive:
            members = set(archive.namelist())
        required = {image_member(row, split) for row in split_rows}
        required.add(f"cxr_first/{'valid' if split == 'val' else split}.csv")
        if not required.issubset(members) or (data.is_dir() and members != required):
            raise ValueError(f"{split} ZIP missing/extra members: {len(required-members)}/{len(members-required)}")
    subset = config["data"].get("train_subset")
    if not subset or subset.get("kind") != "patient_fold_v1":
        raise ValueError("Cloud entry requires the frozen Phase17 patient folds")
    train = select_fold_rows(rows["train"], subset, role="train")
    oof = select_fold_rows(rows["train"], subset, role="oof")
    if {r["Subject_id"] for r in train} & {r["Subject_id"] for r in oof}:
        raise ValueError("Train/OOF patient leakage")
    if {r["Subject_id"] for r in rows["val"]} & {r["Subject_id"] for r in rows["train"]}:
        raise ValueError("Official Val patient leakage")
    project = Path(__file__).resolve().parents[1]
    weights_home = pretrained_home(project)
    _, url, filename = imagenet_weight_info(config["model"]["name"])
    weight = weights_home / "hub" / "checkpoints" / filename
    if sha256(weight) != config["model"]["pretrained_sha256"]:
        raise ValueError(f"ImageNet weight SHA mismatch: {weight}")
    output = output_root()
    output.mkdir(parents=True, exist_ok=True)
    free_data = shutil.disk_usage(data.resolve()).free
    free_output = shutil.disk_usage(output.resolve()).free
    if free_data < 500_000_000 or free_output < 6_000_000_000:
        raise OSError(f"Insufficient free storage: data={free_data}, output={free_output}; require 0.5/6 GB reserve")
    dataset = CXRArchiveDataset(data, "train", rows=train[:2],
                                preprocessing=config["data"].get("preprocessing", "black_pad_v1"),
                                input_size=int(config["data"]["input_size"]))
    sample = dataset[0]
    dataset.close()
    if sample["image"].shape != (3, 384, 384) or not torch.isfinite(sample["image"]).all():
        raise ValueError("Real-image preprocessing failed")
    os.environ["TORCH_HOME"] = str(weights_home)
    model = create_model(config["model"]["name"], pretrained=True).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = sample["image"].unsqueeze(0).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
    if logits.shape != (1, 10) or not torch.isfinite(loss):
        raise ValueError("GPU forward/loss failed")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("GPU backward produced non-finite gradients")
    report = {"state": "passed", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256(args.config), "fold": subset["fold"],
              "python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "gpu": gpu.name,
              "gpu_memory_bytes": gpu.total_memory, "gpu_index_requested": args.gpu,
              "data_root": str(data), "pack": pack, "pretrained_url": url,
              "pretrained_sha256": sha256(weight), "fold_index_sha256": subset["sha256"],
              "train_images": len(train), "oof_images": len(oof),
              "free_data_bytes": free_data, "free_output_bytes": free_output,
              "smoke_loss": loss.item(), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
