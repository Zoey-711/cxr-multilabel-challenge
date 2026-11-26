"""Cache frozen M02 image features/logits for official train and validation only."""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.lib.format import open_memmap

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = project / config["source"]["m02_run"]
    output = args.output_dir or project / config["source"]["feature_cache"]
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads((source / "frozen_manifest.json").read_text(encoding="utf-8"))
    checkpoint = source / "best_model.pt"
    if sha(checkpoint) != config["source"]["checkpoint_sha256"] or sha(checkpoint) != frozen["best_model.pt"]["sha256"]:
        raise ValueError("M02 checkpoint SHA mismatch")
    m02 = load_config(source / "config.yaml")
    if m02["model"]["name"] != "convnext_tiny" or m02["data"]["input_size"] != 384:
        raise ValueError("Unexpected M02 model/preprocessing")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    archive = archive_path(m02)
    model = load_model_state(checkpoint, device="cuda")
    if len(model.classifier) != 3 or model.classifier[-1].in_features != 768:
        raise ValueError("Unexpected ConvNeXt classifier")
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    start = time.perf_counter()
    report = {"checkpoint_sha256": sha(checkpoint), "m02_config_sha256": sha(source / "config.yaml"),
              "archive_path": str(archive), "input_size": 384,
              "preprocessing": "black_pad_v1; RGB, aspect-preserving resize and black pad, ImageNet normalize",
              "training_feature_difference": "Training images were randomly augmented by M02; this cache uses no random augmentation.",
              "splits": {}}
    for split in ("train", "val"):
        rows = read_split_rows(archive, split)
        loader = make_dataloader(archive, split, rows=rows, batch_size=16, num_workers=m02["data"]["num_workers"],
                                 shuffle=False, augmentation=None, preprocessing="black_pad_v1", input_size=384)
        features = open_memmap(output / f"{split}_features.npy", mode="w+", dtype="float16", shape=(len(rows), 768))
        logits = open_memmap(output / f"{split}_logits.npy", mode="w+", dtype="float32", shape=(len(rows), 10))
        metadata = []
        index = 0
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    pooled = model.avgpool(model.features(images))
                    hidden = model.classifier[1](model.classifier[0](pooled))
                    raw = model.classifier[2](hidden)
                count = len(images)
                features[index:index + count] = hidden.float().cpu().numpy().astype("float16")
                logits[index:index + count] = raw.float().cpu().numpy()
                for j in range(count):
                    metadata.append({"image_index": index + j, "Subject_id": batch["subject_id"][j],
                                     "Study_id": batch["study_id"][j], "Image_id": batch["image_id"][j]})
                index += count
                if index % 4096 < count:
                    print(f"{split}: {index}/{len(rows)} images", flush=True)
        if index != len(rows):
            raise ValueError(f"Incomplete {split} cache")
        features.flush(); logits.flush()
        del features, logits
        keys = [(row["Subject_id"], row["Study_id"], row["Image_id"]) for row in metadata]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate {split} image keys")
        with (output / f"{split}_images.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
            writer.writeheader(); writer.writerows(metadata)
        targets = study_targets_from_rows(rows, unlisted_policy=m02["data"]["label_encoding"]["unlisted_policy"])
        with (output / f"{split}_studies.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Subject_id", "Study_id"] + [f"label_{i}" for i in range(1, 11)])
            for item in targets:
                if any(label is None for label in item.labels):
                    raise ValueError("Unexpected unknown class target")
                writer.writerow([item.subject_id, item.study_id, *[int(label) for label in item.labels]])
        report["splits"][split] = {"images": len(rows), "studies": len(targets),
                                   "patients": len({item.subject_id for item in targets})}
        print(f"{split}: complete", flush=True)
    train_subjects = {row["Subject_id"] for row in csv.DictReader((output / "train_studies.csv").open(encoding="utf-8"))}
    val_subjects = {row["Subject_id"] for row in csv.DictReader((output / "val_studies.csv").open(encoding="utf-8"))}
    if train_subjects & val_subjects:
        raise ValueError("Train/Val patient leakage")
    with (source / "validation_analysis" / "image_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        frozen_probs = {(r["Subject_id"], r["Study_id"], r["Image_id"]): np.array(
            [float(r[f"probability_{i}"]) for i in range(1, 11)]) for r in csv.DictReader(handle)}
    val_logits = np.load(output / "val_logits.npy", mmap_mode="r")
    with (output / "val_images.csv").open(encoding="utf-8", newline="") as handle:
        val_meta = list(csv.DictReader(handle))
    if len(frozen_probs) != len(val_meta):
        raise ValueError("M02 frozen validation image count differs")
    maximum = 0.0
    for row in val_meta:
        key = (row["Subject_id"], row["Study_id"], row["Image_id"])
        new_prob = torch.sigmoid(torch.tensor(val_logits[int(row["image_index"])], device="cuda", dtype=torch.float16)).float().cpu().numpy()
        maximum = max(maximum, float(np.max(np.abs(new_prob - frozen_probs[key]))))
    report["max_probability_difference_from_frozen_m02_val_images"] = maximum
    if maximum > 0.0005:
        raise ValueError(f"Val preprocessing/logits disagree with frozen M02: {maximum}")
    report["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["elapsed_seconds"] = time.perf_counter() - start
    report["files_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "max_probability_difference": maximum,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
