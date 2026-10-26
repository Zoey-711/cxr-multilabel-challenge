"""Train one predeclared V03 residual head on frozen M02 Train features."""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cxr.engine.b01_analysis import read_study_predictions
from cxr.models.study_residual import MultiViewResidual
from train_study_heads import Studies, collate, metrics, paired_bootstrap, write_predictions


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualStudies(Dataset):
    def __init__(self, studies: Studies, baseline_probabilities: np.ndarray):
        self.studies = studies
        if baseline_probabilities.shape != (len(studies), 10):
            raise ValueError("Baseline Study probabilities have the wrong shape")
        if not np.isfinite(baseline_probabilities).all() or not ((baseline_probabilities >= 0) & (baseline_probabilities <= 1)).all():
            raise ValueError("Invalid baseline probabilities")
        self.baseline_probs = np.array(baseline_probabilities, dtype=np.float32)
        safe = np.clip(self.baseline_probs, 1e-6, 1 - 1e-6)
        self.baseline_logits = np.log(safe / (1 - safe)).astype(np.float32)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        return *self.studies[index], self.baseline_logits[index]


def residual_collate(batch):
    x, mask, y, indices = collate([item[:3] for item in batch])
    baseline = torch.from_numpy(np.stack([item[3] for item in batch]))
    return x, mask, y, indices, baseline


def mean_cached_probability(cache: Path, split: str, studies: Studies) -> np.ndarray:
    image_logits = np.load(cache / f"{split}_logits.npy", mmap_mode="r")
    if image_logits.shape[1] != 10:
        raise ValueError("Expected ten cached image logits")
    image_probs = torch.sigmoid(torch.from_numpy(np.array(image_logits, dtype=np.float16))).float().numpy()
    return np.stack([image_probs[indices].mean(axis=0) for indices in studies.indices]).astype(np.float32)


@torch.inference_mode()
def predict(model, loader, dataset):
    model.eval()
    probabilities = np.empty((len(dataset), 10), dtype=np.float32)
    for x, mask, _, indices, baseline in loader:
        raw = model(x.cuda(), mask.cuda(), baseline.cuda())
        probabilities[indices] = raw.sigmoid().float().cpu().numpy()
    single = np.array([len(group) == 1 for group in dataset.studies.indices])
    probabilities[single] = dataset.baseline_probs[single]
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite V03 probabilities")
    return probabilities


def load_previous(path, keys, targets):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["Subject_id"], row["Study_id"]): row for row in rows}
    if len(rows) != len(keys) or set(lookup) != set(keys):
        raise ValueError(f"Previous validation Study IDs differ: {path}")
    ordered = [lookup[key] for key in keys]
    y = np.array([[int(row[f"label_{j}"]) for j in range(1, 11)] for row in ordered])
    if not np.array_equal(y, targets):
        raise ValueError(f"Previous Study labels differ: {path}")
    return np.array([[float(row[f"probability_{j}"]) for j in range(1, 11)] for row in ordered])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = root / config["source"]["m02_run"]
    cache = root / config["source"]["feature_cache"]
    prior = root / config["source"]["phase12_run"]
    output = args.output_dir or root / "outputs/experiments/v03_m02_multiview_residual"
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if sha(source / "best_model.pt") != config["source"]["checkpoint_sha256"]:
        raise ValueError("Frozen M02 checkpoint SHA mismatch")
    frozen_cache = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if frozen_cache["checkpoint_sha256"] != config["source"]["checkpoint_sha256"]:
        raise ValueError("Cache source mismatch")
    for name, expected in frozen_cache["files_sha256"].items():
        if sha(cache / name) != expected:
            raise ValueError(f"Frozen cache changed: {name}")
    if not (prior / "frozen_manifest.json").is_file():
        raise ValueError("V00/V01/V02 comparison is not frozen")
    if config["model"] != {"feature_dim": 768, "dropout": 0.1, "max_logit_delta": 0.25,
                            "variance": "population", "single_view_delta": "exactly_zero"}:
        raise ValueError("V03 architecture differs from predeclared design")
    seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train = Studies(cache, "train"); val = Studies(cache, "val")
    if set(key[0] for key in train.keys) & set(key[0] for key in val.keys):
        raise ValueError("Train/Val patient overlap")
    m02, m02_y, subjects, study_ids = read_study_predictions(source / "validation_analysis/study_predictions.csv")
    lookup = {key: i for i, key in enumerate(zip(subjects, study_ids))}
    if len(lookup) != len(val):
        raise ValueError("M02 validation coverage mismatch")
    indices = [lookup[key] for key in val.keys]
    m02 = m02[indices]
    if not np.array_equal(m02_y[indices], val.labels):
        raise ValueError("M02 and cached validation labels differ")
    train_set = ResidualStudies(train, mean_cached_probability(cache, "train", train))
    val_set = ResidualStudies(val, m02)
    output.mkdir(parents=True)
    (output / "config.yaml").write_bytes(args.config.read_bytes())
    train_loader = DataLoader(train_set, batch_size=config["training"]["study_batch_size"], shuffle=True,
                              collate_fn=residual_collate, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=config["training"]["study_batch_size"], shuffle=False,
                            collate_fn=residual_collate, num_workers=0)
    model = MultiViewResidual(768, 0.1, 0.25).cuda()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                   weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    best = metrics(m02, val.labels)["macro_auc"]
    best_epoch = 0
    torch.save({"format_version": 1, "kind": "v03", "model_state_dict": model.state_dict()}, output / "best_head.pt")
    write_predictions(output / "study_predictions.csv", val.keys, val.labels, m02)
    logs = [{"epoch": 0, "val_macro_auc": best, "train_loss": None, "selection": "M02 unchanged baseline"}]
    started = time.perf_counter()
    expected_multi = sum(len(group) > 1 for group in train.indices)
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        loss_sum = 0.0; seen_multi = 0
        for x, mask, y, _, baseline in train_loader:
            x = x.cuda(); mask = mask.cuda(); y = y.cuda(); baseline = baseline.cuda()
            multi = mask.sum(dim=1) > 1
            if not multi.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, mask, baseline)
            residual = raw[multi] - baseline[multi]
            loss = (nn.functional.binary_cross_entropy_with_logits(raw[multi], y[multi]) +
                    config["training"]["residual_l2"] * residual.square().mean())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite V03 training loss")
            loss.backward()
            optimizer.step()
            count = int(multi.sum())
            loss_sum += loss.item() * count; seen_multi += count
        if seen_multi != expected_multi:
            raise ValueError("Not all multi-image Train Studies were supervised")
        scheduler.step()
        p = predict(model, val_loader, val_set)
        score = metrics(p, val.labels)["macro_auc"]
        record = {"epoch": epoch, "train_loss": loss_sum / seen_multi,
                  "val_macro_auc": score, "val_macro_f1": metrics(p, val.labels)["macro_f1_at_0_5"],
                  "lr": scheduler.get_last_lr()[0]}
        logs.append(record)
        print(f"V03 epoch {epoch}: train_loss {record['train_loss']:.5f}, AUC {score:.5f}", flush=True)
        if score > best:
            best = score; best_epoch = epoch
            torch.save({"format_version": 1, "kind": "v03", "model_state_dict": model.state_dict()}, output / "best_head.pt")
            write_predictions(output / "study_predictions.csv", val.keys, val.labels, p)
    (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs), encoding="utf-8")
    checkpoint = torch.load(output / "best_head.pt", map_location="cpu", weights_only=True)
    reloaded = MultiViewResidual(768, 0.1, 0.25).cuda().eval()
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    probabilities = predict(reloaded, val_loader, val_set)
    saved = load_previous(output / "study_predictions.csv", val.keys, val.labels)
    if not np.allclose(saved, probabilities, atol=1e-7, rtol=0):
        raise ValueError("Best V03 checkpoint does not reproduce 859 Study predictions")
    single = np.array([len(group) == 1 for group in val.indices])
    if not np.array_equal(saved[single], m02[single]):
        raise ValueError("Single-image Study baseline was changed")
    predecessors = {"M02": m02}
    for kind in ("v00", "v01", "v02"):
        predecessors[kind] = load_previous(prior / kind / "study_predictions.csv", val.keys, val.labels)
    bootstrap = {name: paired_bootstrap(predictions, saved, val.labels, val.keys,
                                       config["evaluation"]["bootstrap_replicates"],
                                       config["evaluation"]["seed"])
                 for name, predictions in predecessors.items()}
    summary = {"best_epoch": best_epoch, "epochs_run": config["training"]["epochs"],
               "train_studies": len(train), "train_multi_image_studies": expected_multi,
               "validation_studies": len(val), "validation_patients": len({key[0] for key in val.keys}),
               "validation_single_studies": int(single.sum()), "validation_multi_studies": int((~single).sum()),
               "scores": {"V03": metrics(saved, val.labels),
                          **{name: metrics(p, val.labels) for name, p in predecessors.items()}},
               "subgroups": {name: {"single": metrics(p[single], val.labels[single]),
                                    "multi": metrics(p[~single], val.labels[~single])}
                             for name, p in {"V03": saved, **predecessors}.items()},
               "paired_patient_bootstrap_v03_minus": bootstrap,
               "head_parameters": sum(p.numel() for p in model.parameters()),
               "head_bytes": (output / "best_head.pt").stat().st_size,
               "head_sha256": sha(output / "best_head.pt"),
               "elapsed_seconds": time.perf_counter() - started,
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "frozen_cache_manifest_sha256": sha(cache / "manifest.json"),
               "m02_checkpoint_sha256": sha(source / "best_model.pt")}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_auc": best,
                      "delta_vs_m02": bootstrap["M02"]["point_delta"]}), flush=True)


if __name__ == "__main__":
    main()
