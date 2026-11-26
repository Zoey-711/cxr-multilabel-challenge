"""Train fixed-budget V00/V01/V02 heads on frozen M02 train features only."""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.models.study_pool import StudyPool
from cxr.utils.labels import CLASS_NAMES


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Studies(Dataset):
    def __init__(self, cache: Path, split: str):
        with (cache / f"{split}_images.csv").open(encoding="utf-8", newline="") as handle:
            images = list(csv.DictReader(handle))
        with (cache / f"{split}_studies.csv").open(encoding="utf-8", newline="") as handle:
            studies = list(csv.DictReader(handle))
        indices = defaultdict(list)
        for row in images:
            indices[(row["Subject_id"], row["Study_id"])].append(int(row["image_index"]))
        self.keys = [(row["Subject_id"], row["Study_id"]) for row in studies]
        self.indices = [indices.pop(key) for key in self.keys]
        if indices or any(not group for group in self.indices) or len(set(self.keys)) != len(self.keys):
            raise ValueError(f"Invalid {split} Study/image mapping")
        self.labels = np.array([[int(row[f"label_{j}"]) for j in range(1, 11)] for row in studies], dtype=np.float32)
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Non-binary Study target")
        self.features = np.load(cache / f"{split}_features.npy", mmap_mode="r")
        if self.features.shape != (len(images), 768):
            raise ValueError(f"Invalid {split} feature shape")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.features[self.indices[index]].astype("float32"), self.labels[index], index


def collate(batch):
    maximum = max(len(item[0]) for item in batch)
    features = torch.zeros(len(batch), maximum, 768)
    valid = torch.zeros(len(batch), maximum, dtype=torch.bool)
    for j, (array, _, _) in enumerate(batch):
        count = len(array)
        features[j, :count] = torch.from_numpy(array)
        valid[j, :count] = True
    targets = torch.from_numpy(np.stack([item[1] for item in batch]))
    indices = np.array([item[2] for item in batch])
    return features, valid, targets, indices


def ap(y, p):
    positive = int(np.sum(y == 1))
    if positive == 0:
        return None
    order = np.argsort(-p, kind="stable")
    positives = y[order] == 1
    return float(np.sum(np.cumsum(positives)[positives] / (np.flatnonzero(positives) + 1)) / positive)


def metrics(p, y):
    classes = []
    for j in range(10):
        pred = p[:, j] >= 0.5
        truth = y[:, j] == 1
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        classes.append({"class_id": j + 1, "class_name": CLASS_NAMES[j], "positive_studies": int(truth.sum()),
                        "negative_studies": int((~truth).sum()), "auc": binary_auc(y[:, j], p[:, j]),
                        "ap": ap(y[:, j], p[:, j]),
                        "f1_at_0_5": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
    valid = [row["auc"] for row in classes if row["auc"] is not None]
    return {"macro_auc": float(np.mean(valid)) if valid else None,
            "macro_ap": float(np.mean([row["ap"] for row in classes if row["ap"] is not None])),
            "macro_f1_at_0_5": float(np.mean([row["f1_at_0_5"] for row in classes])),
            "classes_with_auc": len(valid), "per_class": classes}


@torch.inference_mode()
def predict(model, loader, count):
    model.eval()
    result = np.empty((count, 10), dtype=np.float32)
    for x, mask, _, indices in loader:
        raw = model(x.cuda(non_blocking=True), mask.cuda(non_blocking=True))
        result[indices] = raw.sigmoid().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite validation probabilities")
    return result


def write_predictions(path, keys, labels, probs):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id"] + [f"label_{j}" for j in range(1, 11)] +
                        [f"probability_{j}" for j in range(1, 11)])
        for key, y, p in zip(keys, labels, probs):
            writer.writerow([*key, *[int(v) for v in y], *[float(v) for v in p]])


def paired_bootstrap(baseline, candidate, targets, keys, replicates, seed):
    groups = defaultdict(list)
    for i, (subject, _) in enumerate(keys):
        groups[subject].append(i)
    patients = list(groups.values())
    rng = np.random.default_rng(seed)
    differences = []
    invalid = 0
    for _ in range(replicates):
        draw = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([patients[j] for j in draw])
        y = targets[idx]
        if any(y[:, j].min() == y[:, j].max() for j in range(10)):
            invalid += 1; continue
        first = [binary_auc(y[:, j], baseline[idx, j]) for j in range(10)]
        second = [binary_auc(y[:, j], candidate[idx, j]) for j in range(10)]
        differences.append(float(np.mean(np.array(second) - np.array(first))))
    if len(differences) < 100:
        raise ValueError("Too few valid patient bootstrap draws")
    return {"patients": len(patients), "requested": replicates, "valid": len(differences), "invalid": invalid,
            "point_delta": metrics(candidate, targets)["macro_auc"] - metrics(baseline, targets)["macro_auc"],
            "ci_95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache = project / cfg["source"]["feature_cache"]
    source = project / cfg["source"]["m02_run"]
    output = args.output_dir or project / "outputs/experiments/v00_v01_v02_m02_frozen_features"
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["checkpoint_sha256"] != cfg["source"]["checkpoint_sha256"]:
        raise ValueError("Feature source checkpoint SHA mismatch")
    for filename, expected in manifest["files_sha256"].items():
        if sha(cache / filename) != expected:
            raise ValueError(f"Feature cache SHA mismatch: {filename}")
    train = Studies(cache, "train"); val = Studies(cache, "val")
    if set(subject for subject, _ in train.keys) & set(subject for subject, _ in val.keys):
        raise ValueError("Train/Val patient overlap")
    if len(train) != 18513 or len(val) != 859 or len(set(k[0] for k in val.keys)) != 228:
        raise ValueError("Unexpected Study/patient count")
    m02, m02_y, subjects, studies = read_study_predictions(source / "validation_analysis/study_predictions.csv")
    index = {key: i for i, key in enumerate(zip(subjects, studies))}
    order = [index[key] for key in val.keys]
    m02 = m02[order]; m02_y = m02_y[order]
    if not np.array_equal(m02_y, val.labels):
        raise ValueError("M02 validation labels and frozen features disagree")
    output.mkdir(parents=True)
    (output / "config.yaml").write_bytes(args.config.read_bytes())
    val_loader = DataLoader(val, batch_size=cfg["heads"]["study_batch_size"], shuffle=False,
                            num_workers=0, collate_fn=collate)
    single_mask = np.array([len(v) == 1 for v in val.indices])
    multi_mask = ~single_mask
    summary = {"M02_mean_probability": metrics(m02, val.labels),
               "M02_single_image": metrics(m02[single_mask], val.labels[single_mask]),
               "M02_multi_image": metrics(m02[multi_mask], val.labels[multi_mask])}
    predictions = {"M02": m02}
    for kind in cfg["heads"]["variants"]:
        torch.manual_seed(cfg["heads"]["seed"])
        torch.cuda.manual_seed_all(cfg["heads"]["seed"])
        train_loader = DataLoader(train, batch_size=cfg["heads"]["study_batch_size"], shuffle=True,
                                  num_workers=0, collate_fn=collate,
                                  generator=torch.Generator().manual_seed(cfg["heads"]["seed"]))
        torch.cuda.reset_peak_memory_stats()
        model = StudyPool(kind, cfg["heads"]["feature_dim"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["heads"]["learning_rate"],
                                       weight_decay=cfg["heads"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["heads"]["epochs"])
        run = output / kind
        run.mkdir()
        logs = []
        best = -math.inf; best_epoch = None
        started = time.perf_counter()
        for epoch in range(1, cfg["heads"]["epochs"] + 1):
            model.train()
            total = 0.0; seen = 0
            for x, mask, y, _ in train_loader:
                x = x.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                raw = model(x, mask)
                loss = nn.functional.binary_cross_entropy_with_logits(raw, y)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {kind} loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y); seen += len(y)
            scheduler.step()
            p = predict(model, val_loader, len(val))
            score = metrics(p, val.labels)
            entry = {"epoch": epoch, "train_loss": total / seen, "val_macro_auc": score["macro_auc"],
                     "val_macro_f1": score["macro_f1_at_0_5"], "lr": scheduler.get_last_lr()[0]}
            logs.append(entry)
            print(f"{kind} epoch {epoch}: loss {entry['train_loss']:.5f}, AUC {entry['val_macro_auc']:.5f}", flush=True)
            if score["macro_auc"] > best:
                best = score["macro_auc"]; best_epoch = epoch
                torch.save({"format_version": 1, "kind": kind, "feature_dim": 768,
                            "model_state_dict": model.state_dict()}, run / "best_head.pt")
                write_predictions(run / "study_predictions.csv", val.keys, val.labels, p)
        (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs), encoding="utf-8")
        head_probs = np.array([[float(row[f"probability_{j}"]) for j in range(1, 11)]
                               for row in csv.DictReader((run / "study_predictions.csv").open(encoding="utf-8"))])
        checkpoint = torch.load(run / "best_head.pt", map_location="cpu", weights_only=True)
        reloaded = StudyPool(kind, cfg["heads"]["feature_dim"]).cuda().eval()
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        repeated = predict(reloaded, val_loader, len(val))
        if not np.allclose(repeated, head_probs, atol=1e-7, rtol=0):
            raise ValueError(f"Best {kind} checkpoint does not reproduce saved validation probabilities")
        predictions[kind] = head_probs
        one = {"best_epoch": best_epoch, "training_epochs": cfg["heads"]["epochs"],
               "metrics": metrics(head_probs, val.labels),
               "single_image": metrics(head_probs[single_mask], val.labels[single_mask]),
               "multi_image": metrics(head_probs[multi_mask], val.labels[multi_mask]),
               "parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_bytes": (run / "best_head.pt").stat().st_size,
               "checkpoint_sha256": sha(run / "best_head.pt"),
               "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "elapsed_seconds": time.perf_counter() - started}
        summary[kind] = one
        (run / "summary.json").write_text(json.dumps(one, indent=2, ensure_ascii=False), encoding="utf-8")
    comparisons = {}
    for kind in ("v00", "v01", "v02"):
        comparisons[f"{kind}_minus_M02"] = paired_bootstrap(m02, predictions[kind], val.labels, val.keys,
            cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["seed"])
    for kind in ("v01", "v02"):
        comparisons[f"{kind}_minus_v00"] = paired_bootstrap(predictions["v00"], predictions[kind], val.labels,
            val.keys, cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["seed"])
    result = {"train_studies": len(train), "val_studies": len(val), "val_patients": len(set(k[0] for k in val.keys)),
              "single_studies": sum(len(v) == 1 for v in val.indices),
              "multi_studies": sum(len(v) > 1 for v in val.indices), "scores": summary, "comparisons": comparisons,
              "feature_cache_sha256": sha(cache / "manifest.json")}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({kind: summary[kind]["metrics"]["macro_auc"] for kind in ("v00", "v01", "v02")}), flush=True)


if __name__ == "__main__":
    main()
