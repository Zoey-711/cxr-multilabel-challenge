"""Reproducible Phase49 Val inference and paired benchmark; never reads Test labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import predict

HERE = Path(__file__).resolve().parent
SAMPLES = predict.ROOT / "outputs/analysis/phase29_deployment/benchmark_images"
ARCHIVE = Path("F:/数据集/cxr_first.zip")


def samples():
    measured = sorted((SAMPLES / "pictures_100").glob("*.jpg"))
    warmup = sorted((SAMPLES / "warmup_5").glob("*.jpg"))
    if len(measured) != 100 or len(warmup) != 5:
        raise ValueError(f"Fixed Val benchmark set missing: {len(measured)} / {len(warmup)}")
    return measured, warmup


def run(name: str):
    measured, warmup = samples()
    if name == "torch_inplace":
        equivalence = predict.validate_preprocessing(measured + warmup)
        if not equivalence["exact"]:
            raise ValueError(f"Preprocessing changed: {equivalence}")
        (HERE / "preprocessing_equivalence.json").write_text(json.dumps(equivalence, indent=2))
    backend = predict.load_backend(name)
    stage = HERE / f"stages_{name}.csv"
    benchmark_dir = HERE / f"benchmark_{name}"
    val = HERE / f"val_{name}.csv"
    if not stage.exists():
        predict.stage_breakdown(backend, measured, stage)
    if not benchmark_dir.exists():
        print("benchmark", name, flush=True)
        print(json.dumps(predict.benchmark(backend, measured, warmup, benchmark_dir)), flush=True)
    if not val.exists():
        print("validation", name, flush=True)
        result = predict.validation(backend, ARCHIVE, val)
        (HERE / f"val_{name}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


def compare():
    from cxr.data.archive import read_split_rows
    from cxr.data.dataset import study_targets_from_rows
    from cxr.engine.metrics import binary_auc
    rows = read_split_rows(ARCHIVE, "val")
    studies = study_targets_from_rows(rows)
    targets = {(s.subject_id, s.study_id): np.asarray(s.labels, dtype=np.int8) for s in studies}
    names = ["torch_frozen", "torch_inplace", "onnx_fp32", "onnx_fp16",
             "onnx_fp32_iobinding", "onnx_fp16_iobinding"]
    predictions = {}
    for name in names:
        path = HERE / f"val_{name}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        pairs = [(r["Subject_id"], r["Study_id"]) for r in data]
        if len(data) != 859 or len(set(pairs)) != 859 or set(pairs) != set(targets):
            raise ValueError(f"Val Study alignment failed: {name}")
        p = {key: np.asarray([float(r[f"Prob_{j}"]) for j in range(1, 11)])
             for key, r in zip(pairs, data)}
        if not all(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in p.values()):
            raise ValueError(f"Invalid probabilities: {name}")
        predictions[name] = p
    keys = sorted(targets)
    y = np.asarray([targets[k] for k in keys])
    records = []
    auc_by_name = {}
    for name, p in predictions.items():
        mat = np.asarray([p[k] for k in keys])
        auc = np.asarray([binary_auc(y[:, j], mat[:, j]) for j in range(10)])
        auc_by_name[name] = auc
        base = predictions.get("torch_frozen")
        diff = np.abs(mat - np.asarray([base[k] for k in keys])) if base else np.zeros_like(mat)
        records.append(dict(candidate=name, studies=859, macro_auc=float(np.mean(auc)),
                            max_abs_probability_diff=float(np.max(diff)),
                            mean_abs_probability_diff=float(np.mean(diff)),
                            p99_abs_probability_diff=float(np.percentile(diff, 99)),
                            probability_equal_to_baseline=bool(np.max(diff) == 0)))
    with (HERE / "probability_consistency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (HERE / "per_class_auc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["class", "positive_studies"] + list(auc_by_name))
        for j in range(10):
            writer.writerow([j + 1, int(y[:, j].sum())] + [float(auc_by_name[n][j]) for n in auc_by_name])
    # Paired patient resampling; a draw with a single-class column is excluded for that class.
    if "torch_frozen" in predictions:
        subjects = sorted({k[0] for k in keys}); positions = {s: np.where(np.asarray([k[0] == s for k in keys]))[0] for s in subjects}
        mats = {n: np.asarray([p[k] for k in keys]) for n, p in predictions.items()}
        rng = np.random.default_rng(20261029); bootstrap = {}
        for name in predictions:
            if name == "torch_frozen":
                continue
            values = []
            for _ in range(2000):
                ids = np.concatenate([positions[subjects[i]] for i in rng.integers(0, len(subjects), size=len(subjects))])
                valid = [j for j in range(10) if len(np.unique(y[ids, j])) == 2]
                if len(valid) < 10:
                    continue
                delta = np.mean([binary_auc(y[ids, j], mats[name][ids, j]) -
                                 binary_auc(y[ids, j], mats["torch_frozen"][ids, j]) for j in valid])
                values.append(float(delta))
            bootstrap[name] = dict(replicates_requested=2000, replicates_valid=len(values),
                                   invalid_draw_policy="discard draw if any class lacks both labels",
                                   delta_macro_auc=float(np.mean(auc_by_name[name] - auc_by_name["torch_frozen"])),
                                   ci95=[float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None)
        (HERE / "bootstrap_comparison.json").write_text(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=predict.BACKENDS)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
    elif args.backend:
        run(args.backend)
    else:
        parser.error("Choose --backend or --compare")
