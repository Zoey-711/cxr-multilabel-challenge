"""Compare independently validated L01 with M02 and fixed E05 with E03."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_phase9_errors import average_precision
from compare_m01_m02 import measures, patient_bootstrap
from cxr.engine.b01_analysis import read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("L01 runtime config mismatch")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("L01 training incomplete")
    trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_m02 = json.loads((RUNS["M02"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    arrays, hashes = {}, {}
    keys_ref, y_ref, subjects_ref = None, None, None
    for name, run in {**RUNS, "L01": args.run_dir}.items():
        path = run / "validation_analysis/study_predictions.csv"
        if name != "L01":
            frozen = trio[name] if name in trio else frozen_m02
            if sha256(path) != frozen["validation_analysis/study_predictions.csv"]["sha256"] or \
                    sha256(run / "best_model.pt") != frozen["best_model.pt"]["sha256"]:
                raise ValueError(f"Frozen {name} file mismatch")
        p, y, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != 859 or len(set(keys)) != 859 or len(set(studies)) != 859:
            raise ValueError(f"Study coverage mismatch: {name}")
        if keys_ref is None:
            keys_ref, y_ref, subjects_ref = keys, y, subjects
            order = list(range(859))
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(keys_ref):
                raise ValueError(f"Subject/Study alignment mismatch: {name}")
            order = [lookup[key] for key in keys_ref]
            if not np.array_equal(y[order], y_ref):
                raise ValueError(f"Study labels differ: {name}")
        arrays[name] = p[order]
        hashes[name] = {"checkpoint_sha256": sha256(run / "best_model.pt"),
                        "validation_probabilities_sha256": sha256(path)}
    e01 = (arrays["B01"] + arrays["B02"] + arrays["B03"]) / 3
    arrays["E03"] = .5 * e01 + .5 * arrays["M02"]
    arrays["E05"] = .5 * e01 + .5 * arrays["L01"]
    score_names = ("M02", "L01", "E03", "E05")
    scores = {name: measures(arrays[name], y_ref) for name in score_names}
    for name in score_names:
        for j, item in enumerate(scores[name]["per_class"]):
            item["ap"] = average_precision(y_ref[:, j], arrays[name][:, j])
        scores[name]["macro_ap"] = float(np.mean([item["ap"] for item in scores[name]["per_class"]]))
    if abs(scores["E03"]["macro_auc"] - 0.8099309507263751) > 1e-12 or \
            abs(scores["M02"]["macro_auc"] - 0.7972442723187514) > 1e-12:
        raise ValueError("Frozen E03/M02 validation score differs")
    best = json.loads((args.run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    independent = json.loads((args.run_dir / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    if abs(scores["L01"]["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(independent["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
        raise ValueError("L01 independent validation mismatch")
    comparison = {}
    for control, candidate in (("M02", "L01"), ("E03", "E05")):
        paired = patient_bootstrap(arrays[control], arrays[candidate], y_ref, subjects_ref,
                                   replicates=args.bootstrap_replicates, seed=args.seed)
        paired["point_difference"] = scores[candidate]["macro_auc"] - scores[control]["macro_auc"]
        comparison[f"{candidate}_minus_{control}"] = paired
    records = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    m02_records = [json.loads(line) for line in (RUNS["M02"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    trend = {}
    for name, history in (("M02", m02_records), ("L01", records)):
        best_row = max(history, key=lambda row: row["val"]["macro_auc"])
        trend[name] = {"best_epoch": best_row["epoch"], "stop_epoch": history[-1]["epoch"],
                       "best_auc": best_row["val"]["macro_auc"], "stop_auc": history[-1]["val"]["macro_auc"],
                       "auc_drop_best_to_stop": best_row["val"]["macro_auc"] - history[-1]["val"]["macro_auc"],
                       "best_train_loss": best_row["train_loss"], "stop_train_loss": history[-1]["train_loss"],
                       "runtime_seconds": sum(row["elapsed_seconds"] for row in history),
                       "peak_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in history),
                       "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in history)}
    # M02 and L01 losses have different scales; their absolute loss values are not directly comparable.
    result = {"study_count": 859, "patient_count": len(set(subjects_ref)), "scores": scores,
              "comparisons": comparison, "trend": trend, "source_sha256": hashes,
              "fusion": {"E03": "0.5*E01+0.5*M02", "E05": "0.5*E01+0.5*L01"},
              "scope": "fixed 1:1 fusion and patient-cluster paired bootstrap; validation only; no weight/threshold search"}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", "m02_auc", "l01_auc", "e03_auc", "e05_auc",
                         "m02_ap", "l01_ap", "e03_ap", "e05_ap"])
        for j in range(10):
            writer.writerow([j+1, CLASS_NAMES[j], scores["L01"]["per_class"][j]["positive_studies"],
                             *[scores[name]["per_class"][j]["auc"] for name in score_names],
                             *[scores[name]["per_class"][j]["ap"] for name in score_names]])
    print(json.dumps({"auc": {name: scores[name]["macro_auc"] for name in score_names},
                      "f1": {name: scores[name]["macro_f1_at_0_5"] for name in score_names},
                      "bootstrap": {key: [item["ci_95_lower"], item["ci_95_upper"]] for key, item in comparison.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
