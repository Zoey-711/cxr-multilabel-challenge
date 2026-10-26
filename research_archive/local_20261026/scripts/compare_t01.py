"""Compare T01 with frozen M02 and fixed E07 with frozen E03 on validation."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_phase9_errors import average_precision
from compare_m01_m02 import measures, patient_bootstrap
from cxr.data.archive import archive_path, read_split_rows
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
}


def stratum(p: np.ndarray, y: np.ndarray, subjects: list[str], index: np.ndarray) -> dict:
    details = []
    for j in range(10):
        yy, pp = y[index, j], p[index, j]
        details.append({"class_id": j+1, "positive_studies": int(yy.sum()),
                        "negative_studies": int(len(yy)-yy.sum()),
                        "auc": binary_auc(yy, pp), "ap": average_precision(yy, pp)})
    aucs = [item["auc"] for item in details]
    return {"studies": len(index), "patients": len({subjects[i] for i in index}),
            "macro_auc": float(np.mean(aucs)) if all(value is not None for value in aucs) else None,
            "per_class": details}


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
    config = load_config(args.config)
    if config != load_config(args.run_dir / "config.yaml") or config["data"]["input_size"] != 384 or config["model"]["name"] != "swin_t":
        raise ValueError("T01 runtime config mismatch")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("T01 training incomplete")
    trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_m02 = json.loads((RUNS["M02"] / "frozen_manifest.json").read_text(encoding="utf-8"))
    arrays, hashes = {}, {}
    keys_ref, y_ref, subjects_ref = None, None, None
    for name, run in {**RUNS, "T01": args.run_dir}.items():
        path = run / "validation_analysis/study_predictions.csv"
        if name != "T01":
            frozen = trio[name] if name in trio else frozen_m02
            if sha256(path) != frozen["validation_analysis/study_predictions.csv"]["sha256"] or \
                    sha256(run / "best_model.pt") != frozen["best_model.pt"]["sha256"]:
                raise ValueError(f"Frozen source mismatch: {name}")
        p, y, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != 859 or len(set(keys)) != 859 or len(set(studies)) != 859:
            raise ValueError(f"Study coverage differs: {name}")
        if keys_ref is None:
            keys_ref, y_ref, subjects_ref = keys, y, subjects
            order = list(range(859))
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(keys_ref):
                raise ValueError(f"Subject/Study keys differ: {name}")
            order = [lookup[key] for key in keys_ref]
            if not np.array_equal(y[order], y_ref):
                raise ValueError(f"Study labels differ: {name}")
        arrays[name] = p[order]
        hashes[name] = {"checkpoint_sha256": sha256(run / "best_model.pt"),
                        "validation_probabilities_sha256": sha256(path)}
    e01 = (arrays["B01"] + arrays["B02"] + arrays["B03"]) / 3
    arrays["E03"] = .5 * e01 + .5 * arrays["M02"]
    arrays["E07"] = .5 * arrays["E03"] + .5 * arrays["T01"]
    names = ("M02", "T01", "E03", "E07")
    scores = {name: measures(arrays[name], y_ref) for name in names}
    for name in names:
        for j, item in enumerate(scores[name]["per_class"]):
            item["ap"] = average_precision(y_ref[:, j], arrays[name][:, j])
        scores[name]["macro_ap"] = float(np.mean([item["ap"] for item in scores[name]["per_class"]]))
    if abs(scores["M02"]["macro_auc"] - 0.7972442723187514) > 1e-12 or \
            abs(scores["E03"]["macro_auc"] - 0.8099309507263751) > 1e-12:
        raise ValueError("Frozen M02/E03 score differs")
    best = json.loads((args.run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    independent = json.loads((args.run_dir / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    if abs(scores["T01"]["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(independent["macro_auc"] - best["val"]["macro_auc"]) > 1e-8:
        raise ValueError("T01 independent validation differs")
    comparisons = {}
    for control, candidate in (("M02", "T01"), ("E03", "E07")):
        paired = patient_bootstrap(arrays[control], arrays[candidate], y_ref, subjects_ref,
                                   replicates=args.bootstrap_replicates, seed=args.seed)
        paired["point_difference"] = scores[candidate]["macro_auc"] - scores[control]["macro_auc"]
        comparisons[f"{candidate}_minus_{control}"] = paired
    val_rows = read_split_rows(archive_path(config), "val")
    image_counts = Counter((row["Subject_id"], row["Study_id"]) for row in val_rows)
    group_indices = {"single_image": np.asarray([i for i, key in enumerate(keys_ref) if image_counts[key] == 1]),
                     "multi_image": np.asarray([i for i, key in enumerate(keys_ref) if image_counts[key] > 1])}
    if len(group_indices["single_image"]) != 391 or len(group_indices["multi_image"]) != 468:
        raise ValueError("Validation image-count strata differ")
    strata = {name: {group: stratum(arrays[name], y_ref, subjects_ref, index)
                     for group, index in group_indices.items()} for name in ("M02", "T01", "E03", "E07")}
    trends = {}
    for name, run in (("M02", RUNS["M02"]), ("T01", args.run_dir)):
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        best_row = max(records, key=lambda item: item["val"]["macro_auc"])
        trends[name] = {"best_epoch": best_row["epoch"], "stop_epoch": records[-1]["epoch"],
                        "best_auc": best_row["val"]["macro_auc"], "stop_auc": records[-1]["val"]["macro_auc"],
                        "auc_drop_best_to_stop": best_row["val"]["macro_auc"] - records[-1]["val"]["macro_auc"],
                        "runtime_seconds": sum(item["elapsed_seconds"] for item in records),
                        "peak_gpu_allocated_mib": max(item["gpu_peak_allocated_mib"] for item in records),
                        "peak_gpu_reserved_mib": max(item["gpu_peak_reserved_mib"] for item in records),
                        "amp_skipped_steps": sum(item["amp_skipped_steps"] for item in records)}
    result = {"study_count": 859, "patient_count": len(set(subjects_ref)), "scores": scores,
              "comparisons": comparisons, "strata": strata, "trends": trends, "source_sha256": hashes,
              "fusion": {"E03": "0.5*E01+0.5*M02", "E07": "0.5*E03+0.5*T01"},
              "scope": "validation only; fixed fusion; paired patient bootstrap; no threshold/weight search"}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", "m02_auc", "t01_auc", "e03_auc", "e07_auc",
                         "m02_ap", "t01_ap", "e03_ap", "e07_ap"])
        for j in range(10):
            writer.writerow([j+1, CLASS_NAMES[j], scores["T01"]["per_class"][j]["positive_studies"],
                             *[scores[name]["per_class"][j]["auc"] for name in names],
                             *[scores[name]["per_class"][j]["ap"] for name in names]])
    print(json.dumps({"auc": {name: scores[name]["macro_auc"] for name in names},
                      "f1": {name: scores[name]["macro_f1_at_0_5"] for name in names},
                      "bootstrap": {key: [value["ci_95_lower"], value["ci_95_upper"]]
                                    for key, value in comparisons.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
