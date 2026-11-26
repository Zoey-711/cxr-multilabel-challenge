"""Compare fixed N01 raw/EMA validation predictions with E03 and E07."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_phase9_errors import average_precision
from compare_m01_m02 import measures
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M02": Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated"),
    "T01": Path("outputs/experiments/t01_swin_tiny_384_20261018"),
}


def paired_bootstrap(arrays: dict[str, np.ndarray], targets: np.ndarray, subjects: list[str],
                     comparisons: list[tuple[str, str]], *, replicates: int, seed: int) -> dict:
    groups = defaultdict(list)
    for i, subject in enumerate(subjects):
        groups[subject].append(i)
    patients = [np.asarray(indices, dtype=np.int32) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    differences = {f"{candidate}_minus_{control}": [] for control, candidate in comparisons}
    invalid = 0
    while len(next(iter(differences.values()))) < replicates:
        if invalid > 20 * replicates:
            raise RuntimeError("Too many invalid patient bootstrap draws")
        draw = rng.integers(0, len(patients), size=len(patients))
        index = np.concatenate([patients[i] for i in draw])
        y = targets[index]
        if any(y[:, j].min() == y[:, j].max() for j in range(10)):
            invalid += 1
            continue
        sampled_auc = {name: float(np.mean([binary_auc(y[:, j], p[index, j]) for j in range(10)]))
                       for name, p in arrays.items()}
        for control, candidate in comparisons:
            differences[f"{candidate}_minus_{control}"].append(sampled_auc[candidate] - sampled_auc[control])
    return {"method": "paired patient-cluster percentile bootstrap; all Studies of drawn patients retained",
            "patient_count": len(patients), "seed": seed, "valid_replicates": replicates,
            "invalid_resamples": invalid,
            "differences": {name: {"point": float(np.mean([binary_auc(targets[:, j], arrays[name.split('_minus_')[0]][:, j]) -
                                                        binary_auc(targets[:, j], arrays[name.split('_minus_')[1]][:, j])
                                                        for j in range(10)])),
                                   "ci95": [float(np.quantile(values, q)) for q in (0.025, 0.975)]}
                            for name, values in differences.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.config)
    if config != load_config(args.run_dir / "config.yaml") or config["model"]["name"] != "efficientnet_v2_s":
        raise ValueError("N01 runtime config mismatch")
    if json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("N01 training not complete")
    trio = json.loads(Path("outputs/analysis/frozen_ensemble/frozen_manifest.json").read_text(encoding="utf-8"))
    frozen = {name: json.loads((RUNS[name] / "frozen_manifest.json").read_text(encoding="utf-8"))
              for name in ("M02", "T01")}
    paths = {name: run / "validation_analysis/study_predictions.csv" for name, run in RUNS.items()}
    paths["N01"] = args.run_dir / "validation_analysis/study_predictions.csv"
    paths["N01_EMA"] = args.run_dir / "ema_validation_analysis/study_predictions.csv"
    arrays, hashes = {}, {}
    reference_keys = reference_y = reference_subjects = None
    for name, path in paths.items():
        if name in RUNS:
            reference_manifest = trio[name] if name in trio else frozen[name]
            if (sha256(path) != reference_manifest["validation_analysis/study_predictions.csv"]["sha256"] or
                    sha256(RUNS[name] / "best_model.pt") != reference_manifest["best_model.pt"]["sha256"]):
                raise ValueError(f"Frozen source SHA mismatch: {name}")
        p, y, subjects, studies = read_study_predictions(path)
        keys = list(zip(subjects, studies))
        if len(keys) != 859 or len(set(keys)) != 859:
            raise ValueError(f"Invalid 859-Study coverage: {name}")
        if reference_keys is None:
            reference_keys, reference_y, reference_subjects = keys, y, subjects
            order = list(range(859))
        else:
            lookup = {key: i for i, key in enumerate(keys)}
            if set(lookup) != set(reference_keys):
                raise ValueError(f"Study/Subject ID mismatch: {name}")
            order = [lookup[key] for key in reference_keys]
            if not np.array_equal(y[order], reference_y):
                raise ValueError(f"Target mismatch: {name}")
        arrays[name] = p[order]
        hashes[name] = {"probabilities_sha256": sha256(path)}
    hashes["N01"]["checkpoint_sha256"] = sha256(args.run_dir / "best_model.pt")
    hashes["N01_EMA"]["checkpoint_sha256"] = sha256(args.run_dir / "best_ema_model.pt")
    e01 = (arrays["B01"] + arrays["B02"] + arrays["B03"]) / 3
    arrays["E03"] = 0.5 * e01 + 0.5 * arrays["M02"]
    arrays["E07"] = 0.5 * arrays["E03"] + 0.5 * arrays["T01"]
    arrays["E08"] = 0.5 * arrays["E03"] + 0.5 * arrays["N01"]
    arrays["E08_EMA"] = 0.5 * arrays["E03"] + 0.5 * arrays["N01_EMA"]
    names = ("N01", "N01_EMA", "E03", "E07", "E08", "E08_EMA")
    scores = {name: measures(arrays[name], reference_y) for name in names}
    for name in names:
        for j, row in enumerate(scores[name]["per_class"]):
            row["ap"] = average_precision(reference_y[:, j], arrays[name][:, j])
        scores[name]["macro_ap"] = float(np.mean([row["ap"] for row in scores[name]["per_class"]]))
    checks = {"E03": 0.8099309507263751, "E07": 0.8115193037842005}
    for name, expected in checks.items():
        if abs(scores[name]["macro_auc"] - expected) > 1e-12:
            raise ValueError(f"Frozen {name} validation AUC differs")
    for name, folder, metrics_file, metric_key in (
            ("N01", "validation_analysis", "best_metrics.json", "val"),
            ("N01_EMA", "ema_validation_analysis", "best_ema_metrics.json", "ema_val")):
        analysis = json.loads((args.run_dir / folder / "analysis.json").read_text(encoding="utf-8"))
        best = json.loads((args.run_dir / metrics_file).read_text(encoding="utf-8"))
        if abs(scores[name]["macro_auc"] - analysis["macro_auc"]) > 1e-8 or \
                abs(scores[name]["macro_auc"] - best[metric_key]["macro_auc"]) > 1e-8:
            raise ValueError(f"Independent N01 {name} probability cross-check failed")
    comparisons = [("N01", "N01_EMA"), ("E03", "E08"), ("E07", "E08"),
                   ("E03", "E08_EMA"), ("E07", "E08_EMA")]
    bootstrap = paired_bootstrap({name: arrays[name] for name in names}, reference_y, reference_subjects,
                                 comparisons, replicates=args.bootstrap_replicates, seed=args.seed)
    residual_corr = {}
    for j in range(10):
        e03_error = reference_y[:, j] - arrays["E03"][:, j]
        n01_error = reference_y[:, j] - arrays["N01"][:, j]
        residual_corr[str(j + 1)] = float(np.corrcoef(e03_error, n01_error)[0, 1])
    result = {"study_count": 859, "patient_count": len(set(reference_subjects)),
              "scores": scores, "bootstrap": bootstrap, "source_sha256": hashes,
              "fusion": {"E08": "0.5*E03+0.5*N01", "E08_EMA": "0.5*E03+0.5*N01_EMA"},
              "error_residual_correlation_e03_n01": residual_corr,
              "interpretation": "EMA is one preconfigured additional estimator; comparisons are correlated and are not independent significance tests"}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_class.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "positive_studies", *[f"{name}_auc" for name in names],
                         *[f"{name}_ap" for name in names]])
        for j in range(10):
            writer.writerow([j + 1, CLASS_NAMES[j], scores["N01"]["per_class"][j]["positive_studies"],
                             *[scores[name]["per_class"][j]["auc"] for name in names],
                             *[scores[name]["per_class"][j]["ap"] for name in names]])
    print(json.dumps({name: scores[name]["macro_auc"] for name in names}, indent=2))


if __name__ == "__main__":
    main()
