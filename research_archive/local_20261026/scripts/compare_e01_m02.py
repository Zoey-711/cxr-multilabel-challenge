"""Compare frozen E01 and independently validated M02 on the same 859 Studies."""

import argparse
import json
from pathlib import Path

import numpy as np

from cxr.engine.b01_analysis import read_study_predictions
from cxr.utils.config import load_config
from compare_m01_m02 import measures, patient_bootstrap


RUNS = {
    "B01": Path("outputs/experiments/b01_densenet121_384_20261016"),
    "B02": Path("outputs/experiments/b02_densenet121_wd1e3_20261016_elevated"),
    "B03": Path("outputs/experiments/b03_densenet121_lr3e5_20261016"),
    "M01": Path("outputs/experiments/m01_convnext_tiny_384_20261016"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="M02 runtime config")
    parser.add_argument("--m02-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/m02_vs_e01_20261016"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if load_config(args.config) != load_config(args.m02_run_dir / "config.yaml"):
        raise ValueError("M02 config differs from its runtime copy")
    if json.loads((args.m02_run_dir / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("M02 training is incomplete")
    arrays = {}
    reference_keys, reference_labels = None, None
    subjects = []
    for name, run in {**RUNS, "M02": args.m02_run_dir}.items():
        probabilities, labels, subject_ids, study_ids = read_study_predictions(
            run / "validation_analysis" / "study_predictions.csv")
        keys = list(zip(subject_ids, study_ids))
        if len(keys) != 859 or len(set(keys)) != 859:
            raise ValueError(f"Expected 859 unique Studies for {name}")
        if reference_keys is None:
            reference_keys, reference_labels, subjects = keys, labels, subject_ids
            order = list(range(859))
        else:
            index = {key: i for i, key in enumerate(keys)}
            if set(index) != set(reference_keys):
                raise ValueError(f"Study IDs differ for {name}")
            order = [index[key] for key in reference_keys]
        probabilities, labels = probabilities[order], labels[order]
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"Study labels differ for {name}")
        arrays[name] = probabilities
    arrays["E01"] = (arrays["B01"] + arrays["B02"] + arrays["B03"]) / 3
    scores = {name: measures(arrays[name], reference_labels) for name in ("E01", "M01", "M02")}
    if abs(scores["E01"]["macro_auc"] - 0.8026744715463039) > 1e-12:
        raise ValueError("Reconstructed E01 validation AUC differs from frozen result")
    paired = patient_bootstrap(arrays["E01"], arrays["M02"], reference_labels, subjects,
                               replicates=args.bootstrap_replicates, seed=42)
    result = {"studies": 859, "patients": len(set(subjects)), "scores": scores,
              "m02_minus_e01_macro_auc": scores["M02"]["macro_auc"] - scores["E01"]["macro_auc"],
              "m02_minus_e01_patient_bootstrap": paired,
              "note": "Validation-only comparison; leaderboard test labels were not used."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "comparison.json"),
                      "auc": {name: scores[name]["macro_auc"] for name in scores},
                      "m02_minus_e01_auc": result["m02_minus_e01_macro_auc"],
                      "ci_95": [paired["ci_95_lower"], paired["ci_95_upper"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
