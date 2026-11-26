"""Write T01 controlled-backbone and fixed E07 validation report."""

import argparse
import json
from pathlib import Path

from cxr.engine.b01_analysis import read_study_predictions
from cxr.utils.config import load_config


FROZEN = ("b01_densenet121_384_20261016", "b02_densenet121_wd1e3_20261016_elevated",
          "b03_densenet121_lr3e5_20261016", "m02_convnext_tiny_384_aug_20261016_elevated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("T01 runtime config mismatch")
    path = args.run_dir / "comparison/report.md"
    if path.exists():
        raise FileExistsError(path)
    data = json.loads((args.run_dir / "comparison/comparison.json").read_text(encoding="utf-8"))
    benchmark = json.loads((args.run_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    independent = json.loads((args.run_dir / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    if data["study_count"] != 859 or data["patient_count"] != 228 or not independent["best_metrics_crosscheck_passed"]:
        raise ValueError("Independent validation incomplete")
    scores, comparisons, trends = data["scores"], data["comparisons"], data["trends"]
    point = comparisons["E07_minus_E03"]["point_difference"]
    lower = comparisons["E07_minus_E03"]["ci_95_lower"]
    recommendation = ("E07 shows a positive paired validation gain over E03; consider it for the next manual submission after checking deployment and submission format."
                      if point > 0 and lower > 0 else
                      "E03 remains the better-supported next manual submission candidate; E07 lacks a reliable paired validation gain.")
    source = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/validation_analysis/study_predictions.csv")
    _, y, subjects, _ = read_study_predictions(source)
    positive_patients = {j + 1: len({subjects[i] for i in range(len(subjects)) if y[i, j] == 1}) for j in range(10)}
    weight_paths = {name: Path("outputs/experiments") / name / "best_model.pt" for name in FROZEN}
    weight_paths["T01"] = args.run_dir / "best_model.pt"
    weights = {name: file.stat().st_size for name, file in weight_paths.items()}
    total = sum(weights.values())
    if total > 500_000_000:
        raise ValueError(f"E07 unique model weights exceed 500 MB: {total}")
    lines = ["# T01 Swin-Tiny controlled experiment and E07 fixed fusion", "",
             "T01 differs from the frozen M02 runtime configuration only in experiment identity and ImageNet backbone. Both use 384 aspect-preserving black padding, the same conservative CXR augmentation, BCE, image shuffle, AdamW, 8x2 effective batch 16, seed 42, and Study Mean-probability validation.",
             "All 859 validation Studies and 228 patients align by Subject_id/Study_id. No hidden test labels or fusion-weight search were used.",
             "E01=(B01+B02+B03)/3; E03=0.5 E01+0.5 M02; E07=0.5 E03+0.5 T01.", "",
             "| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |", "|---|---:|---:|---:|"]
    for name in ("M02", "T01", "E03", "E07"):
        row = scores[name]
        lines.append(f"| {name} | {row['macro_auc']:.5f} | {row['macro_f1_at_0_5']:.5f} | {row['macro_ap']:.5f} |")
    lines += ["", "## Paired 228-patient bootstrap", ""]
    for name in ("T01_minus_M02", "E07_minus_E03"):
        row = comparisons[name]
        lines.append(f"- {name}: delta AUC {row['point_difference']:+.5f}; 95% CI [{row['ci_95_lower']:+.5f}, {row['ci_95_upper']:+.5f}]; {row['valid_replicates']}/{row['requested_replicates']} valid draws.")
    lines += ["", "## Ten-class AUC / AP", "", "| Class | Positive Studies | M02 AUC | T01 AUC | E03 AUC | E07 AUC | M02 AP | T01 AP | E03 AP | E07 AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for j in range(10):
        rows = [scores[name]["per_class"][j] for name in ("M02", "T01", "E03", "E07")]
        lines.append(f"| {j+1} {rows[0]['class_name']} | {rows[0]['positive_studies']} | " +
                     " | ".join(f"{row['auc']:.5f}" for row in rows) + " | " +
                     " | ".join(f"{row['ap']:.5f}" for row in rows) + " |")
    lines += ["", "Focused class support: " + ", ".join(
        f"class {j}: {scores['T01']['per_class'][j-1]['positive_studies']} positive Studies / {positive_patients[j]} positive patients"
        for j in (1, 4, 8)) + ".", "", "## Single/multi-image Studies", "",
              "Subgroup class composition differs; subgroup AUCs are descriptive.",
              "| Scheme | Group | Studies | Patients | Macro-AUC |", "|---|---|---:|---:|---:|"]
    for name in ("M02", "T01", "E03", "E07"):
        for group in ("single_image", "multi_image"):
            row = data["strata"][name][group]
            value = "undefined" if row["macro_auc"] is None else f"{row['macro_auc']:.5f}"
            lines.append(f"| {name} | {group} | {row['studies']} | {row['patients']} | {value} |")
    lines += ["", "## Runtime, memory and deployment", ""]
    for name in ("M02", "T01"):
        row = trends[name]
        latency = benchmark["results"][name]
        lines.append(f"- {name}: best/stop epoch {row['best_epoch']}/{row['stop_epoch']}; best/stop AUC {row['best_auc']:.5f}/{row['stop_auc']:.5f}; training {row['runtime_seconds']/3600:.2f} h; peak allocated/reserved GPU {row['peak_gpu_allocated_mib']}/{row['peak_gpu_reserved_mib']} MiB; AMP skips {row['amp_skipped_steps']}; local median/p95 end-to-end single-image latency {latency['latency_ms_median']:.1f}/{latency['latency_ms_p95']:.1f} ms.")
    lines += ["", f"T01 checkpoint {weights['T01']/1e6:.2f} MB. All five unique E07 inference checkpoints total {total/1e6:.2f} MB, below 500 MB.",
              "Latency is a local RTX 4060 Laptop measurement that includes validation ZIP read, decode, preprocessing, transfer, FP16 model and CPU probabilities. The official T4 100 ms figure remains a reference requiring separate measurement; it is not treated here as an automatic exclusion rule.",
              "", "## Next manual submission candidate", "", recommendation,
              "No hidden-test inference, submission CSV generation or leaderboard upload was performed."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "e07_weights_bytes": total, "recommendation": recommendation}))


if __name__ == "__main__":
    main()
