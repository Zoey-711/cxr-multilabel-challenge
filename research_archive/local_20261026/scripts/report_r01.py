"""Write R01 resolution-control and fixed E06 validation report."""

import argparse
import json
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("R01 runtime config mismatch")
    path = args.run_dir / "comparison/report.md"
    if path.exists():
        raise FileExistsError(path)
    result = json.loads((args.run_dir / "comparison/comparison.json").read_text(encoding="utf-8"))
    benchmark = json.loads((args.run_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    independent = json.loads((args.run_dir / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    if result["study_count"] != 859 or result["patient_count"] != 228 or not independent["best_metrics_crosscheck_passed"]:
        raise ValueError("Independent validation incomplete")
    scores, comparisons, trends = result["scores"], result["comparisons"], result["trends"]
    point = comparisons["E06_minus_E03"]["point_difference"]
    lower = comparisons["E06_minus_E03"]["ci_95_lower"]
    if point > 0 and lower > 0:
        recommendation = "E06 shows a positive paired validation gain over E03; consider it for the next manual submission after separate submission-format review."
    else:
        recommendation = "E03 remains the better-supported next manual submission candidate; E06 has no reliable paired validation gain."
    lines = ["# R01 512px controlled experiment and fixed E06 fusion", "",
             "All 859 validation Studies and 228 patients align by Subject_id/Study_id. No hidden test labels or fusion-weight search were used.",
             "M02 and R01 differ in input resolution (384→512) and experiment identity only; both use batch 8 and accumulation 2.",
             "E01=(B01+B02+B03)/3; E03=0.5 E01+0.5 M02; E06=0.5 E01+0.5 R01.", "",
             "| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |", "|---|---:|---:|---:|"]
    for name in ("M02", "R01", "E03", "E06"):
        item = scores[name]
        lines.append(f"| {name} | {item['macro_auc']:.5f} | {item['macro_f1_at_0_5']:.5f} | {item['macro_ap']:.5f} |")
    lines.extend(["", "## Paired patient bootstrap", ""])
    for name in ("R01_minus_M02", "E06_minus_E03"):
        item = comparisons[name]
        lines.append(f"- {name}: delta AUC {item['point_difference']:+.5f}; 95% CI "
                     f"[{item['ci_95_lower']:+.5f}, {item['ci_95_upper']:+.5f}]; "
                     f"{item['valid_replicates']}/{item['requested_replicates']} valid patient resamples.")
    lines.extend(["", "## Weak classes", "",
                  "| Class | Positive Studies / Patients | M02 AUC/AP | R01 AUC/AP | E03 AUC | E06 AUC |",
                  "|---:|---:|---:|---:|---:|---:|"])
    patients = {1: 16, 4: 37, 8: 72}
    for class_id in (1, 4, 8):
        j = class_id-1
        m, r, e3, e6 = [scores[name]["per_class"][j] for name in ("M02", "R01", "E03", "E06")]
        lines.append(f"| {class_id} | {r['positive_studies']} / {patients[class_id]} | "
                     f"{m['auc']:.4f}/{m['ap']:.4f} | {r['auc']:.4f}/{r['ap']:.4f} | "
                     f"{e3['auc']:.4f} | {e6['auc']:.4f} |")
    lines.extend(["", "## Single-/multi-image Study strata", "",
                  "These groups have different disease prevalence and patient composition; their AUC difference is descriptive.",
                  "| Model | Group | Studies | Patients | Macro-AUC |",
                  "|---|---|---:|---:|---:|"])
    for name in ("M02", "R01"):
        for group in ("single_image", "multi_image"):
            item = result["strata"][name][group]
            auc = f"{item['macro_auc']:.5f}" if item["macro_auc"] is not None else "undefined"
            lines.append(f"| {name} | {group} | {item['studies']} | {item['patients']} | {auc} |")
    lines.extend(["", "## Runtime and local inference", ""])
    for name in ("M02", "R01"):
        item = trends[name]
        latency = benchmark["results"][name]
        lines.append(f"- {name}: best/stop epoch {item['best_epoch']}/{item['stop_epoch']}; "
                     f"best/stop AUC {item['best_auc']:.5f}/{item['stop_auc']:.5f}; "
                     f"training {item['runtime_seconds']/3600:.2f} h; "
                     f"peak training allocated/reserved {item['peak_gpu_allocated_mib']}/{item['peak_gpu_reserved_mib']} MiB; "
                     f"AMP skips {item['amp_skipped_steps']}; "
                     f"local median/p95 end-to-end latency {latency['latency_ms_median']:.1f}/{latency['latency_ms_p95']:.1f} ms.")
    lines.extend(["", "Latency includes validation ZIP read, decode, preprocessing, GPU transfer, FP16 model, sigmoid, and CPU probabilities on the local RTX 4060 Laptop. It does not certify the official T4 100 ms limit.",
                  "All ten class AUC/AP values, subgroup class supports, full bootstrap details and SHA-256 provenance are in `per_class.csv` and `comparison.json`.",
                  "", "## Next submission candidate", "", recommendation,
                  "No test inference, submission CSV generation, or leaderboard upload was performed."])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "recommendation": recommendation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
