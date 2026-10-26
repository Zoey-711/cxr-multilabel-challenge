"""Write a validation-only report for completed L01 and its fixed E05 fusion."""

import argparse
import json
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/comparison/report.md")
    args = parser.parse_args()
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("L01 config mismatch")
    path = args.output or args.run_dir / "comparison/report.md"
    if path.exists():
        raise FileExistsError(path)
    result = json.loads((args.run_dir / "comparison/comparison.json").read_text(encoding="utf-8"))
    independent = json.loads((args.run_dir / "validation_analysis/analysis.json").read_text(encoding="utf-8"))
    if result["study_count"] != 859 or result["patient_count"] != 228 or not independent["best_metrics_crosscheck_passed"]:
        raise ValueError("Independent validation is incomplete")
    scores, comparisons, trend = result["scores"], result["comparisons"], result["trend"]
    point = comparisons["E05_minus_E03"]["point_difference"]
    lower = comparisons["E05_minus_E03"]["ci_95_lower"]
    if point > 0 and lower > 0:
        recommendation = "E05 has a positive validation gain with a paired 95% interval above zero; it merits consideration for the next manual submission."
    else:
        recommendation = "E03 remains the better-supported next manual submission candidate; E05 has no clear paired-bootstrap improvement over E03."
    lines = ["# L01 ASL controlled experiment and fixed E05 fusion", "",
             "All 859 validation Studies and 228 patients align on Subject_id/Study_id; no hidden test labels or weight search were used.",
             "E01=(B01+B02+B03)/3; E03=0.5 E01 + 0.5 M02; E05=0.5 E01 + 0.5 L01.", "",
             "| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |", "|---|---:|---:|---:|"]
    for name in ("M02", "L01", "E03", "E05"):
        item = scores[name]
        lines.append(f"| {name} | {item['macro_auc']:.5f} | {item['macro_f1_at_0_5']:.5f} | {item['macro_ap']:.5f} |")
    lines.extend(["", "## Paired patient bootstrap", ""])
    for name in ("L01_minus_M02", "E05_minus_E03"):
        item = comparisons[name]
        lines.append(f"- {name}: delta Macro-AUC {item['point_difference']:+.5f}; 95% CI "
                     f"[{item['ci_95_lower']:+.5f}, {item['ci_95_upper']:+.5f}]; "
                     f"{item['valid_replicates']}/{item['requested_replicates']} valid patient resamples.")
    lines.extend(["", "## Weak classes", "",
                  "| Class ID | Positive Studies | M02 AUC/AP | L01 AUC/AP | E03 AUC | E05 AUC |",
                  "|---:|---:|---:|---:|---:|---:|"])
    positive_patients = {1: 16, 4: 37, 8: 72}
    for class_id in (1, 4, 8):
        j = class_id - 1
        m, l, e3, e5 = [scores[name]["per_class"][j] for name in ("M02", "L01", "E03", "E05")]
        lines.append(f"| {class_id} | {l['positive_studies']} ({positive_patients[class_id]} patients) | "
                     f"{m['auc']:.4f}/{m['ap']:.4f} | {l['auc']:.4f}/{l['ap']:.4f} | "
                     f"{e3['auc']:.4f} | {e5['auc']:.4f} |")
    lines.extend(["", "## Training trend", ""])
    for name in ("M02", "L01"):
        item = trend[name]
        lines.append(f"- {name}: best/stop epoch {item['best_epoch']}/{item['stop_epoch']}; "
                     f"AUC at best/stop {item['best_auc']:.5f}/{item['stop_auc']:.5f}; "
                     f"drop {item['auc_drop_best_to_stop']:.5f}; elapsed {item['runtime_seconds']/3600:.2f} h; "
                     f"peak GPU allocated {item['peak_gpu_allocated_mib']} MiB; AMP skips {item['amp_skipped_steps']}.")
    lines.extend(["", "BCE and ASL training losses have different scales; compare validation AUC trends rather than their absolute loss values.",
                  "All 10 per-class AUC/AP values and bootstrap details are in `per_class.csv` and `comparison.json`.",
                  "", "## Next submission candidate", "", recommendation, "No submission file was generated or uploaded."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "recommendation": recommendation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
