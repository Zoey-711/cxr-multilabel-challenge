"""Build Phase49 audit reports from measured artifacts, never fabricate missing runs."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from predict import HERE, PACKAGE, sha


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


names = ["torch_frozen", "torch_inplace", "onnx_fp32", "onnx_fp16",
         "onnx_fp32_iobinding", "onnx_fp16_iobinding"]
prob = {r["candidate"]: r for r in csv.DictReader((HERE / "probability_consistency.csv").open(encoding="utf-8"))}
bootstrap = load(HERE / "bootstrap_comparison.json")
deploy = HERE / "deployment_candidate"
stage_rows, comparison = [], []
for name in names:
    stage = list(csv.DictReader((HERE / f"stages_{name}.csv").open(encoding="utf-8")))
    for field in [x for x in stage[0] if x != "image"]:
        values = [float(row[field]) for row in stage if row[field] not in ("", None)]
        stage_rows.append(dict(candidate=name, stage=field, mean_ms=statistics.mean(values) if values else "N/A",
                               sample_images=len(stage),
                               note="Separately synchronized stages; means are not additive to timed full path"))
    bench = load(HERE / f"benchmark_{name}/cost_report.json")
    val = load(HERE / f"val_{name}.json")
    if name.startswith("torch"):
        model_bytes = (PACKAGE / "model/H17_02.pth").stat().st_size
    else:
        model_bytes = (HERE / "onnx_models" / f"h17_raw_{'fp16' if 'fp16' in name else 'fp32'}.onnx").stat().st_size
    comparison.append(dict(candidate=name, mean_ms=bench["mean_ms"], p50_ms=bench["p50_ms"],
                           p95_ms=bench["p95_ms"], max_ms=bench["max_ms"],
                           val_macro_auc=val["macro_auc"], model_bytes=model_bytes,
                           max_probability_diff_vs_frozen=prob[name]["max_abs_probability_diff"],
                           bootstrap_ci95_vs_frozen=bootstrap.get(name, {}).get("ci95"),
                           gpu=bench["gpu"], official_sample=False,
                           t4_status="T4_UNVERIFIED"))
write_csv(HERE / "latency_breakdown.csv", stage_rows)
write_csv(HERE / "onnx_comparison.csv", comparison)
write_csv(HERE / "tensorrt_comparison.csv", [dict(candidate=x, status="BLOCKED_NATIVE_RUNTIME_MISSING",
                                                       mean_ms="N/A", val_macro_auc="N/A", reason="tensorrt Python module, trtexec and nvinfer DLL unavailable")
                                               for x in ("TensorRT FP16", "TensorRT FP32")])
package_files = [p for p in deploy.iterdir() if p.is_file()]
size = dict(frozen_weight_bytes=(PACKAGE / "model/H17_02.pth").stat().st_size,
            frozen_weight_and_config_bytes=(PACKAGE / "model/H17_02.pth").stat().st_size +
                                           (PACKAGE / "config/deployment_config.json").stat().st_size,
            candidate_files=[dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in package_files],
            candidate_total_bytes=sum(p.stat().st_size for p in package_files),
            official_limit_bytes=500_000_000,
            size_status="PASS" if sum(p.stat().st_size for p in package_files) <= 500_000_000 else "FAIL")
(HERE / "deployment_size.json").write_text(json.dumps(size, indent=2, ensure_ascii=False), encoding="utf-8")

base = next(r for r in comparison if r["candidate"] == "torch_frozen")
best = min(comparison, key=lambda r: r["mean_ms"])
repeat = HERE / "benchmark_torch_frozen_repeat/cost_report.json"
repeat_text = f"; repeated baseline {load(repeat)['mean_ms']:.2f} ms" if repeat.exists() else ""
frozen_status = load(HERE.parent / "h17_02_batch1_77_7_audit/candidate_status.json")
(HERE / "baseline_audit.md").write_text(
    "# Phase49 frozen H17-02 baseline\n\n"
    f"Source: `{PACKAGE}`; checkpoint SHA-256 `{sha(PACKAGE / 'model/H17_02.pth')}`; "
    f"original submission SHA-256 `{frozen_status['submission_sha256']}`. "
    "The original files were read only. Official 77.70 is user confirmed, not a local hidden-Test AUC.\n\n"
    f"Same-session fixed Val benchmark: mean {base['mean_ms']:.2f} ms, P50 {base['p50_ms']:.2f} ms, "
    f"P95 {base['p95_ms']:.2f} ms{repeat_text}. Batch=1, FP16 autocast, five warmup images, "
    "100 fixed Val images, three rounds; read to CSV write with CUDA synchronization. "
    "These Val images are not the organizer's fixed Test timing sample.\n\n"
    "Full 859 Study Val Macro-AUC 0.8086102168, Study Mean of image probabilities. "
    "Local RTX4060 only; T4_UNVERIFIED.\n", encoding="utf-8")
(HERE / "t4_validation.md").write_text(
    "# T4 validation pending\n\nNo T4 device was accessible locally. The 4060 results cannot prove "
    "the official 100 ms limit. Candidate code and reproducer: `deployment_candidate/` and its "
    "`README_T4.md`. On the actual T4, first verify driver/CUDA/ORT compatibility and model SHA, "
    "then use organizer supplied fixed 5+100 images with the candidate's `cost.py`. "
    "Check the output against a separately generated submission from the same entry. "
    "A 4060 ONNX model is portable, but no T4 execution or latency has been proven. "
    "No TensorRT engine was built; a 4060 engine would not prove T4 portability.\n", encoding="utf-8")
entry = load(HERE / "deployment_entry_consistency.json")
decision = (
    "# Phase49 decision\n\n"
    f"Fastest same-sample option: **{best['candidate']}**, {best['mean_ms']:.2f} ms mean / "
    f"{best['p95_ms']:.2f} ms P95 on RTX4060, versus {base['mean_ms']:.2f} ms baseline{repeat_text}. "
    "The 4060 sample is official Val images; the official T4 timing sample is unavailable.\n\n"
    f"Baseline Val AUC {base['val_macro_auc']:.9f}; fastest candidate Val AUC {best['val_macro_auc']:.9f}. "
    f"Max probability difference {best['max_probability_diff_vs_frozen']}; "
    f"paired patient bootstrap 95% CI {bootstrap[best['candidate']]['ci95']} for Macro-AUC difference. "
    "The ONNX probability output is a distinct version and must never reuse the frozen FP16 submission.\n\n"
    f"Independent deployment entry matched evaluated ONNX predictions exactly on {entry['studies']} Val Studies "
    f"({entry['images']} images). Candidate package {size['candidate_total_bytes']} bytes, "
    f"size {size['size_status']}. TensorRT FP16/FP32 blocked by missing native runtime. "
    "T4_UNVERIFIED. No new Test submission was generated; doing so requires a full official Test "
    "prediction run from this new entry and complete CSV validation. Current candidate is suitable "
    "for T4 measurement, not yet for a claim of official deployment compliance.\n")
(HERE / "final_decision.md").write_text(decision, encoding="utf-8")
files = [p for p in HERE.rglob("*") if p.is_file() and p.name != "manifest_sha256.json" and "__pycache__" not in p.parts]
manifest = {str(p.relative_to(HERE)).replace("\\", "/"): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)}
(HERE / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(dict(best=best, deployment_size_bytes=size["candidate_total_bytes"], files_hashed=len(manifest)), ensure_ascii=False))
