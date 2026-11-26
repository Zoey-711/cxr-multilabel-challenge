"""Generate Phase30 evidence from measured artifacts; never infer missing metrics."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
from pathlib import Path

import PIL
import numpy
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEPLOY = HERE / "candidate_deployment"
P29 = ROOT / "outputs/analysis/phase29_deployment"
E08 = ROOT / "outputs/submissions/E08_EMA_20261020"
FRIEND = ROOT.parent / "cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/experiments/H17_DISTILL_02"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    full = {name: read(HERE / f"{name}_full_batch1.json") for name in ("E08_EMA", "H17_02")}
    sample = {name: read(HERE / f"{name}_sample.json") for name in full}
    pre = read(HERE / "E08_EMA_full_preprocessing.json")
    h17_baseline = read(HERE / "H17_02_full_baseline.json")
    test_sample = read(HERE / "local_test_100_submission_consistency.json")
    repeats = read(HERE / "timing_repeatability.json")
    phase20 = read(ROOT / "outputs/analysis/phase20_inference_optimization/prediction_equivalence.json")
    stages = {"E08_EMA": read(HERE / "e08_stage_breakdown.json"),
              "H17_02": read(HERE / "h17_stage_breakdown.json")}
    new_cost = {name: read(DEPLOY / f"{name}_cost_report.json") for name in full}
    old_cost = {"E08_EMA": read(P29 / "cost_protocol_report.json"),
                "H17_02": read(P29 / "h17_cost_protocol_report.json")}
    with (HERE / "probability_differences.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["candidate", "class_id", "max_abs_diff", "mean_abs_diff", "p99_abs_diff",
                  "studies_above_1e-4", "worst_subject", "worst_study", "worst_new_prob",
                  "worst_frozen_prob", "auc_new", "auc_frozen"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, data in full.items():
            for row in data["ensemble"]["per_class"]:
                writer.writerow({"candidate": name, **row})
    with (HERE / "precision_batch_factor_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["candidate", "model", "precision", "same_input",
                  "max_abs_logit_batch16_vs_batch1", "max_abs_probability_batch16_vs_batch1",
                  "fp16_vs_fp32_single_max_abs_logit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, data in sample.items():
            for model, modes in data["precision_batch_factors"].items():
                for precision in ("fp16", "fp32"):
                    item = modes[precision]
                    writer.writerow({"candidate": name, "model": model, "precision": precision,
                                     "same_input": item["same_input"],
                                     "max_abs_logit_batch16_vs_batch1": item["max_abs_logit_batch16_vs_batch1"],
                                     "max_abs_probability_batch16_vs_batch1": item["max_abs_probability_batch16_vs_batch1"],
                                     "fp16_vs_fp32_single_max_abs_logit": modes["fp16_vs_fp32_single_max_abs_logit"]})
    with (HERE / "latency_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["candidate", "mode", "stage", "mean_ms", "p50_ms", "p95_ms", "max_ms", "device"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, data in stages.items():
            for mode, stage_data in data["modes"].items():
                for stage, metrics in stage_data.items():
                    writer.writerow({"candidate": name, "mode": mode, "stage": stage,
                                     "device": data["device"], **metrics})
    with (HERE / "latency_before_after.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["candidate", "old_phase29_mean_ms", "new_phase30_mean_ms", "delta_ms_unpaired",
                  "paired_old_mean_ms", "paired_new_mean_ms", "paired_delta_ms",
                  "new_p50_ms", "new_p95_ms", "new_max_ms", "device", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in full:
            old = old_cost[name]
            new = new_cost[name]
            paired = stages[name]["modes"]
            writer.writerow({"candidate": name, "old_phase29_mean_ms": old["mean_ms"],
                             "new_phase30_mean_ms": new["mean_ms"],
                             "delta_ms_unpaired": new["mean_ms"] - old["mean_ms"],
                             "paired_old_mean_ms": paired["double_rgb"]["end_to_end"]["mean_ms"],
                             "paired_new_mean_ms": paired["single_rgb"]["end_to_end"]["mean_ms"],
                             "paired_delta_ms": paired["single_rgb"]["end_to_end"]["mean_ms"] - paired["double_rgb"]["end_to_end"]["mean_ms"],
                             "new_p50_ms": new["p50_ms"], "new_p95_ms": new["p95_ms"],
                             "new_max_ms": new["max_ms"], "device": new["gpu"],
                             "status": "T4_UNVERIFIED"})
    manifest = read(E08 / "manifest.json")
    size = {}
    for name, meta in manifest["models"].items():
        ckpt = ROOT / "outputs/experiments" / meta["run_id"] / meta["checkpoint"]
        cfg = ckpt.parent / "config.yaml"
        for path, expected in ((ckpt, meta["checkpoint_sha256"]), (cfg, meta["config_sha256"])):
            if sha(path) != expected:
                raise ValueError(f"Frozen source SHA mismatch: {path}")
        size[name] = {"checkpoint": str(ckpt), "checkpoint_bytes": ckpt.stat().st_size,
                      "checkpoint_sha256": meta["checkpoint_sha256"],
                      "config": str(cfg), "config_bytes": cfg.stat().st_size,
                      "config_sha256": meta["config_sha256"]}
    h17weight = P29 / "weights/H17_DISTILL_02_best_model.pth"
    h17config = FRIEND / "resolved_config.yaml"
    h17sha = "35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
    if sha(h17weight) != h17sha:
        raise ValueError("H17 weight SHA mismatch")
    h17 = {"checkpoint": str(h17weight), "checkpoint_bytes": h17weight.stat().st_size,
           "checkpoint_sha256": h17sha, "config": str(h17config),
           "config_bytes": h17config.stat().st_size, "config_sha256": sha(h17config)}
    sources = [DEPLOY / x for x in ("predict.py", "cost.py", "build_cost_study_index.py",
                                     "frozen_e08_manifest.json", "README_T4.md", "requirements_t4.txt")]
    sources += [ROOT / x for x in ("src/cxr/data/dataset.py", "src/cxr/data/archive.py",
                                   "src/cxr/data/sampling.py", "src/cxr/models/factory.py",
                                   "src/cxr/utils/checkpoint.py", "src/cxr/engine/validate.py",
                                   "src/cxr/inference/aggregate.py")]
    sources += [HERE / x for x in ("run_consistency_audit.py", "benchmark_h17_stages.py",
                                   "benchmark_e08_stages.py", "audit_submission_test_sample.py",
                                   "write_phase30_reports.py")]
    source_files = {str(x): {"bytes": x.stat().st_size, "sha256": sha(x)} for x in sources}
    e08_total = sum(x["checkpoint_bytes"] + x["config_bytes"] for x in size.values()) + sum(x["bytes"] for x in source_files.values())
    h17_total = h17["checkpoint_bytes"] + h17["config_bytes"] + sum(x["bytes"] for x in source_files.values())
    write_json(HERE / "model_size_and_hashes.json", {
        "limit_bytes": 500000000, "E08_EMA": {"models": size, "weight_plus_config_bytes":
        sum(x["checkpoint_bytes"] + x["config_bytes"] for x in size.values()),
        "conservative_package_files_bytes": e08_total, "size_pass": e08_total <= 500000000},
        "H17_02": {"model": h17, "weight_plus_config_bytes": h17["checkpoint_bytes"] + h17["config_bytes"],
        "conservative_package_files_bytes": h17_total, "size_pass": h17_total <= 500000000},
        "candidate_code_files": source_files,
        "note": "Package byte sum includes shared code/config and excludes platform PyTorch runtime; all actually loaded weight files are counted."})
    write_json(HERE / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "numpy": numpy.__version__, "Pillow": PIL.__version__,
        "target_hardware": "T4 16GB; unavailable locally", "timing_input": "100 local Val images, not official fixed 100"})
    summary = {
        "official_protocol_source": "https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf",
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU", "official_target": "T4 16GB",
        "precision": "FP16 autocast", "batch": 1, "warmup": 5, "measured_images": 100, "rounds": 3,
        "image_source": "local Val fixed images; official fixed 100 unavailable",
        "scope": "file read through final probability CSV write, model load/warmup excluded, CUDA synchronized",
        "submission_equivalence": "FAIL for existing frozen submissions on a separate 100 single-image Test Study sample; official fixed 100/index still unavailable",
        "local_test_sample_consistency": test_sample["candidates"],
        "candidates": {name: {"mean_ms": new_cost[name]["mean_ms"], "p50_ms": new_cost[name]["p50_ms"],
                               "p95_ms": new_cost[name]["p95_ms"], "max_ms": new_cost[name]["max_ms"],
                               "latency_status": "T4_UNVERIFIED"} for name in full}}
    write_json(HERE / "cost_protocol_report.json", summary)
    lines = ["# Phase30 一致性审计", "", "对齐全部 859 个官方 Val Study、1422 张图像，10 类固定顺序。冻结概率只读。差异阈值保持 1e-4。", "",
             f"- 原数据集预处理、旧的双 RGB 转换与新的单 RGB 转换在 **{pre['preprocessing']['sampled_images']} / {pre['preprocessing']['sampled_images']} 张**图像上生成逐位相同的输入张量。输入最大误差为 0。",
             f"- E08 原 Phase20 推理按 B01–B03 batch16、M02/N01_EMA batch8，FP16，与冻结 859 Study 概率最大差 {phase20['ensembles']['E08_EMA']['max_abs_probability_diff']}。",
             "- 新部署入口必须 batch1：同一个输入张量的模型前向在 batch1 与较大 batch 中有数值差，FP32 与 FP16 也有差；模型始终 eval，类别 1–10，先逐图 sigmoid 再 Study Mean，融合公式不变。",
             f"- H17 在本地4060即使用原 batch16，也与朋友保存概率相差最多 {h17_baseline['ensemble']['max_abs_diff']:.9f}；硬件/软件数值环境差异仍在，不能把全部差值单独归因于 batch。", ""]
    for name in full:
        d = full[name]["ensemble"]
        test = test_sample["candidates"][name]
        lines += [f"## {name}", "",
                  f"- batch1 FP16 对冻结 Val：最大差 **{d['max_abs_diff']:.9f}**，平均差 {d['mean_abs_diff']:.9f}，P99 {d['p99_abs_diff']:.9f}；最差 {d['max_diff_study'][0]}/{d['max_diff_study'][1]} 类别 {d['max_diff_class']}。",
                  f"- Macro-AUC 新推理 {d['macro_auc_new']:.9f}，冻结 {d['macro_auc_frozen']:.9f}。概率一致性 **FAIL（>1e-4）**。逐类最大/平均/P99与最差 Study 见 `probability_differences.csv`。",
                  f"- 单/双 RGB 转换的配对 100 图概率最大差 {stages[name]['max_probability_diff_same_image']:.9f}；这项优化不会改变输入像素。", ""]
        lines += [f"- 另外在预先按 ID 选定的 100 个无标签 Test 单图 Study 上，对原冻结 Submission 最大差 {test['max_abs_diff']:.9f}，结果 {test['status_1e-4']}。该样本不是官方固定测速图片，未使用 Test 标签或计算 Test AUC。", ""]
    lines += ["## 判定", "", "冻结 CSV 若来自较大 batch 或另一设备，不能直接作为 batch1 部署路径的逐概率一致性证明。不得放宽阈值。若正式采用 batch1 路径，需生成独立 Test Submission 并在官方固定图片及 Study 映射上复核；现有冻结 Submission 保持不动。"]
    (HERE / "prediction_consistency_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Phase30 部署优化与 T4 验收状态", "",
             "- 已从旧推理的双 `convert('RGB')` 删去外层一次；1422 张 Val 输入张量完全一致，固定 100 图的成对前向概率完全一致。未改模型、384 分辨率、融合权重或 Study Mean。",
             "- 优化前后 4060 `cost.py` 测试使用相同 100 张 Val 图片、batch1、5 预热、3 轮；两次运行不是严格配对，归因以独立分段配对数据为准。", ""]
    for name in full:
        old, new = old_cost[name], new_cost[name]
        modes = stages[name]["modes"]
        lines += [f"- {name}：原 `cost.py` {old['mean_ms']:.2f}ms，优化版 {new['mean_ms']:.2f}ms（P50 {new['p50_ms']:.2f} / P95 {new['p95_ms']:.2f} / max {new['max_ms']:.2f}ms）；分段交替配对 {modes['double_rgb']['end_to_end']['mean_ms']:.2f}→{modes['single_rgb']['end_to_end']['mean_ms']:.2f}ms。"]
    e_runs = repeats["E08_EMA"]["optimized_run_mean_ms"]
    h_runs = repeats["H17_02"]["optimized_run_mean_ms"]
    lines += ["", f"重复独立运行存在波动：E08 优化版均值 {min(e_runs):.2f}–{max(e_runs):.2f}ms，H17 优化版均值 {min(h_runs):.2f}–{max(h_runs):.2f}ms；交替配对结果更适合归因。记录见 `timing_repeatability.json`。",
             "", "模型加必要配置及源代码的保守字节数见 `model_size_and_hashes.json`，均低于 500,000,000 字节。", "",
             "本地只有 RTX4060；无 T4、无官方固定 100 张计时图片及对应完整 Test Study 映射。因此两者均为 **T4_UNVERIFIED**，对原冻结 Submission 的本地 100 单图 Study 一致性也未通过。对已有冻结 Val 概率的 1e-4 阈值均为 FAIL；不得标记正式部署 PASS。",
             "", "候选代码位于 `candidate_deployment/`；`cost.py` 确实导入同目录 `predict.py` 的加载与单图推理函数。其结果 CSV 包含完整文件读取到写出概率的墙钟时间。见 `candidate_deployment/README_T4.md` 复现命令。"]
    (HERE / "phase30_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_names = ["probability_differences.csv", "precision_batch_factor_summary.csv", "latency_breakdown.csv",
                      "latency_before_after.csv", "prediction_consistency_audit.md",
                      "phase30_report.md", "model_size_and_hashes.json", "environment.json",
                      "cost_protocol_report.json", "local_test_100_submission_consistency.json",
                      "timing_repeatability.json",
                      "E08_EMA_full_batch1.json", "H17_02_full_batch1.json",
                      "E08_EMA_full_preprocessing.json", "E08_EMA_sample.json",
                      "H17_02_sample.json", "H17_02_full_baseline.json",
                      "e08_stage_breakdown.json", "h17_stage_breakdown.json",
                      "e08_stage_outputs.csv", "h17_stage_outputs.csv"]
    artifact_names += [f"candidate_deployment/{name}_cost_report.json" for name in full]
    artifact_names += [f"candidate_deployment/{name}_cost_predictions.csv" for name in full]
    artifacts = {name: {"bytes": (HERE / name).stat().st_size,
                        "sha256": sha(HERE / name)} for name in artifact_names}
    write_json(HERE / "phase30_manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "independent Phase30 outputs; frozen models and submissions unchanged",
        "files": artifacts, "source_files": source_files,
        "T4_status": "UNVERIFIED", "prediction_status_vs_frozen_submission": "FAIL_1e-4"})
    print(json.dumps({"reports": ["probability_differences.csv", "latency_breakdown.csv",
                                  "latency_before_after.csv", "prediction_consistency_audit.md",
                                  "phase30_report.md", "model_size_and_hashes.json"],
                      "E08_new_ms": new_cost["E08_EMA"]["mean_ms"],
                      "H17_new_ms": new_cost["H17_02"]["mean_ms"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
