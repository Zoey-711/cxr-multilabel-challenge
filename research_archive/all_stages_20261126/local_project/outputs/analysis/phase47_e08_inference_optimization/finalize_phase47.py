"""CPU-only consolidation of measured Phase47 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P44 = ROOT / "outputs/analysis/phase44_e08_acceleration"
P45 = ROOT / "outputs/submissions/E08_ONNX_FP32_20261028"
P29 = ROOT / "outputs/analysis/phase29_deployment"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    benchmarks = [load(HERE / f"{name}_benchmark.json") for name in
                  ("pytorch_fp16", "onnx_fp32", "onnx_fp16")]
    a = load(HERE / "preprocess_a_validation.json")
    paired_a = load(HERE / "paired_preprocess.json")
    paired_binding = load(HERE / "paired_binding.json")
    fp32 = load(P44 / "onnx_fp32_val_audit.json")
    fp16 = load(P44 / "onnx_fp16_val_audit.json")
    p45 = load(P45 / "final_entry_consistency.json")
    size = load(P44 / "model_size_report.json")
    if p45["status"] != "PASS" or a["input_bitwise_equal"] is not True:
        raise ValueError("Prediction equivalence gate failed")
    baseline = []
    for b in benchmarks:
        baseline.append({"candidate": b["candidate"], "gpu": b["device"],
                         "mean_ms": b["mean_ms"], "p50_ms": b["p50_ms"],
                         "p95_ms": b["p95_ms"], "max_ms": b["max_ms"],
                         "warmup": 5, "images": 100, "rounds": 3, "batch": 1,
                         "sample": "fixed Val; not organizer fixed Test sample",
                         "resource_status": "4060_SHARED_GPU_LOAD_CHANGED_DURING_PHASE47",
                         "t4_status": "T4_UNVERIFIED"})
    write_csv("baseline_rebenchmark.csv", baseline)
    raw_path = HERE / "onnx_graph_single_model_raw.csv"
    if not raw_path.exists():
        raw_path.write_bytes((HERE / "onnx_graph_optimization.csv").read_bytes())
    raw_graph = list(csv.DictReader(raw_path.open(newline="", encoding="utf-8")))
    graph_rows = [{"candidate": row["candidate"], "scope": "B01 single graph", "mean_ms": row["mean_ms"],
                   "max_probability_or_logit_diff": row["max_abs_logit_diff"],
                   "complete_five_model_saving_ms": "N/A", "decision": "LOCAL_ONLY"} for row in raw_graph]
    graph_rows.append({"candidate": "io_binding_five_model_paired", "scope": "five model full path paired",
                       "mean_ms": paired_binding["candidate"]["mean_ms"],
                       "max_probability_or_logit_diff": paired_binding["max_abs_probability_diff"],
                       "complete_five_model_saving_ms": paired_binding["paired_mean_saving_ms"],
                       "decision": "NO_GAIN_GPU_CONTENTION"})
    graph_rows.append({"candidate": "ORT_ENABLE_ALL_default", "scope": "verified SessionOptions default",
                       "mean_ms": "N/A", "max_probability_or_logit_diff": "N/A",
                       "complete_five_model_saving_ms": "N/A", "decision": "ALREADY_ENABLED"})
    graph_rows.append({"candidate": "CUDA_node_assignment", "scope": "five model node-level profiling",
                       "mean_ms": "N/A", "max_probability_or_logit_diff": "N/A",
                       "complete_five_model_saving_ms": "N/A", "decision": "PENDING_PROFILING_GPU_BUSY"})
    write_csv("onnx_graph_optimization.csv", graph_rows)
    consistency=[]
    for name, record in (("onnx_fp32_vs_frozen_e08",fp32),("onnx_fp16_vs_frozen_e08",fp16)):
        consistency.append({"comparison":name,"study_count":859,
                            "max_abs_probability_diff":record["max_abs_diff"],
                            "mean_abs_probability_diff":record["mean_abs_diff"],
                            "p99_abs_probability_diff":record["p99_abs_diff"],
                            "val_macro_auc":record["onnx_macro_auc"],
                            "auc_delta":record["macro_auc_delta"],
                            "patient_bootstrap_ci95_low":record["bootstrap_ci95"][0],
                            "patient_bootstrap_ci95_high":record["bootstrap_ci95"][1],
                            "status":"DISTINCT_NUMERICAL_VERSION"})
    consistency.append({"comparison":"preprocess_a_vs_phase45_fp32", "study_count":859,
                        "max_abs_probability_diff":a["max_val_probability_diff_vs_phase45_fp32"],
                        "mean_abs_probability_diff":0,"p99_abs_probability_diff":0,
                        "val_macro_auc":a["val_macro_auc"],"auc_delta":0,
                        "patient_bootstrap_ci95_low":0,"patient_bootstrap_ci95_high":0,
                        "status":"BITWISE_EQUAL_FULL_VAL_NO_SPEED_GAIN"})
    write_csv("probability_consistency.csv", consistency)
    per_class=[]
    for mode in ("fp32","fp16"):
        with (P44/f"probability_consistency_{mode}.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                per_class.append({"comparison":f"onnx_{mode}_vs_frozen_e08",**row})
    write_csv("per_class_auc.csv", per_class)
    trt=[]
    for precision in ("FP32","FP16"):
        trt.append({"precision":precision,"windows_4060_mean_ms":"N/A",
                    "val_macro_auc":"N/A","max_probability_diff":"N/A",
                    "engine_bytes":"N/A","status":"BLOCKED_NO_TENSORRT_RUNTIME_OR_TRTEXEC",
                    "t4_status":"T4_UNVERIFIED"})
    write_csv("tensorrt_precision_comparison.csv", trt)
    deployment = {"limit_bytes":500000000,
                  "original_pytorch_weight_plus_config_bytes":size["frozen_pytorch"]["weight_plus_config_bytes"],
                  "onnx_fp32_graph_bytes":size["fp32"]["graph_bytes"],
                  "onnx_fp32_conservative_required_bytes":size["fp32"]["deployment_total_bytes"],
                  "onnx_fp16_graph_bytes":size["fp16"]["graph_bytes"],
                  "onnx_fp16_conservative_required_bytes":size["fp16"]["deployment_total_bytes"],
                  "size_status":"PASS", "t4_latency_status":"T4_UNVERIFIED",
                  "loaded_graphs":size["fp32"]["graphs"],
                  "pretrained_model_license_status":"MANUAL_CONFIRMATION_REQUIRED"}
    (HERE/"deployment_size_report.json").write_text(json.dumps(deployment,indent=2),encoding="utf-8")
    (HERE/"environment_audit.md").write_text(
        "# Phase47 环境与规则核查\n\n"
        "- 本机 Windows、RTX 4060 Laptop 8GB，驱动 581.08；PyTorch 2.10.0+cu128、"
        "torchvision 0.25.0+cu128、CUDA 12.8、Pillow 12.3.0、NumPy 2.5.2。"
        "隔离目录的 ONNX Runtime GPU 1.23.2；原图 opset 17，输入 1×3×384×384。\n"
        "- 当前 ORT SessionOptions 默认 `ORT_ENABLE_ALL`。CUDA Provider 位于 session 第一位，"
        "但这不足以证明每个节点都在 GPU 上；逐节点 profiling PENDING。\n"
        "- Python TensorRT 与 `trtexec` 均不存在。虽然 ORT 构建列出 TensorRT EP，"
        "本机无法据此认定 TensorRT 引擎可用；未安装、未构建、未测速。\n"
        "- 跑完基准后发现另有 `F:\\glaucoma-bdc-2026\\.venv\\Scripts\\python.exe` "
        "进程使用 GPU；随后 4060 利用率约 95%、显存约 5.3GB。"
        "Phase47 后段分段和配对计时受资源竞争影响，不可与早段孤立基准作严格绝对值比较。"
        "没有停止该进程，也未继续做新 GPU 实验。\n"
        "- 分段墙钟计时见 `latency_breakdown.csv`：受竞争影响的一轮中，三次 DenseNet 前向"
        "各约 38.7–38.9ms，解码约 25.35ms、预处理约 32.76ms。"
        "ORT `session.run` 内部的 H2D/D2H 与计算尚未分离；独立 GPU kernel 时间未测，"
        "不得把这些前向墙钟数当作纯 GPU 计时或未受干扰的耗时占比。\n"
        "- [官方规则 PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)"
        "第6–8条允许固定参数前向与固定多模型集成，要求 T4 指定环境的 5 张预热、100 张固定图、"
        "batch=1、连续3轮，读图至最终结果写出；实际调用模型与必要配置解压后 ≤500MB。"
        "本地 Val 图片不是官方固定 Test 计时图。\n"
        "- 预训练权重上游许可证全文及原始文件 SHA 的最终材料复核仍未完成，"
        f"详见 `{P29/'pretrained_weights_compliance.md'}`。\n",
        encoding="utf-8")
    (HERE/"t4_deployment_instructions.md").write_text(
        "# T4 验收步骤（未执行）\n\n"
        "1. 在独立 Linux/Python 3.12 环境安装 `requirements_inference.txt`，"
        "并固定与官方指定 CUDA/cuDNN/驱动兼容的版本。复制 Phase45 `predict.py`、`cost.py`，"
        "五个 FP32 ONNX 图、导出哈希清单及 `src/cxr/data` 预处理代码；按原项目相对目录布置。"
        "切勿复制本机 Windows `isolated_deps` 作为 Linux 依赖。\n"
        "2. 按 SHA-256 验证所有图、代码、配置；确认五个 session 均使用 CUDAExecutionProvider，"
        "并运行单图及 859 Study Val 一致性检查。\n"
        "3. 使用官方 5 张预热与固定 100 张 Test 计时图，运行 "
        "`python cost.py submission.csv pictures_100 pictures_5 --test-archive /path/to/cxr_first.zip`。"
        "核对实际 T4 端到端均值 ≤100ms、100 张对应预测与提交 CSV 同一入口，"
        "并明确精度 FP32。\n"
        "4. 如尝试 TensorRT，必须在 T4 兼容环境针对 1×3×384×384 逐图构建五个引擎；"
        "重新跑完整 Val 数值/AUC 与 Test 预测并生成该引擎独立 Submission。"
        "4060 引擎不能视为 T4 合规证明。\n"
        "5. 当前 `cost.py` 对不完整 Study 样本只标记等价性未验证，"
        "不能把部分图像概率直接与 Study Mean Submission 比较。\n",
        encoding="utf-8")
    (HERE/"final_decision.md").write_text(
        "# Phase47 结论\n\n"
        "- 同口径本机 4060 参考均值：PyTorch FP16 188.74ms、ONNX FP32 126.83ms、"
        "ONNX FP16 130.26ms。比历史 164.83ms 高的差别可能受运行时条件影响，"
        "不能把不同时间结果混作严格配对。最早完成的三条基准未能证明 T4 合规。\n"
        "- 100 图微基准中原位归一化比原实现快约 3.65ms，完整 1,422 图输入和 "
        "859 Study 概率逐元素相同，Macro-AUC 0.81173315565。"
        f"但同进程交替五模型全流程平均节省 {paired_a['paired_mean_saving_ms']:.3f}ms，"
        "未确认端到端收益，故不替换正式路径。\n"
        "- 单图 B01 I/O Binding 微基准快约 0.50ms，五模型配对全流程"
        f"平均节省 {paired_binding['paired_mean_saving_ms']:.3f}ms，概率完全相同；不保留为候选。"
        "当前图优化已是 ORT_ENABLE_ALL，CUDA 节点级回退未完成 profiler 审核。\n"
        "- TensorRT FP32/FP16 均未安装、未构建、未测量，状态 BLOCKED。"
        "本机另一 GPU 任务已占用资源，停止进一步测速。\n"
        f"- ONNX FP32 相对冻结 E08 最大/平均/P99 概率差为 {fp32['max_abs_diff']:.6f}/"
        f"{fp32['mean_abs_diff']:.6f}/{fp32['p99_abs_diff']:.6f}；"
        f"Val AUC {fp32['onnx_macro_auc']:.9f}，患者配对差值95%CI "
        f"[{fp32['bootstrap_ci95'][0]:.9f},{fp32['bootstrap_ci95'][1]:.9f}]。"
        "Phase45 独立 FP32 入口 Val/Test 完整复现差异 0，但它不是旧 E08 的逐概率等价版本。\n"
        "- 五个 FP32 图及必要文件保守计 276,788,287 字节，低于 500MB；"
        "T4_UNVERIFIED。当前没有新的已验证加速候选，也不生成或上传新 Submission。"
        "下一步需在 GPU 空闲且无竞争时重测，再于目标 T4 实测官方固定样本，"
        "并补齐许可证与最终材料核验。\n",
        encoding="utf-8")
    files={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in HERE.iterdir()
           if p.is_file() and p.name!="manifest_sha256.json"}
    (HERE/"manifest_sha256.json").write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"NO_NEW_DEPLOYMENT_CANDIDATE","files":len(files),
                      "baseline_mean_ms":{x['candidate']:x['mean_ms'] for x in benchmarks},
                      "preprocess_paired_saving_ms":paired_a['paired_mean_saving_ms'],
                      "binding_paired_saving_ms":paired_binding['paired_mean_saving_ms']}))


if __name__=="__main__":
    main()
