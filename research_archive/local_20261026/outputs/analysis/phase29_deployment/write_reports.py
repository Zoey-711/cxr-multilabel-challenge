"""Generate Phase29 audit reports from verified local JSON and frozen manifests."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FRIEND = Path("/LOCAL_WORKSPACE/cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/experiments/H17_DISTILL_02")
PDF = "https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf"

def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8-sig"))

def write(name, value):
    (OUT/name).write_text(value.strip()+"\n", encoding="utf-8")

def main():
    size=read("model_size_report.json")
    val=read("val_comparison.json")
    cost=read("cost_protocol_report.json")
    hrun=json.loads((FRIEND/"run_manifest.json").read_text(encoding="utf-8"))
    hbench=json.loads((FRIEND/"inference_benchmark.json").read_text(encoding="utf-8"))
    hcost=read("h17_cost_protocol_report.json")
    hverify=read("h17_weight_val_verification.json")
    with (OUT/"h17_e08_per_class.csv").open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    weak=sorted(rows,key=lambda r:float(r["h17_minus_e08_auc"]))[:3]
    weight_local=bool(size["H17_02"]["local_weight_available"])
    write("e08_deployment_audit.md", f"""
# E08-EMA 部署审计

- 官方规则：{PDF}（第 1、7、8 条及附录 A）。主办方目标硬件为 T4 16GB；单图 batch=1、5 张预热、100 张正式、连续 3 轮，读图至写出最终结果。官方上限为 **≤100ms**。
- 冻结来源：`{ROOT / 'outputs/submissions/E08_EMA_20261020/manifest.json'}`；5 个推理 checkpoint 的文件长度和 SHA-256 与 manifest 逐一一致，5 份配置哈希也一致。完整权重 **{size['E08_EMA']['weight_bytes']:,} 字节**，连同现有配置 **{size['E08_EMA']['weight_plus_config_bytes']:,} 字节**，低于 500,000,000 字节。仍需对最终材料目录全部必要文件重新计数。
- 公式：`E01=(B01+B02+B03)/3`; `E03=0.5*E01+0.5*M02`; `E08=0.5*E03+0.5*N01_EMA`。所有模型 384×384、`black_pad_v1`、ImageNet RGB 归一化、FP16 autocast、Study Mean。
- Phase20 的独立共享预处理入口在 `outputs/analysis/phase20_inference_optimization/optimized_inference/shared_preprocess.py`；其全量 1422 图/859 Study 检验在 `prediction_equivalence.json` 为 `accepted=true`、概率差 0。**该实现未并入正式 `scripts/predict.py`。** 本目录 `deployment/predict.py` 是独立副本，读图/预处理只做一次，五模型共同使用同一张量。
- 本次 4060 测速来自官方 **Val** 中 100 个不同单图 Study，原图文件读入到 CSV 写出；5 张另选的 Val 图预热、100 张图 × 3 轮。均值 **{cost['mean_ms']:.2f}ms**、P50 {cost['p50_ms']:.2f}ms、P95 {cost['p95_ms']:.2f}ms、最大 {cost['max_ms']:.2f}ms。结果是本地参考，**T4_UNVERIFIED**；所用图片也不是主办方正式固定 Test 计时图。
- `cost.py` 直接调用 `predict.py` 的 `load_models()` 与 `predict_image()`。batch=1 图像输出对照冻结单图 Study 的 Val 概率最大差 **{cost['submission_max_abs_probability_diff']:.9f}**，高于事先采用的 1e-4 门槛；100/100 Study 完整参与比较。推测与原验证批量 8/16 和单张前向的数值计算路径有关，尚未通过同一预测一致性门禁。**不能宣称已达到最终提交文件一致性要求。**
- E08 冻结验证 Macro-AUC {val['e08_macro_auc']:.9f}，官方排行榜 77.75（历史成绩，非本次测速得出）。正式部署仍需 T4 上以官方 5+100 图片执行 `cost.py`，并验证预测与最终 `submission.csv`；如采用不同精度/实现，还需全量 Val 概率与 AUC 复算。
""")
    write("h17_02_deployment_audit.md", f"""
# H17-02 部署审计

- 朋友结果来源 commit `cbfc5997438f371086cd1fbe232890590cd67ffb`，任务 `H17_DISTILL_02`；配置、Val 概率及 `run_manifest.json` 已存在于 `{FRIEND}`。训练源代码 commit `{hrun['git_commit']}`。
- 冻结 manifest 中纯推理权重 `best_model.pth` 为 **{hrun['files']['best_model.pth']['bytes']:,} 字节**，SHA-256 `{hrun['files']['best_model.pth']['sha256']}`。已从私有 Release **只下载这一份权重**至 `{OUT/'weights/H17_DISTILL_02_best_model.pth'}`，本地实测长度及 SHA-256 均匹配；连同现有配置为 **{size['H17_02']['weight_plus_config_bytes']:,} 字节**，低于 500,000,000 字节。未经授权未下载其他朋友权重。
- 859 个相同 Val Study 的 H17-02 Macro-AUC **{val['h17_macro_auc']:.9f}**，E08 **{val['e08_macro_auc']:.9f}**，差值 **{val['h17_minus_e08']:+.9f}**。228 患者配对 Bootstrap（seed 44，2000 有效，{val['invalid']} 无效）的 95% 区间 **[{val['ci95'][0]:+.6f}, {val['ci95'][1]:+.6f}]**，不能认为有可靠差异。
- 相对 E08，下降最多的类别：{', '.join(f'{r['class_id']} {r['class_name']} {float(r['h17_minus_e08_auc']):+.4f}' for r in weak)}。完整十类在 `h17_e08_per_class.csv`。单个 Val 多次用于模型选择，该比较为开发性证据。
- 从本地权重独立重算 1422 张 Val 图、859 个 Study，batch8 得 Macro-AUC **{hverify['recomputed_macro_auc']:.9f}**，保存的朋友 batch16 概率对应 **{hverify['reference_macro_auc']:.9f}**；最大概率差 {hverify['max_abs_probability_diff']:.9f}。本地 RTX4060 同 E08 的 100 图 × 3 轮测速：均值 **{hcost['mean_ms']:.2f}ms**，P50 {hcost['p50_ms']:.2f}ms、P95 {hcost['p95_ms']:.2f}ms、最大 {hcost['max_ms']:.2f}ms。batch1 与冻结单图 Study 概率最大差 {hcost['submission_max_abs_probability_diff']:.9f}，超过 1e-4 一致性门槛。
- 朋友历史 RTX4090 研究测速：20 图、3 预热、均值 **{hbench['research_single_model']['mean_ms']:.2f}ms**；口径不同。以上两种设备数据都**不能证明 T4 ≤100ms**。本地 4060 均值略高于 100ms，H17 部署状态仍为 **T4_UNVERIFIED**。
- H17-02 使用全部 Train，故没有独立学生 OOF；其教师软标签来自冻结完整 OOF。朋友使用比赛数据的授权及其训练数据声明需人工核实，不能仅凭 manifest 视为最终合规。
""")
    write("pretrained_weights_compliance.md", f"""
# 预训练权重来源与许可核验

官方 PDF 第 1、2 条要求非医学开源权重、可用于比赛并允许再分发，最终提供公开链接、许可证全文、上游数据说明及 SHA-256：{PDF}。

| 架构 | torchvision 权重 | 原始公开 URL | 训练 manifest 记录 SHA-256 | 本地原始权重文件复核 |
|---|---|---|---|---|
| DenseNet-121 | IMAGENET1K_V1 | https://download.pytorch.org/models/densenet121-a639ec97.pth | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` | PENDING：原始权重未定位 |
| ConvNeXt-Tiny | IMAGENET1K_V1 | https://download.pytorch.org/models/convnext_tiny-983f1562.pth | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` | PENDING：原始权重未定位 |
| EfficientNetV2-S | IMAGENET1K_V1 | https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth | `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74` | PENDING：原始权重未定位 |

来源：本地 B01/M02/N01 的 `pretrained_source.json`，朋友 H17-02 的 `run_manifest.json`。torchvision 官方分类权重表注明这三种 ImageNet-1K 权重：https://docs.pytorch.org/vision/main/models 。torchvision 0.25.0 包内 BSD-3-Clause 全文已复制为 `torchvision_LICENSE_BSD3.txt`（SHA-256 `c06363f9d33627dc5173b13522444cc85fbb4739f63e16d4587d5c0a165b5b1e`），官方仓库许可：https://github.com/pytorch/vision/blob/main/LICENSE 。**软件仓库许可证不自动证明每份预训练权重的再分发权利；最终材料需核对权重适用许可及上游 model card/训练 recipe，并附全文。**

推理 checkpoint 的冻结 SHA 已核实；原始 ImageNet checkpoint 在本机未定位，因此这里只能确认训练记录所填哈希，不能声称完成原始预训练文件复核。朋友的额外医学数据/授权使用情况标记 MANUAL_CONFIRMATION_REQUIRED。
""")
    write("final_submission_materials_checklist.md", f"""
# 最终提交材料整改清单

依据：{PDF} 第 7、8、10 条及附录 A。此处只审计，不生成或提交最终材料包。

| 材料/条件 | E08 | H17-02 |
|---|---|---|
| 历史最高成绩 `submission.csv` | 已有冻结 E08 文件及哈希；本轮未改动 | 已有朋友候选 CSV，尚非当前最高成绩 |
| 最终推理纯权重与 SHA | 5 份均 PASS | 单个纯权重已下载并通过本地 SHA 校验 |
| 必要配置、类别表及 500MB 计数 | 已核配置哈希和 278,442,617 字节；最终材料目录仍需重算 | 本地权重加现有配置 {size['H17_02']['weight_plus_config_bytes']:,} 字节；最终材料目录仍需重算 |
| `predict.py`/`cost.py` 同一模型入口 | Phase29 独立副本已实现；未合入正式路径 | 同一入口已完成 859 Val 和本地测速 |
| 官方 5+100×3、T4 16GB、batch1 | PENDING | PENDING |
| cost 结果与最终 submission 一致 | 本地单图 Val 最大差 0.000793 > 1e-4，尚未通过 | 本地单图 Val 最大差 {hcost['submission_max_abs_probability_diff']:.6f} > 1e-4，尚未通过 |
| `train.py`、`model.py`、`utils.py`、`requirements.txt` 可复现代码 | 原项目有相关源码；材料目录尚未组装/验收 | 需取得朋友训练代码对应 commit |
| 原始预训练权重及 SHA | 训练记录有来源哈希，本地原始文件待定位 | 同左 |
| 许可证全文、上游训练数据 model card/论文 | 已取得 torchvision 软件 BSD-3 原文；逐权重再分发许可/说明待确认 | 同左 |
| 模型核验材料与数据申明清单.xlsx | PENDING | PENDING |
| 朋友使用比赛数据的授权与数据声明 | 不适用本地训练 | MANUAL_CONFIRMATION_REQUIRED |
| 最终提交材料包与主办方复测 | PENDING | PENDING |

官方 PDF 要求材料包中含 `code/`、`model/`、`pretrain_model/`、`licenses/`、`cost.py`、核验 Excel 和历史最高分 `submission.csv`。现有 Phase29 目录是审计副本，**不是已验收的最终材料包**。
""")
    write("phase29_decision.md", f"""
# Phase29 决策

1. **E08 优先保留。** 冻结 5 权重 278,436,014 字节，计入现有配置 278,442,617 字节，满足 500MB；Val 0.811669，官方 77.75。4060 官方形式参考测速均值 {cost['mean_ms']:.2f}ms，T4 尚无实测。batch1 与冻结概率相差最大 {cost['submission_max_abs_probability_diff']:.6f}，必须解决提交一致性门禁。
2. **H17-02 是轻量备选，但尚不能判定可部署。** 单权重本地 SHA 已核实，81,651,999 字节；Val 0.808614，差值 95% CI 跨零。本地 4060 100 图 × 3 轮均值 {hcost['mean_ms']:.2f}ms，贴近并略高于 100ms；不能替代 T4 实测。
3. 截止前的先后顺序：在目标 T4 上按官方 5+100×3、batch1 对 E08 和 H17-02 实测；让 `cost.py` 对每个完整 Study 的结果与最终 CSV 比较并确认容差；整理许可证/上游数据/原始权重哈希/数据使用授权及最终材料清单。
4. 本轮无训练、无 Test 标签使用、无 Submission 改写或上传；所有输出位于本独立目录。完整 Val 对照结果见 `val_comparison.json`，十类见 `h17_e08_per_class.csv`。
""")
    print(json.dumps({"reports":5,"e08_ms":cost['mean_ms'],"h17_ms":hcost['mean_ms'],"h17_weight_local":weight_local},ensure_ascii=False))

if __name__=="__main__":main()
