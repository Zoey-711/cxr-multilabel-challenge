# 本地历史实验代码快照（2026-10-26）

这是 `cxr-multilabel-challenge` 本地工作区的**代码与配置归档**，用于备份并让合作者查看此前本地开展的实验。归档没有并入 `friend-gpu-batch` 的26项、Phase32、E08-Tune或Phase35训练队列；克隆仓库不会自动运行这些历史脚本。

归档覆盖 `scripts/`、`src/`、`configs/`、`docs/`、`tests/`、`requirements/`，以及保存在 `outputs/analysis/`、`outputs/experiments/` 下的 Python实验脚本和YAML配置。包括早期B/M/N/T实验、Phase17五折、Study聚合/蒸馏/辅助监督、Phase18–31分析与部署、Phase33和Phase39等本地研究代码。`outputs/`中只选取源码和配置；未复制CSV预测、标签、胸片、模型权重、训练日志、Submission或任何结果包。

`archive_manifest.json`逐项记录原工作区文件路径、原始SHA-256、上传快照SHA-256、字节数及本机路径替换次数。部分历史源码包含本机盘符、旧云端SSH主机或授权数据路径，快照中已将其替换为占位符；这不改变本地原件，但意味着该快照**不是可以原样启动的训练包**。复现时应检查对应实验配置、数据授权、预训练权重、患者折索引和历史依赖。不同阶段脚本的接口也可能不同，不应把快照中的旧实验直接加入当前无人值守队列。

本地仍在运行的Phase39实验采用主项目原文件；本归档复制的是读取时的源码，归档操作不会改变它的训练进程。Phase39可移植版本另见仓库当前的 `src/cxr/phase39_*`、`configs/phase39/` 和 `docs/phase39_local_code.md`。

校验快照：逐项计算文件SHA-256，并与 `archive_manifest.json` 的 `archived_sha256` 比较。`original_sha256`用于证明其来自本地原件；有路径脱敏的文件两种哈希不同是预期行为。
