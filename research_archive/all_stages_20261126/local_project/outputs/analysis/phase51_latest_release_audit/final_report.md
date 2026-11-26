# Phase51：Phase41R 修复结果验收

## 来源和完整性

- Private Release：`phase41r-recovery-v1`，2026-10-29 05:44:03 UTC，附件 37 个；完整附件名称、大小、状态和 GitHub digest 见 `release_metadata.json`。
- 代码版本：`4392f581a5daec87bc1ef66159669a1b3a4df48b`；独立结果快照分支 `results/phase41r-recovery`，commit `47cd764a4ae4414eedbaaeb08946562cf530550a`。
- 结果目录 `E:\计挑赛\cxr-friend-results\snapshots\phase41r-47cd764a\results\phase41r_batch`。轻量 export manifest 的 130 项全部通过本机 SHA-256；Release 的 `binary_release_manifest.json` 本机哈希与 GitHub digest 一致。
- 8 项状态 DONE、0 FAILED：BASE F2 为复用此前成功的历史结果，BASE F3–5、FUSION F2–5 共 7 项修复后重跑。BASE/FUSION F1 的原始 checkpoint、OOF、配置及 manifest 与本地冻结原件及复用清单 SHA 匹配。
- 其余 Release 模型二进制未下载；其大小及哈希核对到 GitHub 资产元数据和 DONE 清单，不代表已在本机对下载权重逐字节验证。所有预测均校验 Study 键、10 类概率范围及每折患者归属；Test 每折 1,414 Study、无重复，未使用 Test 标签。训练/验证患者隔离依据冻结 Fold 映射和配置，不是对训练过程的独立观测。

## 修复与训练协议

7 个重跑任务各记录一次 AMP 溢出跳过，和训练历史、事件日志一致；每项约 15,159–16,708 次成功 optimizer 更新，没有发现大规模静默跳步。短步诊断为 32 次更新、0 跳步，checkpoint 恢复及相同 batch 重放通过。修复处理了已观测故障并使任务完成，不能证明未来不会出现非有限梯度。其余超参数、Fold 索引哈希、ImageNet 初始化哈希、微批量 4、梯度累积 4、EMA 等与 Fold1 配置匹配；修复改变 AMP 溢出处理，4060 与 4090 浮点执行不同，因此不是完全相同的训练实现或轨迹。结果属混合环境的配对研究。

## 独立复算

| 指标 | BASE | FUSION |
| --- | ---: | ---: |
| 完整五折 OOF | 0.806439 | 0.808267 |
| Fold2–5 合并 OOF | 0.807001 | 0.808579 |
| 五折概率平均 Val，859 Study | 0.806989 | 0.806540 |

完整 OOF 覆盖 18,513 Study、11,055 患者，每位患者只属于一个冻结 Fold。FUSION−BASE 完整 OOF 为 +0.001827，2,000 次患者配对 Bootstrap 95% CI [0.000382, 0.003274]；Fold2–5 差值 +0.001578，95% CI [-0.000084, 0.003367]，确认折区间包含 0，尚不能确认结构收益。随机种子 4143，抽样患者簇；任一类缺少正负样本时排除该次抽样，本次 2,000 次全部有效。

远端 FUSION 五折 Val 表报告 0.806538813，本机对五份 Val 逐项平均得到 0.806540150。两份概率最大差仅 1.11e-16，第三类有并列排序受浮点舍入影响；属于极小计算口径差异，不能以此解释模型效果。每折 AUC、每类 AUC 和阳性支持数分别见 `fold_auc.csv`、`per_class_auc.csv`；完整输入和 Bootstrap 结果见 `audit_results.json`。

同 859 Study 的冻结参考：E08 0.811669、H17 batch1 0.808610、H17 ONNX 0.808716；这些是已有结果，非本轮新模型 OOF。FUSION 五折 Val 低于 E08 约 0.00513，不意味着隐藏 Test 有同样差距。Val 已被重复用于开发评估。

## 部署和决策

根据 Release 元数据，BASE 五个 EMA 权重共 408,318,591 字节，FUSION 共 408,607,571 字节；两者单独都小于 500,000,000 字节，必要配置及正式加载文件仍需计入。每个单模型约 81.7 MB；融合结构推理时须保留中间特征分支。ONNX 导出和端到端性能尚未在该模型验证。无 T4 实测，五模型和单模型均为 `T4_UNVERIFIED`；五模型推理成本有明显风险。

建议保留 E08 作为当前官方成绩基线、H17 batch1 作为轻量部署基线。Phase41R FUSION 的完整 OOF 有探索信号，但未通过后四折确认，暂不推荐以此挑战 E08，也不建议继续扩大该结构训练；可保留模型与结果，待其他独立证据及真实部署测试再评估。未生成或上传新 Submission。
