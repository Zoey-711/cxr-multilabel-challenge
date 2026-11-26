# E03 + H17 固定替换：真实Val复现验收

## 实际完成
原始Val图像1422张，859 Study，228患者；两遍完整CUDA推理。未读取Test、未训练、未生成Submission。
固定公式：0.5 × E03 + 0.5 × H17_BATCH1；E03 = 0.5 × (B01+B02+B03)/3 + 0.5 × M02。
最终入口：`cxr.inference.e03_h17_validation.predict_image`；batch=1，FP16 autocast，原FP32权重，384 black_pad_v1，Study Mean，无随机增强。

## 概率与AUC
真实入口Val Macro-AUC：**0.8130544037**；历史缓存固定融合：0.8130579943。
相对冻结E08：+0.0013857576；2000次患者级配对Bootstrap 95% CI：[-0.0026906649, +0.0051743654]。
Bootstrap固定seed42，按患者重采样，任一类无正/负样本则丢弃整次抽样；有效2000次。
反复使用的官方Val仅属探索性开发证据，不是完整OOF或独立泛化确认。

|组成/融合|新AUC|缓存AUC|最大绝对差|平均绝对差|P99差|历史1e-4门槛|
|---|---:|---:|---:|---:|---:|---|
|B01|0.7884028032|0.7883919646|0.00048828125|3.09167095e-05|0.000244140625|FAIL|
|B02|0.7893758026|0.7893893728|0.00048828125|2.96768155e-05|0.000244140625|FAIL|
|B03|0.7894184048|0.7894350141|0.00048828125|3.24743776e-05|0.000244140625|FAIL|
|M02|0.7972364687|0.7972442723|0.00048828125|2.32652518e-05|0.000244140625|FAIL|
|H17_BATCH1|0.8086102168|0.8086102168|0|0|0|PASS|
|E03_H17|0.8130544037|0.8130579943|0.000172932943|1.05493918e-05|8.13802083e-05|FAIL|

最终融合最大差所在：subject_11827 / study_20899 / 类别8。
全部1422图与冻结H17预处理逐张张量最大差：0.0。
新入口两遍全Val最大差：0；事先固定内部门槛1e-6：**PASS**。
Study Mean与按组成模型聚合后固定融合的交换一致性已检查（误差<1e-14）。类别固定1–10，概率有限且在[0,1]，ID覆盖与原始Val映射一致。

## 来源及差异解释
五个checkpoint和必要来源配置逐字节SHA-256均核验，推理结束后再次核验，原文件未改动。
E03四个模型来自冻结E08 manifest。H17来自冻结Phase31 H17_02_T4_candidate，使用相同纯state_dict与模型构造。
历史E03实际批量为DenseNet 16、ConvNeXt 8（Phase20记录），不能仅依据配置中的推理batch_size=16推断所有模型实际批量。
当前候选统一batch=1；历史比较失败时不得称与缓存数值等价，不放宽1e-4门槛、不人为修改概率。
历史批量诊断：{"protocol": "E03 historic B01/B02/B03 batch16, M02 batch8; actual new H17 batch1; diagnostic only", "components": [{"name": "B01", "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "auc": 0.7883919646087458, "reference_auc": 0.7883919646087458, "strict_1e4_pass": true, "worst_subject": "subject_11813", "worst_study": "study_20000", "worst_class": 1}, {"name": "B02", "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "auc": 0.7893893728132187, "reference_auc": 0.7893893728132187, "strict_1e4_pass": true, "worst_subject": "subject_11813", "worst_study": "study_20000", "worst_class": 1}, {"name": "B03", "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "auc": 0.7894350140517664, "reference_auc": 0.7894350140517664, "strict_1e4_pass": true, "worst_subject": "subject_11813", "worst_study": "study_20000", "worst_class": 1}, {"name": "M02", "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "auc": 0.7972442723187514, "reference_auc": 0.7972442723187514, "strict_1e4_pass": true, "worst_subject": "subject_11813", "worst_study": "study_20000", "worst_class": 1}], "fusion": {"name": "E03_H17_historical_batches", "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "auc": 0.8130579943218, "reference_auc": 0.8130579943218, "strict_1e4_pass": true, "worst_subject": "subject_11813", "worst_study": "study_20000", "worst_class": 1}, "elapsed_seconds": 80.06300000008196, "note": "Does not change final batch1 implementation or its acceptance tolerance."}

## 决策
内部复现与来源/输入校验通过，可作为独立batch=1候选继续讨论；历史缓存数值等价状态为 **FAIL**。
不将历史缓存与新入口混用，不将概率数值变化解读为泛化提升。当前没有Test CSV，也没有READY_TO_SUBMIT标记。
原E08仍保留为官方77.75基准；新候选的Val点估计不能保证官方提分。

## 资源与部署边界
五个实际权重278,421,790字节；权重加来源配置278,429,724字节，低于500,000,000字节。
两次实际推理与输入比较耗时372.2秒，不作为官方单图测速。显存峰值未成功持久化，标记UNAVAILABLE。
两遍预测写盘后，报告保存遇到NumPy整数JSON序列化错误；仅将报告类别编号转为Python int后从已保存预测进行CPU复算，未改动任何预测。原执行源码快照与修复后的源码均保留并校验单行差异。
这不是官方测速，未给出100ms结论。用户报告原E08 T4通过保持不变；新替换候选仍T4_UNVERIFIED。
本机目录没有可用Git仓库元数据，不能编造新Git commit；以代码快照、配置及SHA-256固定本次入口。H17历史来源commit见原部署配置。
