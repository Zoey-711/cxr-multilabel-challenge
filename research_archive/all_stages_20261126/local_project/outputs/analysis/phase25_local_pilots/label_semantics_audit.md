# Phase25 标签语义审计

原始Train/Val CSV字段均为 `Dicom_id,Subject_id,Study_id,ViewPosition,PatientOrientation,Rows,Columns,Predict_class`；没有独立不确定标签字段。`Predict_class`是1–10的分号分隔阳性类别。30,344张Train、1,422张Val均成功解析，没有空白或非法类别；同一Study的多张图像标签一致。No Finding（类10）与其他阳性类别同时出现的Study数为0。

项目当前比赛标签协议为列出的类别=1、未列出的类别=0。`configs/f17_densenet121_fold5_384.yaml`和15份训练运行配置的 `unlisted_policy: negative` 一致；`src/cxr/data/dataset.py`为十类建立标签，并对该策略设置全部10个`valid_mask=True`；`scripts/train.py`和`batch_train.py`用同一mask归一化逐标签BCE。`batch_assets/train_labels.csv`与从原始ZIP逐Study解析的10类标签在18,513个Study上完全一致。这里的“0”是**比赛标注阴性**，不保证医学上绝对无病。`src/cxr/data/archive.py`解析器仅提取列出的阳性ID，其文档提及未列出类别的未知含义；最终训练含义由配置中的`unlisted_policy`确定。

| 类别 | Train阳性Study/患者 | Val阳性Study/患者 |
|---|---:|---:|
| 1 Enlarged Cardiomediastinum | 622 / 572 | 24 / 16 |
| 4 Pneumonia | 1,436 / 1,284 | 68 / 37 |
| 8 Lung Opacity | 4,508 / 3,295 | 214 / 72 |

Train共现：Consolidation(3)+Pneumonia(4)=202 Study，3+Lung Opacity(8)=242，4+8=716。Val相应为10、11、36。类别8支持数充足，但共现和此前较低AUC不能单独证明标签错误；类别1在Val只有16名阳性患者，估计波动较大。详细10类阳性支持和字段统计见`label_audit.json`。

比赛资料`docs/rules.md`记录了主办方提供的完整多标签标注口径：列出=1，未列出=0；未观察到独立不确定编码。目前未发现可复现的标签解析、损失掩码或No Finding冲突问题，因此没有依据修改标签或开展标签损失Pilot。此审计只核对机器可读字段，不能替代医学报告或人工影像重审。
