# 固定六模型：预测一致性规范化

本目录是独立实现，不替换历史E08、78.81 Submission或前次测速产物。
固定模型公式：`0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3`。

## 两道校验

1. **内部复现**：同一batch=1 CUDA Graph入口重复输出，以及此前全Val加速前后结果。内部复现要求完全一致。
2. **历史提交对应**：与用户确认78.81的原始Submission比较。SHA256固定为 `8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c`，不能换成新CSV来通过这一关。

历史对应必须同时满足：全部Study/类别/图片来源完整；概率最大绝对差≤1e-3；二值判定无变化；10类排序无变化。一项不满足即FAIL。0.5是既有审计中使用的固定二值阈值，PDF未给出本任务具体Gate定义；若官方提供其他判定接口，须按其接口再核验，不能凭该检查自称官方认定。排序采用概率降序、并列时按类别编号稳定排序，不对概率舍入、加epsilon或修补排序。

## 实现

- `predict.py`引用`cxr.e08_s17_consistent_entry`的模型加载、单图推理和Study Mean。
- `cost.py`直接导入同目录`predict.py`，调用其加载、推理和聚合函数。
- `src/cxr/prediction_consistency.py`集中维护映射、类别、数值、二值和排序校验。
- `locked_files.json`锁定六个checkpoint、代码、配置、全Test图像映射及历史Submission。任一锁定文件改变时拒绝运行。
- 元数据来自原始官方`cxr_first/test.csv`，2218张图像、1414个Study；不读取Test标签。
- `cost.py`检查三轮全部输出；不再只保存最后一轮用于核验。
- 所有Study Mean必须使用该Study的全部原图预测，不能把单图概率直接当作Study概率。
- 若100张样本中存在不完整Study，当前接口返回`SUBMISSION_CONSISTENCY_UNVERIFIED`和缺失同Study图像清单，不运行GPU测速、更不会只核对完整子集后报PASS。正式官方样本出现此情况时，需补齐独立一致性核验所需的同Study图片，并确认官方单图/Study接口对应方式；不得擅自替换官方100张样本。
- 计时从文件读取开始，包括六模型前向、融合、单图CSV写出、Study完成时的Mean聚合和最终Study结果写出，再CUDA同步。初始化和5张预热不计入。
- 100张×3轮必须产生300条计时及预测。即使抽样通过，如果完整Test审计失败，仍标记`BLOCKED_FULL_TEST_INCONSISTENCY`。

## 本地命令

从项目根目录执行。输出目录必须不存在，防止覆盖旧证据。

```powershell
.\.venv\Scripts\python.exe outputs/analysis/e08_s17_consistency_standardization_20261110/predict.py --config outputs/analysis/e08_s17_consistency_standardization_20261110/config.json --output outputs/analysis/e08_s17_consistency_standardization_20261110/test_audit
```

这条命令执行完整Test只读推理审计，输出图像概率、Study概率、差异明细和报告；不生成新的正式Submission。审计FAIL时返回退出码2，但保留真实结果。

```powershell
.\.venv\Scripts\python.exe outputs/analysis/e08_s17_consistency_standardization_20261110/cost.py --config outputs/analysis/e08_s17_consistency_standardization_20261110/config.json --pictures-100 <100张图片目录> --pictures-5 <5张独立预热图片目录> --submission outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv --output <新的测速输出目录>
```

当前配置引用本地依赖，尚不是自包含Linux交付包。T4部署需要复制锁定文件、改为相对路径并重新建立来源清单，在T4重新捕获CUDA Graph和复核结果。更改平台不能沿用4060测速作为验收。

## 测试与判断边界

CPU测试共18项通过：15项数值/映射测试，加3项真实cost循环的CPU模拟测试。后者核验一次加载、305次真实函数调用（5次预热+300次计时调用）、150条完整Study聚合输出、早期轮次失败阻断及完整Test审计失败阻断。CUDA在这3项测试中被mock，不是GPU测速。结果见`cpu_regression.json`和`cpu_regression.log`。

`val_reaudit.json`用新校验器复核已锁定的859 Study全Val推理结果，不虚称重新跑了GPU。新Test真实推理见`test_audit/audit_report.json`。

本次真实Test结论：全部2218张/1414 Study完成，历史提交一致性FAIL；10个Study排序改变，概率最大误差0.00091552734375，未超过1e-3，0.5二值判定变化为0。严格规则要求同时满足，不能只用概率误差达标放行。详见`final_decision.md`。

通过内部复现不能替代历史Submission对应关系；通过本地核验不能替代T4验收。没有调整权重、预处理参数、融合比例或预测数值来迎合Test结果。
