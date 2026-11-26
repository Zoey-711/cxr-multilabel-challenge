# 78.81历史数值路径核验

从项目根目录运行，使用已安装的项目`.venv`。以下命令不会覆盖旧产物；输出目录已存在会拒绝执行。

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m cxr.historical_val_path_recovery --config outputs/analysis/e08_s17_historical_val_recovery_20261110/config.json
```

该命令先在1422张官方Val图像上验证历史batch=16/8 FP16来源，再运行四个预先固定的真正batch=1精度方案。每项报告859个Study及六模型的概率差异、排序、判定和AUC。所有参数与参考哈希在`config.json`，诊断结果在`run/result.json`。

只有`selected_policy`非空且Val状态PASS，才可运行`cxr.historical_test_path_confirmation --config ...`，对唯一预选batch=1策略进行Test核验。**本次selected_policy为null，该脚本会拒绝执行。**

另有只读来源核验：

```powershell
.\.venv\Scripts\python.exe -m cxr.historical_test_provenance --config outputs/analysis/e08_s17_historical_val_recovery_20261110/config.json
```

它仅重放已在Val逐值确认的原始batch=16/8流程，在全Test上确认历史CSV来源。报告在`historical_test_provenance/report.json`。该命令并非官方batch=1性能测试，不可用其耗时证明100ms合规。

细节和限制见`protocol.md`、`final_decision.md`。本地规则文件来自`F:/计挑赛/人工智能规则/比赛参赛规则.pdf`及`关于比赛参赛规则若干事项的补充说明 (1).pdf`。
