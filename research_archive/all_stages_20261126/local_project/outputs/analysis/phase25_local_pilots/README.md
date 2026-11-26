# Phase25 可复算命令

在项目根目录运行，无需GPU：

```powershell
.\.venv\Scripts\python.exe -m cxr.engine.phase25_audit --config configs\phase25_aggregation_audit.yaml
.\.venv\Scripts\python.exe -m cxr.engine.phase25_val_check --config configs\phase25_aggregation_audit.yaml
```

第一条命令要求输出目录尚不存在，以防覆盖本次结果；若需要独立复算，请复制配置并将`output`改为新的目录。第二条命令只在既有输出目录中新增`aggregation_val_check.csv`，已存在时拒绝覆盖。两条命令只读取冻结模型预测与官方Train/Val元数据，不加载模型、不调用CUDA。
