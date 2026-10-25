# phase32_batch：完整结果

正式任务已完成；训练代码 `01fa68bdac65ce56d04a1fc99e8f5996ab09aa20`。原始服务器结果保持完整。

- [报告目录](reports/)
- [全部权重、恢复检查点及预检状态](https://github.com/Zoey-711/cxr-multilabel-challenge/releases/tag/phase32-e08-results-20261025)
- [交付清单与 SHA-256](github_delivery_manifest.json)

权重通过 Release 按原文件交付；没有压缩打包。使用有读取权限的 GitHub CLI 登录后，可用 Python 3.9+ 按原目录下载并核对 SHA-256：

```bash
python3 download_weights.py --output /path/to/cxr_results
```

该命令恢复两批权重到输出目录下的 phase32_batch/ 与 e08_tuning/。已有正确文件会跳过，内容不符的文件不会被覆盖。

E08 仅按预设门槛补折；单折与完整五折结果按报告区分。不包含原始影像，没有 Test AUC，也没有 T4 延迟合规结论。
