# phase35_batch：完整结果

正式任务已完成；训练代码 `046ab1bb240139f4243b564202efc0203af32cad`。原始服务器结果保持完整。

- [报告目录](reports/)
- [全部权重、恢复检查点及预检状态](https://github.com/Zoey-711/cxr-multilabel-challenge/releases/tag/phase35-fivefold-v1)
- [交付清单与 SHA-256](github_delivery_manifest.json)

权重通过 Release 按原文件交付；没有压缩打包。使用有读取权限的 GitHub CLI 登录后，可用 Python 3.9+ 按原目录下载并核对 SHA-256：

```bash
python3 download_full_weights.py --output /path/to/cxr_results
```

该命令恢复完整二进制结果到输出目录下的 phase35_batch/。已有正确文件会跳过，内容不符的文件不会被覆盖。

两个方案各复用冻结 Fold1、补齐 Fold2–5；完整 OOF 及 Fold2–5 确认结果按报告区分。不包含原始影像，没有 Test AUC，也没有 T4 延迟合规结论。
