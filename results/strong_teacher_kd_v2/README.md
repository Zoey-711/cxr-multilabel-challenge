# strong_teacher_kd_v2 完整结果

已完成本批全部 6 项正式训练。源代码 `474b175c34167cb52b92f7e4fe8150fe2acef3f7`。

- [报告目录](reports/)
- [权重、恢复检查点和预检文件](https://github.com/Zoey-711/cxr-multilabel-challenge/releases/tag/strong-teacher-kd-v2)
- [完整交付文件与 SHA-256](github_delivery_manifest.json)
- [完整二进制资产清单](binary_release_manifest_full.json)

文件逐项交付，无压缩。原始服务器结果保留；原始胸片和原始标签文件未包含在本次结果中。

使用有仓库读取权限的 GitHub CLI 登录后，可用 Python 3.9+ 下载全部二进制文件并恢复原目录：

```bash
python3 download_full_weights.py --output /path/to/cxr_results
```

脚本先核对 SHA-256，跳过已匹配文件，并保留内容不符的现有文件。

学生 OOF 为 N/A；Val 比较是探索性分析，未生成 Test 或 Submission，也没有 T4 延迟合规结论。
