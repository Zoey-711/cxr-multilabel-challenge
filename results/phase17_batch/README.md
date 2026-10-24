# Phase 17：26 项实验结果（2026-10-24）

26 项正式实验全部完成，无失败或阻塞项。最终汇总生成时间：北京时间 2026-10-24 12:00。

训练代码提交：`8b7944d744f91e753b912367adeb354e8b7e21d5`。服务器已通过完整 REMOTE_PREFLIGHT，包含三项 CUDA 测试、真实数据前反向与恢复、Val/Test 短推理、六条 G/H 路径和 1/2/4/8 卡并发检查。

- [最终报告](reports/phase17_final_report.md)
- [基础 22 项指标](reports/experiment_summary.csv)
- [G/H 四项指标](reports/gh_experiment_summary.csv)
- [26 项完成状态](reports/one_launch_status.json)
- [Submission 候选与状态](submissions/submission_candidates.csv)
- [交付验证清单](manifests/delivery_manifest.json)
- [本次上传文件及 SHA-256](github_upload_manifest.json)

本目录包含报告、指标、Val/OOF/Test 预测、训练日志、配置及产物校验清单。134 个模型权重与恢复文件约 35.66 GB，保留在服务器 `/home/zlhll/cxr_results/phase17_batch/`，未放入普通 Git 历史；文件列表见 [binary_artifacts.json](binary_artifacts.json)。原始胸片数据未包含在本目录中。

Test 没有公开标签，因此没有 Test AUC。G/H 全 Train 任务的 OOF 为 N/A；五折 OOF 由 D/C/S/N 的各五折模型提供。具体指标、研究测速和部署资格均以原始报告为准。
