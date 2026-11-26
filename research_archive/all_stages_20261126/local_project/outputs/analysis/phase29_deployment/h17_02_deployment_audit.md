# H17-02 部署审计

- 朋友结果来源 commit `cbfc5997438f371086cd1fbe232890590cd67ffb`，任务 `H17_DISTILL_02`；配置、Val 概率及 `run_manifest.json` 已存在于 `E:\计挑赛\cxr-friend-results\versions\cbfc5997438f371086cd1fbe232890590cd67ffb\results\phase17_batch\experiments\H17_DISTILL_02`。训练源代码 commit `8b7944d744f91e753b912367adeb354e8b7e21d5`。
- 冻结 manifest 中纯推理权重 `best_model.pth` 为 **81,651,999 字节**，SHA-256 `35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69`。已从私有 Release **只下载这一份权重**至 `E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\phase29_deployment\weights\H17_DISTILL_02_best_model.pth`，本地实测长度及 SHA-256 均匹配；连同现有配置为 **81,654,329 字节**，低于 500,000,000 字节。未经授权未下载其他朋友权重。
- 859 个相同 Val Study 的 H17-02 Macro-AUC **0.808613911**，E08 **0.811668646**，差值 **-0.003054735**。228 患者配对 Bootstrap（seed 44，2000 有效，0 无效）的 95% 区间 **[-0.009454, +0.004556]**，不能认为有可靠差异。
- 相对 E08，下降最多的类别：1 Enlarged Cardiomediastinum -0.0318, 4 Pneumonia -0.0200, 2 Pneumothorax -0.0071。完整十类在 `h17_e08_per_class.csv`。单个 Val 多次用于模型选择，该比较为开发性证据。
- 从本地权重独立重算 1422 张 Val 图、859 个 Study，batch8 得 Macro-AUC **0.808607304**，保存的朋友 batch16 概率对应 **0.808613911**；最大概率差 0.000488281。本地 RTX4060 同 E08 的 100 图 × 3 轮测速：均值 **100.38ms**，P50 100.46ms、P95 113.55ms、最大 119.27ms。batch1 与冻结单图 Study 概率最大差 0.000732422，超过 1e-4 一致性门槛。
- 朋友历史 RTX4090 研究测速：20 图、3 预热、均值 **59.90ms**；口径不同。以上两种设备数据都**不能证明 T4 ≤100ms**。本地 4060 均值略高于 100ms，H17 部署状态仍为 **T4_UNVERIFIED**。
- H17-02 使用全部 Train，故没有独立学生 OOF；其教师软标签来自冻结完整 OOF。朋友使用比赛数据的授权及其训练数据声明需人工核实，不能仅凭 manifest 视为最终合规。
