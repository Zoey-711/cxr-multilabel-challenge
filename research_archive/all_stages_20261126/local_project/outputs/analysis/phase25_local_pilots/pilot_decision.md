# Phase25 Pilot 决策

本轮**未启动GPU Pilot**。完整OOF的Mean可精确重建；Max与半Mean半Max下降。ViewBalanced增加0.000152 OOF AUC，五折和官方Val方向相同，但仅改变1,307/18,513 Study，改善幅度很小；现有标签编码也没有发现明确错误。没有足够理由花费4060时间再训练一个骨干或注意力头。依据为本目录的固定聚合CSV、2,000次患者级配对bootstrap、`label_audit.json`及既有M01/V00–V03报告。

下一步最有信息价值的是在未来真正独立的新患者结果上，按本次冻结的ViewBalanced规则做一次只读复核，同时确保有逐图概率和可信视角。朋友的26项计划只规定Study概率产物，若未保存逐图概率便无法复核；不能从Study概率反推。若未来有清晰且有实质规模的收益，再设计单Fold单变量训练Pilot。当前继续使用冻结Mean作为主协议，保留ViewBalanced为探索性候选，不修改已有Submission或融合权重。
