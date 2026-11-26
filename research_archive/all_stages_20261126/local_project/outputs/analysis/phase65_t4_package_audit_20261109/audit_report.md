# Phase65 T4包只读审计

结论：T4参考测速有可复核证据，大小低于500MB；官方完整合规仍待验收，不应标记OFFICIAL_PASS。

来源：F:/PHASE65_RANDOM_BENCHMARK_20261102/PHASE65_FULL_REPRODUCTION_PORTABLE_20261102.zip。只读取包内记录、源码及原始CSV，本轮没有运行GPU推理或中断训练。验算脚本verify_cpu.py，详细结果audit_results.json，文件哈希检查file_hash_checks.json。

## 已核验

- 根SHA256SUMS列出的159项文件全部匹配。这证明包内自洽，不等于第三方签署的运行证明。
- 环境记录为Tesla T4、15360MiB、驱动580.82.07、PyTorch2.8.0+cu128、TensorRT11.4.0.106。实际执行环境来源为包内记录，当前没有连接该服务器复测。
- 两轮独立运行目录各有300条计时、100张图，每图3次。按每图三轮均值再总体平均复算：run_01为75.818736ms，run_portable_retest为76.100758ms。后者P50为79.276147ms，P95为85.057037ms。
- cost.py实际调用predict.load_models和predict.predict_image，读图、预处理、五模型前向、融合、单图CSV写出/flush纳入墙钟计时，计时前后有CUDA同步。模型初始化及预热不计时。
- 此版本不是原始纯PyTorch FP16入口：B01/B03/N01_EMA使用PyTorch FP16 CUDA Graph，B02/M02使用TensorRT FP32；融合公式保持原E08公式。predict.py的PRECISION字符串未完整反映混合精度，正式材料应修正声明。
- 实际加载3份checkpoint和2份engine共282390415字节。包内所有checkpoint与engine合计422232766字节，即便全部保留也低于500000000字节；正式清单还需逐项列入必要配置。压缩包包含样本图片，其总大小不能当作推理模型大小。
- 参考Submission SHA256为9a9846947a544e759cc19c79f6cc8c7fae39508058ee0e3d98d3deb24e8b2a07，与本地主项目E08_EMA_20261020/submission.csv完全相同。
- 100图对应64个Study，按包内索引为完整Study；独立从单图概率重算Study Mean与保存结果完全一致。相对原版Submission最大绝对差0.000763008992，平均差0.000082848443。三个轮次均无0.5阈值二值判定变化、无类别排名变化。

## 不能直接宣布官方通过的原因

1. benchmark/cost.py第542行起明确把Study Mean及最终Study CSV写出放在单图计时之外。现有76.10ms是包含单图写出的参考值；若最终提交接口输出Study结果，应把必要聚合和最终写出计入，或取得官方对单图接口口径的确认。不能假设这部分耗时为零。
2. sample_manifest.json声明RANDOM_TEST_BENCHMARK、official_sample_attested=false。是自选100张Test图与5张预热图，不是官方指定样本。两次均低于100ms不替代官方复测。
3. 一致性仅覆盖64个Study，不能外推到全部1414个Test Study或859个Val Study；原始Test元数据与图片来源本轮未独立交叉核验，完整性结论限于包内索引与哈希。
4. 当前读取的是本地保存的official_rules_extracted.txt（第7条要求最终预测写出，第8条实际加载文件限制）。补充说明原件及材料模板未在本轮重新确认；1e-3、二值判定与排序标准按既有要求核查，不能称全部最新材料条款已核验。
5. TensorRT引擎与目标GPU及软件版本相关，包内通过记录不能保证其他T4服务器直接加载。仍需目标环境加载、重现和提交材料完整性验收，包括预训练来源及许可证。

建议：保留此包为E08的T4工程验证证据。下一次在T4仅补全计时范围、混合精度声明、最终入口对应预测一致性及官方指定样本验收，不需要重新训练或改变融合比例。本轮没有修改包或冻结模型。
