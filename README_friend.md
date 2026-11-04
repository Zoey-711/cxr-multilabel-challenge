# Strong KD v2: 3 primary tasks + optional 3 paired seed replicas

See [Strong KD v2 instructions](docs/strong_kd/README.md). Tasks: STUDENT_BASE (BCE only), SKD_CONTROL, SKD_COMPLEMENT; optional complete seed2026 triplet. Independent of Phase42. CPU-tested; real Linux 4090 preflight required.

# Phase42：19项新候选与跨折确认

最新批次见 [Phase42 运行说明](docs/phase42/README.md) 和 [结构/参数/MAC清单](docs/phase42/experiment_catalog.md)。本批复用 Phase41 同条件 BASE，不重复其8项训练。填写 `configs/phase42_server.local.yaml` 后运行 `python scripts/phase42_run.py --config configs/phase42_server.local.yaml`。默认19项Fold2 Pilot；确认阶段需预先授权预算，最多4种机制各补4折。新结果分支为 `results/phase42-pilots`。

# Phase41：Phase39 BASE/FUSION 五折验证

最新交付见 [Phase41 朋友服务器说明](docs/phase41/README.md)。本批只新增 BASE/FUSION 各 Fold2–5 共 8 项，Fold1 从冻结产物按 SHA-256 复用。填写 `configs/phase41_server.local.yaml` 后运行 `python scripts/phase41_run.py --config configs/phase41_server.local.yaml`；入口自动执行远程预检、预算门禁、8 卡独立调度与可恢复训练。结果使用独立的 `phase41_batch` 目录，可选增量上传到私有 `results/phase41-fivefold` 分支。

# Phase35：复用两个Fold1，新增8项Fold2–5训练

赛事规则原件及当前执行核对见[比赛规则](docs/competition/README.md)；已上传的本地历史实验代码与各批次说明见[实验索引](docs/experiment_history.md)。历史快照中的旧规则摘要不能替代 2026-10-23 官方 PDF。

最新交付见[Phase35一键运行说明](docs/phase35/README.md)。入口为`python scripts/phase35_run.py --config configs/phase35_server.local.yaml`；先复制并填写`configs/phase35_server.example.yaml`，明确GPU预算并完成自动远程预检。新输出`phase35_batch`、新结果分支`results/phase35-fivefold`与历史批次分离。

# 朋友服务器运行 Phase32 与 E08-Tune

请先阅读 [两批实验的顺序、预检与资源隔离说明](docs/friend_server_phase32_e08.md)。两套GPU队列应依次执行；共有的主机锁会拒绝同时启动。

# E08组成模型专项调参

新增专项请阅读 [E08调参运行说明](docs/e08_tune/README.md)：复用N17基准，5个新Pilot，最多2项条件补齐五折；独立输出与预算，不影响下面的Phase32或Phase17任务。

# Phase32 新增8项Pilot

本次新增实验请阅读 [Phase32运行说明](docs/phase32/README.md)。使用独立克隆、独立输出及明确GPU预算；原Phase17说明保留如下。

# Phase 17-Batch：朋友服务器运行说明

本分支只交付代码和冻结协议。运行目标为**一台 Linux 服务器，最多 8 张 NVIDIA RTX 4090（每张 24GB）**，共享文件系统；每张卡同时最多运行一个独立模型任务。没有 DDP、`torchrun` 或跨节点通信。代码仓库不包含约 55GB 比赛 ZIP、冻结患者索引、标签、E07 参考概率、ImageNet 权重、私钥或令牌。请通过比赛允许的独立渠道取得这些文件。用户或 Codex 不登录朋友服务器。

## 朋友的三步

1. 克隆用户提供的**私有**仓库 `friend-gpu-batch` 分支，并检出用户确认的固定 commit。将官方 `cxr_first.zip`、冻结协议目录和六种合法 torchvision ImageNet 权重放在授权存储位置。协议目录保留仓库相对路径，例如 `batch_assets/train_labels.csv`、`batch_assets/val_labels.csv`、`batch_assets/e07_official_val.csv`、`outputs/analysis/phase17_patient_folds/*.csv`；脚本逐个核对冻结 SHA-256，绝不重划 Fold。
2. 复制 `configs/friend_server.example.env` 为 `configs/friend_server.local.env`，填写绝对路径、授权 GPU 编号、并发数、预检 GPU 小时 `CXR_PREFLIGHT_GPU_HOURS`、覆盖全部26项的总预算 `CXR_GPU_HOURS`，以及其中为G/H四项预留的子预算 `CXR_GH_GPU_HOURS`。三项预算都须由朋友明确授权；总预算必须大于G/H子预算。本地配置已被 `.gitignore` 排除。默认示例为 `0,1,2,3,4,5,6,7` 与并发 8；CPU、内存、磁盘、GPU 必须实测通过。`CXR_DATA_ROOT` 指向官方 ZIP；`CXR_OUTPUT_ROOT` 是结果父目录；`CXR_BACKUP_ROOT` 指向独立授权备份目录。
3. 在仓库根目录只运行下面一条命令。朋友在配置中填写三项预算，即明确授权预检与最多 26 项正式任务；启动后无需发送预检报告，也无需人工二次批准：

```bash
bash scripts/friend_run_all.sh --config configs/friend_server.local.env
```

入口自动安装所需 Python 包、导入并核对冻结协议，然后执行 REMOTE_PREFLIGHT：检查 8 张 4090、数据与权重哈希、真实数据训练和 Val/Test 短推理、此前跳过的三项 CUDA 测试、六条 G/H 训练路径的真实 Train 单步及 checkpoint 恢复与 Val/Test 结果校验，再以 1→2→4→授权并发数逐级检查 GPU 绑定。G/H 初始短测使用**仅限预检的合成软标签**检查蒸馏代码路径，绝不用于正式训练；最终 G/H 配置及真实 OOF 教师在五折完成后再次独立预检。预检全部通过才自动启动先 22 项、后 G/H 4 项。失败保留 `phase17_batch_preflight_<version>/receipt.json`、逐步日志及 `phase17_batch_launcher/failure.json`，不启动训练。修复后将配置中的 `CXR_PREFLIGHT_VERSION` 改为新版本再运行同一命令，旧证据不会被覆盖。中断后重运行同一命令会核验并复用已通过的凭证、已完成任务与可恢复断点；只会清理同主机已确认死亡的队列锁，绝不接管仍在运行的任务。远程 4090 实机结果目前仍为 `PENDING`。

### G/H 远程短测门禁

G/H 四项在完成依赖检查后，会先用各自最终配置读取真实 Train 图像，执行一个 AMP 优化器步，并在独立的 `*_GH_PREFLIGHT` 目录保存、重新加载包含模型、优化器、调度器、GradScaler 和随机数状态的短测 checkpoint。随后使用该恢复状态对真实 Val 与无标签 Test 各两张图像运行确定性推理，按 Study Mean 写出十类概率 CSV。队列核对 checkpoint、两份概率文件、配置和 SHA-256 后才允许对应正式任务启动。短测文件不作为正式模型或提交候选；任一检查失败即将该 G/H 任务标记 `BLOCKED`。此流程已通过本地 CPU 门禁测试，真实 4090 执行结果须以朋友服务器日志为准。

## 22 项固定任务

| 组 | ID | 次数 | 来源 |
|---|---|---:|---|
| A | `D17_F01`–`D17_F05` | 5 | B01 实际 DenseNet-121 配置 |
| A | `C17_F01`–`C17_F05` | 5 | M02 实际 ConvNeXt-Tiny 配置 |
| A | `S17_F01`–`S17_F05` | 5 | T01 实际 Swin-Tiny 配置 |
| N | `N17_F01`–`N17_F05` | 5 | N01 EfficientNetV2-S，训练中同时维护 EMA |
| P | `CS17_F01`、`NM17_F01` | 2 | ConvNeXt-Small、EfficientNetV2-M 的 Fold 1 Pilot |

三骨干原 15 项配置和冻结患者五折不变。每折只用另外四折训练；官方 Val 仅用于早停和最佳 checkpoint，当前折仅作 OOF；Test 完全无标签，只推理。Pilot 单折 OOF 不冒充完整五折 OOF。EMA 是训练内附加分支，单独保存其最佳权重和 Val/OOF/Test 概率；早停仍监控原始权重 Study Macro-AUC。

## G/H 四项：自动依赖、独立预算与运行时安全锁

原来预留的「Pilot 所选骨干 Fold 2–5」四个槽位已由 `G17_FULL_01`、`G17_FULL_02`、`H17_DISTILL_01`、`H17_DISTILL_02` 替代，总任务上限 26。原22项队列先结束；G/H依赖四骨干共20个五折模型的完整OOF及产物核验，两个Pilot失败不阻断G/H。G组使用完整官方Train重训两个骨干；H组使用同患者从未参加教师训练的折外概率作软标签。两组都以官方Val选择最佳权重；Test只推理。它们**没有独立OOF，必须记录N/A**。

G组选拔在运行前固定：候选D/C/S/N的完整五折Train OOF必须齐全且患者隔离；五份原始推理权重分别≤500,000,000字节、4090单模型研究测速p95均<100ms；合格者按完整OOF Macro-AUC降序取前二，完全相同分数按D、C、S、N顺序。4090研究测速**不能证明**官方目标设备合规。每个全Train任务的Epoch上限是相应骨干五折原始最佳Epoch的中位数（五个值，整数）；其余配置继承对应Fold1冻结训练协议。H组学生固定为DenseNet-121和EfficientNetV2-S，教师固定 `0.25×D17_OOF+0.25×C17_OOF+0.50×S17_OOF`。蒸馏采用温度1.0、真实标签BCE与软标签BCE各占0.5；硬标签保持原图像级训练权重，软标签每张图像权重为该Study图像数的倒数，以免多图Study因重复教师概率放大。每个任务保留原骨干学习率、增强、采样、种子与patience，并重新从合法ImageNet权重初始化。配置、教师文件与选择依据在 `manifests/extension_configs/` 和 `extension_selection.json` 冻结。

**默认计划为22+4项。** 必需的20个五折模型完成且OOF来源、患者隔离核验成功后，G/H四项自动取得从OOF冻结的运行配置并逐项做真实4090短步AMP预检。`CXR_GPU_HOURS` 为整个流程上限，基础队列预留 `CXR_GH_GPU_HOURS` 给扩展阶段；缺少朋友授权预算、关键依赖或GPU预检失败时，相关任务为 `BLOCKED`，已完成结果保留。G/H的本地CPU验收不等于4090实机验收。官方Val反复用于选择最佳Epoch，其结果不是新的独立患者泛化证据。

G/H 沿用独立 `experiments/<ID>/`、`validation/`、`test_predictions/`、`submissions/`、`analysis/`、`deployment/` 等直接目录；不压缩打包。每项成功后立即保存最佳推理权重及SHA、完整配置与训练历史、Val/Test十类Study概率、指标、日志、显存与延迟、恢复checkpoint、manifest和DONE。预先固定的`G17_PAIR`及`H17_PAIR`仅用两模型各0.5作研究性精简融合。完整OOF字段为`N/A`，不能生成隐藏Test AUC。Submission通过格式校验才可标`FORMAT_PASS`；真正部署还须全部权重≤500MB、目标设备完整端到端≤100ms。4090测速不能作为官方T4证据，缺失目标硬件实测记`LATENCY_UNVERIFIED`，不能宣称可提交合规。

## 自动化与状态

队列动态分配空闲卡，子进程各自设置 `CUDA_VISIBLE_DEVICES`，每卡最多一个任务。配置 `CXR_MAX_CONCURRENT` 控制并发；启动前核对 GPU 型号/显存、CPU 核数、可用内存、输出空间和 ZIP 随机读取速度，资源不足则明确报错，**不静默改训练 batch 或模型参数**。预检还检查 Python3.12、PyTorch2.10.0、torchvision0.25.0、CUDA12.8、固定分割、AMP、真实前反向与断点恢复。失败任务保留 `FAILED.json`、错误日志和最近可恢复 checkpoint；有限重试只用于判定为可恢复的 I/O 类错误。OOM、非有限梯度、哈希错误不会自动改配置重跑。

每个成功模型立即在 `outputs/phase17_batch/experiments/<ID>/` 保存纯 `state_dict` 的 `best_model.pth`、SHA、实际配置、逐 Epoch 历史、日志、Val/OOF/Test Study 概率、逐类指标、环境、模型大小、推理研究测速、部署状态、恢复 checkpoint 和 `DONE.json`；全Train G/H的OOF为`N/A`。任务状态写入 `reports/experiment_status.csv` 与 `reports/extension_status.csv`。Test 缺失或其推理失败时，已完成的训练和Val/适用OOF保留，状态为 `test_pending`/`test_failed`；补齐 Test 后重复同一启动命令会走恢复路径，**不会重训已完成模型**。

完成后 `reports/experiment_summary.csv` 汇总每个Epoch历史对应的最佳Epoch、每个模型（含EMA）的官方Val Macro-AUC/Macro-AP/Macro-F1、各Fold OOF指标、完整五折OOF指标及可用的固定融合Val指标；类别1–10的AUC/AP和支持数同时写入 `analysis/per_class_metrics.csv`，并在总表提供逐类AUC/AP列。G/H全Train模型的OOF字段为`N/A`。Test只保存无标签概率与submission格式候选；`test_auc`固定为`N/A: hidden Test labels unavailable`，不会计算或伪造隐藏Test AUC。

五折齐全时按 ID 而非 CSV 行号合并完整 OOF、官方 Val 和无标签 Test 概率。固定 E17 公式为 `0.25×D5+0.25×C5+0.5×S5`，不搜索权重。Test 合格时自动写四列、每 Study 全 10 类的候选 CSV 和 manifest，**不会自动上传比赛平台**。若一个 Fold 缺失，对应五折 Test/OOF 标记 `INCOMPLETE`，不生成假完整结果。

查看和安全停止：

```bash
source configs/friend_server.local.env
source .venv/bin/activate
python scripts/batch_run.py --config configs/batch_plan.yaml --action status --root "$CXR_OUTPUT_ROOT/phase17_batch"
python scripts/batch_run.py --config configs/batch_plan.yaml --action stop --root "$CXR_OUTPUT_ROOT/phase17_batch"
```

`stop` 只阻止新任务，正在运行的任务按各自超时/完成逻辑结束。**确认原队列进程已退出且 `queue.lock` 不存在**后，执行下面的恢复命令；如果锁仍在，脚本拒绝清除 STOP，不能强行解锁活跃队列。服务器若使用 Slurm 等调度器，请先让管理员确认实际分配的 GPU ID 和 CPU/RAM；不要同时运行两个队列共享一个输出目录。

```bash
source configs/friend_server.local.env
source .venv/bin/activate
python scripts/batch_run.py --config configs/batch_plan.yaml --action clear-stop --root "$CXR_OUTPUT_ROOT/phase17_batch"
bash scripts/friend_run_all.sh --config configs/friend_server.local.env
```

## 结果直接交付

结果一直保存在 `outputs/phase17_batch/{manifests,experiments,oof,validation,test_predictions,analysis,deployment,submissions,reports}`。没有 ZIP/TAR/GZIP 打包；不自动删除服务器文件。`manifests/delivery_manifest.json`、各模型 `batch_frozen_manifest.json` 记录 SHA-256。完成后把整个 `phase17_batch/` 文件夹通过双方同意的大文件传输渠道交给用户，或提供授权下载位置。仓库不接受胸片数据、checkpoint 或提交 CSV。Windows 导入与核验见 [README_results.md](README_results.md)。

## 部署硬约束

最终全部推理权重压缩前总大小 ≤500,000,000 字节；官方单图端到端延迟 **≤100ms**，80ms 是内部安全目标，并非额外的合规门槛。4090 的研究测速不能证明目标评测硬件（可能是 T4）合规；官方硬件、精度、batch、起止范围和统计量尚缺完整原文。因此未经官方同口径实测均为 `LATENCY_UNVERIFIED`，超过大小为 `SIZE_FAIL`，已证实延迟不合规为 `LATENCY_FAIL`，只有准确率、大小和官方时延都核实才为 `PASS`。单模型的 Forward 时间不能代替完整融合测速。FP16、量化和蒸馏后的实际产物都需重新验证 Val AUC、大小和时延。

待官方给出完整目标计时口径并在目标设备运行时，可使用 `python scripts/benchmark_target_latency.py --config configs/batch_plan.yaml --root "$CXR_OUTPUT_ROOT/phase17_batch" --candidate D17_F01 --official-protocol /authorized/official_timing_protocol.json`。脚本仅在设备、精度、批量、计时范围和统计量逐项吻合时输出官方证据；否则仅存研究测速，不标记 `PASS`。该协议文件必须由用户审核并按脚本的规范化 JSON 规则固定 SHA-256，目前尚未具备。

官方 torchvision 权重文件：DenseNet `densenet121-a639ec97.pth`；ConvNeXt-Tiny `convnext_tiny-983f1562.pth`；Swin-Tiny `swin_t-704ceda3.pth`；EfficientNetV2-S `efficientnet_v2_s-dd5fe13b.pth`；Pilot 分别为 `convnext_small-0c510722.pth`（SHA-256 `0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a`）和 `efficientnet_v2_m-dc08266a.pth`（SHA-256 `dc08266a3779e28992d6a42528a32453f6361d0e4e247b5f20f45e13e7add499`）。六者完整 SHA 均在计划中固定；朋友需通过官方来源或单独授权传输取得权重，并由预检逐字节核对。权重文件不提交 Git。
