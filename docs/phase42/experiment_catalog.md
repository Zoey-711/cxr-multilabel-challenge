# Phase42 冻结候选与成本审计

原始24项去重后保留19项：17个结构、2个训练策略。原Phase42八项只做过读取审计，没有开发提交或训练记录；本扩展版是唯一正式Phase42计划。

所有数值来自本地CPU在384输入上的模型图统计，不是4090测速。MAC只计Conv/Linear及注意力矩阵乘法，未计归一化、激活、GeM幂运算、插值、读图等；因此不是完整FLOPs，也不能直接代表延迟。

|ID|结构/唯一变化|新增参数|MAC（G）|相对BASE|预计峰值显存|
|---|---|---:|---:|---:|---|
|A01|Gated Multi-Stage Fusion：Stage5 GAP ->64; sigmoid gate conditioned on final+mid; concat final1280+64|97,024|8.364447|1.000012|待远程实测，预算上限20GiB|
|A02|FPN-Lite：Stage4 and final maps ->64; bilinear top-down sum; residual depthwise3; GAP|78,720|8.381185|1.002013|待远程实测，预算上限20GiB|
|A04|Cross-Scale Attention Lite：Stage4/final ->64; fixed4x4 pooled tokens; 4-head cross attention; concatenate final GAP|107,520|8.381161|1.002010|待远程实测，预算上限20GiB|
|B01|GeM：Replace final GAP with FP32 GeM; p initialized3, learned clamp[1,6], epsilon1e-6|1|8.364350|1.000000|待远程实测，预算上限20GiB|
|B02|GAP+GMP：Concatenate final average and maximum pooled vectors|12,800|8.364363|1.000002|待远程实测，预算上限20GiB|
|B03|Spatial Pyramid Pooling：Concatenate final adaptive-average 1x1 and2x2 bins|51,200|8.364401|1.000006|待远程实测，预算上限20GiB|
|B04|Shared Attention Pooling：One shared spatial softmax map for all classes; original linear classifier|1,281|8.364719|1.000044|待远程实测，预算上限20GiB|
|C01|Class-Specific Attention：Ten spatial softmax maps; class-specific weighted feature pooling|12,810|8.368024|1.000439|待远程实测，预算上限20GiB|
|C02|Label Query Decoder：Final map ->64; ten learned queries, one four-head cross-attention layer, no FFN|86,529|8.377580|1.001582|待远程实测，预算上限20GiB|
|C03|Class-Aware GeM：Ten softplus evidence maps, ten independent learned p initialized3 clamp[1,6], affine logits|30|8.366180|1.000219|待远程实测，预算上限20GiB|
|C04|Low-Rank Label Interaction：Original logits plus rank4 tanh residual label interaction|94|8.364350|1.000000|待远程实测，预算上限20GiB|
|D01|Large-Kernel Depthwise Head：Residual depthwise7x7 SiLU before original GAP|64,000|8.373382|1.001080|待远程实测，预算上限20GiB|
|D02|SE Classification Head：Final GAP channel gate with reduction16 (1280->80->1280)|206,160|8.364555|1.000024|待远程实测，预算上限20GiB|
|D03|ECA Classification Head：Final GAP channel gate with fixed one-dimensional kernel5|5|8.364356|1.000001|待远程实测，预算上限20GiB|
|D04|Dilated Spatial Head：Residual depthwise3x3 dilation2 SiLU before GAP|12,800|8.366009|1.000198|待远程实测，预算上限20GiB|
|E03|BCE Label Smoothing：Only target changes to0.95*y+0.025; mask unchanged|0|8.364350|1.000000|待远程实测，预算上限20GiB|
|E04|Stochastic Depth：Only existing stochastic-depth probabilities scaled1.5 (maximum schedule .2->.3)|0|8.364350|1.000000|待远程实测，预算上限20GiB|
|F01|Coordinate Attention：Row/column pooled channel gates, shared reduction to40, no added BatchNorm|156,200|8.366808|1.000294|待远程实测，预算上限20GiB|
|F04|Low-Rank Spatial Mixing：Residual nonlinear token mixing 144->8->144 per final channel; fixed12x12 grid|2,304|8.367299|1.000353|待远程实测，预算上限20GiB|

BASE：20,190,298参数，8.364350G MAC。训练常驻参数/梯度/Adam双动量/EMA张量下界约0.404GB；激活、workspace及缓冲另计，不能以此估计峰值显存。

## 去重与历史对照

- A03：标量加权融合与A01门控/Phase39已有融合重叠，取消。
- F03：未定义独立于GAP+GMP/SPP的机制，取消。
- F02：CBAM同时包含SE通道门控与空间注意力，与D02/B04机制重叠；先测单因素。
- E01/E02：两次前向会改变BatchNorm更新次数与随机数/计算预算，需要额外匹配对照；本批暂缓而非把混杂实验写成单变量。
- B04为所有类别共享一个注意力图；C01为10个独立类别图，二者不等价。D02仅通道门控，F01仅坐标分解空间门控。

已读来源：`configs/phase32/baseline_N17_F01.yaml`、Phase32全部Pilot配置、`configs/e08_tune/`、Phase35协议、Phase39模型/训练/配置、Phase41计划及训练入口、`research_archive/local_20261026`历史代码。原有SE已在EfficientNet内部存在；D02的唯一新因素是最终分类前额外SE门控，不声称发明新的SE机制。

N17微批8×累积2；Phase39/41微批4×累积4。两者BatchNorm不同，不能把N17当严格BASE。本批锁定Phase41同折BASE，沿用4×4、worker0、seed42、30epoch、patience6、EMA .999与相同ImageNet初始化、BCE、AdamW及增强。Phase41现有8项不计入本批。

探索折固定Fold2；它是本轮未参与结构筛选的Fold，但历史Phase17/35等已经研究同一队列，不能宣称完全全新独立验证。Fold2用于checkpoint早停也会带来选择偏差。

Phase41历史manifest只有不透明identity哈希，没有内嵌完整Git来源，故原代码commit采用操作者声明，并交叉核验预检绑定、实际配置/数据划分及全部产物哈希；报告保留此来源限制。缺失可比BASE直接BLOCKED，不能拿N17替代或自动重训Phase41。
