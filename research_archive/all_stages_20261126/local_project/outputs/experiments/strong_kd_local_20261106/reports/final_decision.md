# 蒸馏受控实验报告

状态：COMPLETE_DEVELOPMENT_ONLY
决策：STOP_TEACHER_REBUILD

学生 OOF：N/A；官方 Val 仅用于探索性开发评估。
raw 固定最后一轮是主要比较，EMA 单列；不挑选最高分种子。

|任务|种子|版本|Val Macro-AUC|
|---|---:|---|---:|
|STUDENT_BASE|42|raw|0.79792666|
|STUDENT_BASE|42|ema|0.80018206|
|SKD_CONTROL|42|raw|0.80562638|
|SKD_CONTROL|42|ema|0.80533904|
|SKD_COMPLEMENT|42|raw|0.80655531|
|SKD_COMPLEMENT|42|ema|0.80591764|

|种子|版本|比较|差值|95% CI|
|---|---|---|---:|---|
|42|raw|SKD_CONTROL − STUDENT_BASE|0.00769972|[-0.00058813, 0.01522282]|
|42|raw|SKD_COMPLEMENT − STUDENT_BASE|0.00862865|[0.00046788, 0.01597594]|
|42|raw|SKD_COMPLEMENT − SKD_CONTROL|0.00092893|[-0.00222297, 0.00399272]|
|42|ema|SKD_CONTROL − STUDENT_BASE|0.00515698|[-0.00284540, 0.01217600]|
|42|ema|SKD_COMPLEMENT − STUDENT_BASE|0.00573559|[-0.00212155, 0.01261265]|
|42|ema|SKD_COMPLEMENT − SKD_CONTROL|0.00057860|[-0.00222488, 0.00335626]|

未完成任务：无

Final raw primary; strong teacher student must beat BOTH BASE and CONTROL: delta >= .002, CI lower > 0, worst class delta >= -.01. All enabled seeds must pass. EMA secondary only.

Repeated official Val, multiple comparisons, and only one or two seeds: exploratory, not independent generalization confirmation. Student OOF=N/A. No automatic teacher rebuild or Submission.

T4_UNVERIFIED；没有自动生成或上传 Submission。
