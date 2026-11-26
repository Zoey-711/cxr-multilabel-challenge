# S01 versus M02 and fixed E04 versus E03

All 859 Studies, 228 patients and ten targets align by Subject_id/Study_id. E03 = 0.5 E01 + 0.5 M02; E04 = 0.5 E01 + 0.5 S01. No weight search.

| Model | Macro-AUC | Macro-F1@0.5 |
|---|---:|---:|
| M02 | 0.79724 | 0.29457 |
| S01 | 0.79308 | 0.29238 |
| E03 | 0.80993 | 0.27581 |
| E04 | 0.80752 | 0.28681 |

S01_minus_M02 Macro-AUC difference -0.00416; paired patient 95% CI [-0.01493, +0.01016] (2000/2000 valid draws).

E04_minus_E03 Macro-AUC difference -0.00241; paired patient 95% CI [-0.00785, +0.00430] (2000/2000 valid draws).

M02 best/stop Epoch 5/11, AUC drop 0.04688; S01 3/9, drop 0.06076.
Single- and multi-image Study metrics and ten-class composition are in `comparison.json`. The groups differ in disease prevalence, so their raw Macro-AUC gap is descriptive, not a causal effect of view count.
