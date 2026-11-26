# M01 validation complementarity

All schemes use the same 859 Study IDs and identical ten-class targets.
Only the three prespecified M01 schemes were evaluated; no weight search was run.

| Scheme | Macro-AUC | Macro-F1@0.5 | Delta vs B01 (95% CI) | Delta vs frozen equal (95% CI) |
|---|---:|---:|---:|---:|
| M01 | 0.79265 | 0.28660 | +0.00426 [-0.01469, +0.01976] | -0.01002 [-0.02578, +0.00166] |
| M01_B01_equal | 0.80253 | 0.29064 | +0.01414 [+0.00461, +0.02175] | -0.00015 [-0.00719, +0.00507] |
| M01_Frozen_equal_half | 0.80344 | 0.28236 | +0.01505 [+0.00224, +0.02546] | +0.00077 [-0.00792, +0.00678] |

Patient-cluster bootstrap: 2000/2000 valid draws; 0 excluded.
Full ten-class AUCs are in `per_class_auc.csv`.
