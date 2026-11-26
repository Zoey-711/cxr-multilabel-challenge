# A02 fixed validation fusions

All 859 Studies and 228 patients align by Subject_id and Study_id; all ten labels match.
No standalone E01 validation probability file was frozen. E01 was reconstructed from the three SHA-verified frozen DenseNet Study probability files and reproduced its recorded Macro-AUC exactly.

| Scheme | Macro-AUC | Macro-F1@0.5 | Delta vs E01 | Paired patient 95% CI |
|---|---:|---:|---:|---:|
| E01 | 0.80267 | 0.27970 | +0.00000 | [0, 0] |
| E01_M01_equal | 0.80344 | 0.28236 | +0.00077 | [-0.00792, +0.00678] |
| E01_M02_equal | 0.80993 | 0.27581 | +0.00726 | [+0.00023, +0.01299] |

Detailed ten-class AUCs are in `per_class_auc.csv` and `a02_results.json`.
These are validation-only fixed fusions; no test-ranking-based selection or weight search was performed.
