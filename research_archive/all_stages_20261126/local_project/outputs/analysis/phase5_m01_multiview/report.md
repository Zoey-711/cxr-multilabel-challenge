# M01 multiview aggregation

Aligned 1422 saved image probabilities to 859 validation Studies and 228 patients.
The reconstructed Mean exactly matches frozen M01 Study probabilities. No model inference or weight change was made.

| Aggregation | Macro-AUC | Delta vs Mean | Paired patient-bootstrap 95% CI |
|---|---:|---:|---:|
| Mean | 0.79265 | +0.00000 | reference |
| Max | 0.78722 | -0.00543 | [-0.00934, -0.00241] |
| HalfMeanHalfMax | 0.79054 | -0.00211 | [-0.00430, -0.00045] |

Bootstrap: 2000/2000 valid patient draws; 0 excluded.

| Subgroup | Studies | Mean AUC | Max AUC | HalfMeanHalfMax AUC |
|---|---:|---:|---:|---:|
| single_image | 391 | 0.71883 | 0.71883 | 0.71883 |
| multi_image | 468 | 0.83800 | 0.83186 | 0.83540 |

| Class | Mean AUC | Max AUC | HalfMeanHalfMax AUC |
|---|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 0.65526 | 0.63004 | 0.64256 |
| 2 Pneumothorax | 0.87881 | 0.87175 | 0.87506 |
| 3 Consolidation | 0.77135 | 0.77163 | 0.77205 |
| 4 Pneumonia | 0.71627 | 0.70921 | 0.71400 |
| 5 Edema | 0.86639 | 0.86255 | 0.86480 |
| 6 Cardiomegaly | 0.76537 | 0.75826 | 0.76369 |
| 7 Atelectasis | 0.78222 | 0.79047 | 0.78733 |
| 8 Lung Opacity | 0.71445 | 0.70973 | 0.71213 |
| 9 Pleural Effusion | 0.90532 | 0.90279 | 0.90501 |
| 10 No Finding | 0.87110 | 0.86575 | 0.86880 |

ViewPosition counts are image-level in `view_distribution.csv`; per-Study combinations are in the JSON report.
