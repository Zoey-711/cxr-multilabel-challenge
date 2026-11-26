# R01 512px controlled experiment and fixed E06 fusion

All 859 validation Studies and 228 patients align by Subject_id/Study_id. No hidden test labels or fusion-weight search were used.
M02 and R01 differ in input resolution (384→512) and experiment identity only; both use batch 8 and accumulation 2.
E01=(B01+B02+B03)/3; E03=0.5 E01+0.5 M02; E06=0.5 E01+0.5 R01.

| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |
|---|---:|---:|---:|
| M02 | 0.79724 | 0.29457 | 0.42561 |
| R01 | 0.79565 | 0.32507 | 0.41579 |
| E03 | 0.80993 | 0.27581 | 0.43778 |
| E06 | 0.80540 | 0.29450 | 0.42672 |

## Paired patient bootstrap

- R01_minus_M02: delta AUC -0.00159; 95% CI [-0.01467, +0.01196]; 2000/2000 valid patient resamples.
- E06_minus_E03: delta AUC -0.00453; 95% CI [-0.01235, +0.00303]; 2000/2000 valid patient resamples.

## Weak classes

| Class | Positive Studies / Patients | M02 AUC/AP | R01 AUC/AP | E03 AUC | E06 AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 24 / 16 | 0.7255/0.0868 | 0.7140/0.0712 | 0.7414 | 0.7220 |
| 4 | 68 / 37 | 0.7155/0.2317 | 0.7154/0.1904 | 0.7431 | 0.7427 |
| 8 | 214 / 72 | 0.7020/0.4138 | 0.7086/0.4162 | 0.7164 | 0.7193 |

## Single-/multi-image Study strata

These groups have different disease prevalence and patient composition; their AUC difference is descriptive.
| Model | Group | Studies | Patients | Macro-AUC |
|---|---|---:|---:|---:|
| M02 | single_image | 391 | 94 | 0.72362 |
| M02 | multi_image | 468 | 198 | 0.84129 |
| R01 | single_image | 391 | 94 | 0.72240 |
| R01 | multi_image | 468 | 198 | 0.83652 |

## Runtime and local inference

- M02: best/stop epoch 5/11; best/stop AUC 0.79724/0.75036; training 4.21 h; peak training allocated/reserved 1932/1992 MiB; AMP skips 9; local median/p95 end-to-end latency 71.2/83.7 ms.
- R01: best/stop epoch 4/10; best/stop AUC 0.79565/0.75329; training 4.07 h; peak training allocated/reserved 3000/3012 MiB; AMP skips 9; local median/p95 end-to-end latency 81.5/94.3 ms.

Latency includes validation ZIP read, decode, preprocessing, GPU transfer, FP16 model, sigmoid, and CPU probabilities on the local RTX 4060 Laptop. It does not certify the official T4 100 ms limit.
All ten class AUC/AP values, subgroup class supports, full bootstrap details and SHA-256 provenance are in `per_class.csv` and `comparison.json`.

## Next submission candidate

E03 remains the better-supported next manual submission candidate; E06 has no reliable paired validation gain.
No test inference, submission CSV generation, or leaderboard upload was performed.
