# L01 ASL controlled experiment and fixed E05 fusion

All 859 validation Studies and 228 patients align on Subject_id/Study_id; no hidden test labels or weight search were used.
E01=(B01+B02+B03)/3; E03=0.5 E01 + 0.5 M02; E05=0.5 E01 + 0.5 L01.

| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |
|---|---:|---:|---:|
| M02 | 0.79724 | 0.29457 | 0.42561 |
| L01 | 0.79854 | 0.40451 | 0.42010 |
| E03 | 0.80993 | 0.27581 | 0.43778 |
| E05 | 0.80772 | 0.36240 | 0.42881 |

## Paired patient bootstrap

- L01_minus_M02: delta Macro-AUC +0.00130; 95% CI [-0.00929, +0.01266]; 2000/2000 valid patient resamples.
- E05_minus_E03: delta Macro-AUC -0.00221; 95% CI [-0.00880, +0.00521]; 2000/2000 valid patient resamples.

## Weak classes

| Class ID | Positive Studies | M02 AUC/AP | L01 AUC/AP | E03 AUC | E05 AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 24 (16 patients) | 0.7255/0.0868 | 0.7340/0.0868 | 0.7414 | 0.7416 |
| 4 | 68 (37 patients) | 0.7155/0.2317 | 0.7258/0.2095 | 0.7431 | 0.7468 |
| 8 | 214 (72 patients) | 0.7020/0.4138 | 0.7143/0.4219 | 0.7164 | 0.7217 |

## Training trend

- M02: best/stop epoch 5/11; AUC at best/stop 0.79724/0.75036; drop 0.04688; elapsed 4.21 h; peak GPU allocated 1932 MiB; AMP skips 9.
- L01: best/stop epoch 5/11; AUC at best/stop 0.79854/0.75756; drop 0.04098; elapsed 4.09 h; peak GPU allocated 1932 MiB; AMP skips 7.

BCE and ASL training losses have different scales; compare validation AUC trends rather than their absolute loss values.
All 10 per-class AUC/AP values and bootstrap details are in `per_class.csv` and `comparison.json`.

## Next submission candidate

E03 remains the better-supported next manual submission candidate; E05 has no clear paired-bootstrap improvement over E03.
No submission file was generated or uploaded.
