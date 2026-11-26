# T01 Swin-Tiny controlled experiment and E07 fixed fusion

T01 differs from the frozen M02 runtime configuration only in experiment identity and ImageNet backbone. Both use 384 aspect-preserving black padding, the same conservative CXR augmentation, BCE, image shuffle, AdamW, 8x2 effective batch 16, seed 42, and Study Mean-probability validation.
All 859 validation Studies and 228 patients align by Subject_id/Study_id. No hidden test labels or fusion-weight search were used.
E01=(B01+B02+B03)/3; E03=0.5 E01+0.5 M02; E07=0.5 E03+0.5 T01.

| Scheme | Macro-AUC | Macro-F1@0.5 | Macro-AP |
|---|---:|---:|---:|
| M02 | 0.79724 | 0.29457 | 0.42561 |
| T01 | 0.79939 | 0.27156 | 0.41698 |
| E03 | 0.80993 | 0.27581 | 0.43778 |
| E07 | 0.81152 | 0.28039 | 0.43723 |

## Paired 228-patient bootstrap

- T01_minus_M02: delta AUC +0.00214; 95% CI [-0.01290, +0.01926]; 2000/2000 valid draws.
- E07_minus_E03: delta AUC +0.00159; 95% CI [-0.00441, +0.00837]; 2000/2000 valid draws.

## Ten-class AUC / AP

| Class | Positive Studies | M02 AUC | T01 AUC | E03 AUC | E07 AUC | M02 AP | T01 AP | E03 AP | E07 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 24 | 0.72547 | 0.72535 | 0.74142 | 0.74057 | 0.08682 | 0.07038 | 0.08776 | 0.09350 |
| 2 Pneumothorax | 37 | 0.86100 | 0.82931 | 0.87973 | 0.86089 | 0.31107 | 0.29471 | 0.32422 | 0.32609 |
| 3 Consolidation | 46 | 0.76194 | 0.76874 | 0.78357 | 0.78298 | 0.16539 | 0.17177 | 0.18233 | 0.17904 |
| 4 Pneumonia | 68 | 0.71554 | 0.72283 | 0.74313 | 0.74842 | 0.23171 | 0.20665 | 0.26565 | 0.24809 |
| 5 Edema | 115 | 0.87015 | 0.87755 | 0.87520 | 0.87973 | 0.47064 | 0.45776 | 0.46473 | 0.46938 |
| 6 Cardiomegaly | 177 | 0.77118 | 0.77944 | 0.77338 | 0.78411 | 0.46124 | 0.44506 | 0.45737 | 0.46537 |
| 7 Atelectasis | 193 | 0.79140 | 0.79585 | 0.79759 | 0.80206 | 0.52592 | 0.50420 | 0.53459 | 0.52961 |
| 8 Lung Opacity | 214 | 0.70205 | 0.71697 | 0.71642 | 0.72462 | 0.41375 | 0.42494 | 0.43365 | 0.43272 |
| 9 Pleural Effusion | 260 | 0.90495 | 0.89891 | 0.91007 | 0.90907 | 0.78866 | 0.76005 | 0.79869 | 0.79028 |
| 10 No Finding | 306 | 0.86876 | 0.87891 | 0.87881 | 0.88274 | 0.80094 | 0.83429 | 0.82881 | 0.83824 |

Focused class support: class 1: 24 positive Studies / 16 positive patients, class 4: 68 positive Studies / 37 positive patients, class 8: 214 positive Studies / 72 positive patients.

## Single/multi-image Studies

Subgroup class composition differs; subgroup AUCs are descriptive.
| Scheme | Group | Studies | Patients | Macro-AUC |
|---|---|---:|---:|---:|
| M02 | single_image | 391 | 94 | 0.72362 |
| M02 | multi_image | 468 | 198 | 0.84129 |
| T01 | single_image | 391 | 94 | 0.72571 |
| T01 | multi_image | 468 | 198 | 0.84126 |
| E03 | single_image | 391 | 94 | 0.74113 |
| E03 | multi_image | 468 | 198 | 0.84961 |
| E07 | single_image | 391 | 94 | 0.74158 |
| E07 | multi_image | 468 | 198 | 0.85239 |

## Runtime, memory and deployment

- M02: best/stop epoch 5/11; best/stop AUC 0.79724/0.75036; training 4.21 h; peak allocated/reserved GPU 1932/1992 MiB; AMP skips 9; local median/p95 end-to-end single-image latency 69.4/78.9 ms.
- T01: best/stop epoch 7/13; best/stop AUC 0.79939/0.79744; training 4.89 h; peak allocated/reserved GPU 2220/2268 MiB; AMP skips 8; local median/p95 end-to-end single-image latency 85.3/95.9 ms.

T01 checkpoint 110.41 MB. All five unique E07 inference checkpoints total 307.18 MB, below 500 MB.
Latency is a local RTX 4060 Laptop measurement that includes validation ZIP read, decode, preprocessing, transfer, FP16 model and CPU probabilities. The official T4 100 ms figure remains a reference requiring separate measurement; it is not treated here as an automatic exclusion rule.

## Next manual submission candidate

E03 remains the better-supported next manual submission candidate; E07 lacks a reliable paired validation gain.
No hidden-test inference, submission CSV generation or leaderboard upload was performed.
