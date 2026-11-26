# Phase 13: V03 conservative multi-view residual

V03 uses the frozen M02 768-D per-image features and Mean probability. It computes masked cross-view feature variance, applies dropout 0.1 and one bias-free 768-to-10 projection, then bounds the logit correction to +/-0.25. The projection starts at zero. A one-image Study has exactly zero variance and therefore exactly preserves M02 probabilities.

The head was trained only on official Train multi-image Studies, using BCE plus residual L2 0.01, AdamW lr 0.001 / weight decay 0.001, cosine schedule, seed 42 and fixed 30 epochs. Official validation selected the best head; no validation label updated its parameters. The frozen training feature cache uses deterministic preprocessing, unlike M02's randomized training augmentation.

Best epoch 3; Train multi-image Studies 10006; validation 859 Studies / 228 patients (single/multi 391/468).

| Method | Macro-AUC | Macro-AP | Macro-F1@0.5 | Single AUC | Multi AUC |
|---|---:|---:|---:|---:|---:|
| M02 | 0.79724 | 0.42572 | 0.29457 | 0.72362 | 0.84129 |
| V00 | 0.80144 | 0.42797 | 0.33076 | 0.73079 | 0.84194 |
| V01 | 0.79886 | 0.42945 | 0.33980 | 0.73061 | 0.83610 |
| V02 | 0.78750 | 0.41433 | 0.37321 | 0.70342 | 0.84112 |
| V03 | 0.79793 | 0.42581 | 0.30283 | 0.72362 | 0.84144 |

Single-image Study predictions match frozen M02 exactly for all 391 Studies. On 468 multi-image Studies, mean absolute probability change is 0.008978, maximum 0.062146.

## Paired 228-patient bootstrap: V03 minus comparator

- M02: +0.00068; 95% CI [-0.00068, +0.00195]; 2000/2000 valid draws.
- V00: -0.00351; 95% CI [-0.01199, +0.00399]; 2000/2000 valid draws.
- V01: -0.00093; 95% CI [-0.00930, +0.00797]; 2000/2000 valid draws.
- V02: +0.01043; 95% CI [+0.00265, +0.01771]; 2000/2000 valid draws.

## Ten-class AUC / AP

| Class | M02 AUC | V00 AUC | V01 AUC | V02 AUC | V03 AUC | M02 AP | V00 AP | V01 AP | V02 AP | V03 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 0.72547 | 0.72849 | 0.72365 | 0.69097 | 0.72687 | 0.08698 | 0.11213 | 0.10760 | 0.06840 | 0.08576 |
| 2 Pneumothorax | 0.86100 | 0.85839 | 0.85816 | 0.84379 | 0.86102 | 0.31107 | 0.30710 | 0.32392 | 0.29962 | 0.31168 |
| 3 Consolidation | 0.76194 | 0.77373 | 0.76571 | 0.75052 | 0.76186 | 0.16539 | 0.17134 | 0.16785 | 0.18149 | 0.16689 |
| 4 Pneumonia | 0.71554 | 0.72180 | 0.71624 | 0.71241 | 0.71492 | 0.23175 | 0.21733 | 0.21909 | 0.22045 | 0.23095 |
| 5 Edema | 0.87015 | 0.87273 | 0.87135 | 0.87162 | 0.87025 | 0.47037 | 0.47638 | 0.47695 | 0.44116 | 0.46919 |
| 6 Cardiomegaly | 0.77118 | 0.77828 | 0.77588 | 0.76779 | 0.76954 | 0.46126 | 0.47322 | 0.47321 | 0.46221 | 0.45918 |
| 7 Atelectasis | 0.79140 | 0.79443 | 0.79475 | 0.77894 | 0.79421 | 0.52594 | 0.51167 | 0.51621 | 0.51012 | 0.52667 |
| 8 Lung Opacity | 0.70205 | 0.70927 | 0.70642 | 0.70246 | 0.70736 | 0.41348 | 0.41234 | 0.41034 | 0.39525 | 0.41715 |
| 9 Pleural Effusion | 0.90495 | 0.90548 | 0.90409 | 0.89812 | 0.90437 | 0.78873 | 0.78825 | 0.78855 | 0.77720 | 0.78808 |
| 10 No Finding | 0.86876 | 0.87176 | 0.87235 | 0.85836 | 0.86889 | 0.80219 | 0.80993 | 0.81079 | 0.78740 | 0.80253 |

## Focus: classes 1, 4, 8

| Class | Positive Studies | M02 AUC | V03 AUC | M02 AP | V03 AP | Single AUC M02/V03 | Multi AUC M02/V03 |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 24 | 0.72547 | 0.72687 | 0.08698 | 0.08576 | 0.60754/0.60754 | 0.84072/0.84411 |
| 4 | 68 | 0.71554 | 0.71492 | 0.23175 | 0.23095 | 0.64712/0.64712 | 0.78294/0.78098 |
| 8 | 214 | 0.70205 | 0.70736 | 0.41348 | 0.41715 | 0.63764/0.63764 | 0.69929/0.69678 |

## Trend and deployment

At best epoch 3, AUC 0.79793; final epoch 30 AUC 0.79744. Multi-Study Train loss fell from 0.14361 to 0.14166. Head training took 61.4 s and peak head GPU allocation was 24.6 MiB.
Head parameters 7680, checkpoint 32489 bytes. Four E03 checkpoints plus V03 head total 196.80 MB (<500 MB). Local single-image Study median 78.2 ms; two-image Study median 159.3 ms; see `deployment.json` for p95 and scope. These are not official T4 measurements.

## Independent-patient protocol

Frozen split: internal Train 9949 patients / 16667 Studies / 27301 images; Holdout 1106 / 1846 / 3043. No patient crosses. See `outputs/analysis/phase13_internal_protocol/report.md` for support counts and the retrain plan. Frozen M02 has seen Holdout patients and is invalid for that comparison. No internal retraining was started.

V03's +0.00068 point AUC change relative to M02 is small and its patient-bootstrap CI crosses zero. The multi-image gain is minimal despite exact single-image preservation. Do not replace the current E03 candidate on this evidence alone.
