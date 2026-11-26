# Phase 12: frozen M02 multi-view Study heads

M02 checkpoint was SHA-256 verified before feature extraction. Only official Train features and labels updated the heads; validation was used for best checkpoint selection. Cached Train features use deterministic validation preprocessing, unlike M02's random training augmentation.

Train 18513 Studies (single/multi-image 8507/10006); validation 859 Studies / 228 patients (single/multi-image 391/468).

## Architecture control

All three heads take the same frozen 768-D per-image M02 features and ten-class Train Study labels. They use BCE, AdamW, lr 0.001, weight decay 0.0001, Study batch 128, cosine scheduler, seed 42, and exactly 30 epochs. Best validation Study Macro-AUC selects each checkpoint.

- V00: masked mean of image features, then a 768-to-10 linear head.
- V01: tanh/sigmoid gated attention with masked softmax, then the same linear head.
- V02: one permutation-equivariant Transformer encoder layer (128-D, four heads, no positional embeddings), masked mean, 768-D projection and the same linear head. Padding is masked from attention.

| Method | Best epoch | Macro-AUC | Macro-AP | Macro-F1@0.5 | Head parameters |
|---|---:|---:|---:|---:|---:|
| M02 Mean probability | — | 0.79724 | 0.42572 | 0.29457 | 0 |
| V00 | 1 | 0.80144 | 0.42797 | 0.33076 | 7690 |
| V01 | 1 | 0.79886 | 0.42945 | 0.33980 | 204683 |
| V02 | 1 | 0.78750 | 0.41433 | 0.37321 | 337674 |

## Paired patient bootstrap AUC differences

- v00_minus_M02: +0.00419, 95% CI [-0.00302, +0.01209], 2000/2000 valid draws.
- v01_minus_M02: +0.00162, 95% CI [-0.00731, +0.00933], 2000/2000 valid draws.
- v02_minus_M02: -0.00975, 95% CI [-0.01706, -0.00194], 2000/2000 valid draws.
- v01_minus_v00: -0.00258, 95% CI [-0.00519, -0.00083], 2000/2000 valid draws.
- v02_minus_v00: -0.01394, 95% CI [-0.02328, -0.00402], 2000/2000 valid draws.

## Per-class AUC / AP

| Class | M02 AUC | V00 AUC | V01 AUC | V02 AUC | M02 AP | V00 AP | V01 AP | V02 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 0.72547 | 0.72849 | 0.72365 | 0.69097 | 0.08698 | 0.11213 | 0.10760 | 0.06840 |
| 2 Pneumothorax | 0.86100 | 0.85839 | 0.85816 | 0.84379 | 0.31107 | 0.30710 | 0.32392 | 0.29962 |
| 3 Consolidation | 0.76194 | 0.77373 | 0.76571 | 0.75052 | 0.16539 | 0.17134 | 0.16785 | 0.18149 |
| 4 Pneumonia | 0.71554 | 0.72180 | 0.71624 | 0.71241 | 0.23175 | 0.21733 | 0.21909 | 0.22045 |
| 5 Edema | 0.87015 | 0.87273 | 0.87135 | 0.87162 | 0.47037 | 0.47638 | 0.47695 | 0.44116 |
| 6 Cardiomegaly | 0.77118 | 0.77828 | 0.77588 | 0.76779 | 0.46126 | 0.47322 | 0.47321 | 0.46221 |
| 7 Atelectasis | 0.79140 | 0.79443 | 0.79475 | 0.77894 | 0.52594 | 0.51167 | 0.51621 | 0.51012 |
| 8 Lung Opacity | 0.70205 | 0.70927 | 0.70642 | 0.70246 | 0.41348 | 0.41234 | 0.41034 | 0.39525 |
| 9 Pleural Effusion | 0.90495 | 0.90548 | 0.90409 | 0.89812 | 0.78873 | 0.78825 | 0.78855 | 0.77720 |
| 10 No Finding | 0.86876 | 0.87176 | 0.87235 | 0.85836 | 0.80219 | 0.80993 | 0.81079 | 0.78740 |

## Single/multi-image subgroups

Subgroup class composition differs; subgroup AUC comparisons are descriptive.

- M02 Mean probability: single-image Macro-AUC 0.72362 (10/10 evaluable classes); multi-image 0.84129 (10/10).
- V00: single-image Macro-AUC 0.73079 (10/10 evaluable classes); multi-image 0.84194 (10/10).
- V01: single-image Macro-AUC 0.73061 (10/10 evaluable classes); multi-image 0.83610 (10/10).
- V02: single-image Macro-AUC 0.70342 (10/10 evaluable classes); multi-image 0.84112 (10/10).

Focused classes 1, 4, 8 by view count (positive/negative Study support; AUC):

| Class | Group | Positive | Negative | M02 | V00 | V01 | V02 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | single | 15 | 376 | 0.60754 | 0.63652 | 0.63599 | 0.54628 |
| 1 | multi | 9 | 459 | 0.84072 | 0.81796 | 0.80973 | 0.84653 |
| 4 | single | 37 | 354 | 0.64712 | 0.64705 | 0.64605 | 0.64078 |
| 4 | multi | 31 | 437 | 0.78294 | 0.79065 | 0.77574 | 0.79110 |
| 8 | single | 137 | 254 | 0.63764 | 0.64972 | 0.65018 | 0.63170 |
| 8 | multi | 77 | 391 | 0.69929 | 0.70150 | 0.69595 | 0.70366 |

## Training trend

- V00: best epoch 1, final epoch 30; AUC 0.80144 to 0.78843 while Train BCE 0.23540 to 0.15663; head training 37.4 s, peak GPU allocation 23.0 MiB.
- V01: best epoch 1, final epoch 30; AUC 0.79886 to 0.78386 while Train BCE 0.23464 to 0.14615; head training 45.3 s, peak GPU allocation 26.6 MiB.
- V02: best epoch 1, final epoch 30; AUC 0.78750 to 0.75102 while Train BCE 0.20831 to 0.08778; head training 57.3 s, peak GPU allocation 38.3 MiB.

## Deployment

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU; M02 backbone median 5.28 ms. E03's four unique checkpoints plus the largest V head total 198.13 MB, below 500 MB.
The separate local one-image end-to-end timings in `deployment.json` include ZIP decode and preprocessing; they exclude other E03 models. Official T4 end-to-end 100 ms compliance remains unverified.

## Interpretation and next study

V00 has the highest head AUC, but its patient-bootstrap improvement over M02 crosses zero. V01 provides no clear gain over V00; V02 falls below the non-learning baseline. Every head peaks at epoch 1 while Train BCE keeps falling, consistent with early overfitting of learned aggregation on these frozen features. Class 1 V00 AP improves but has only 24 positive validation Studies, while classes 4 and 8 show mixed AUC/AP changes. No E03+head fusion was tested; E03 remains the frozen submission candidate.

A controlled residual correction could add a small bounded correction to the frozen M02 Study logits, initialized at zero, with the same Train-only supervision and a predeclared strength. Design only; no residual model was trained here. For a stronger generalization check, a true internal patient holdout requires retraining the backbone without those patients; the frozen M02 backbone has already seen all Train patients. No backbone fine-tuning, FPN or disease-specific query decoder was run.
