# Frozen-model validation ensemble

All schemes use exactly 859 validation Studies aligned by Subject_id and Study_id, with identical targets.
The saved files contain one row per Study with image_count and ten Study probabilities, but no image identifiers or per-image probabilities. Image-level Mean versus Max cannot be tested from these artifacts; no image probabilities were inferred.
Only one prespecified fusion was evaluated: the unweighted mean of B01, B02, and B03 Study probabilities. No weight search was performed.

| Scheme | Macro-AUC | Delta vs B01 | Paired patient-bootstrap 95% CI |
|---|---:|---:|---:|
| B01 | 0.78839 | +0.00000 | [+0.00000, +0.00000] |
| B02 | 0.78939 | +0.00100 | [-0.01151, +0.01372] |
| B03 | 0.78944 | +0.00104 | [-0.01361, +0.01542] |
| B01_B02_B03_equal | 0.80267 | +0.01428 | [+0.00642, +0.02104] |

Bootstrap: 2000/2000 valid patient resamples; 0 excluded for absent class positives/negatives.

| Class ID | Label | B01 | B02 | B03 | Equal fusion |
|---:|---|---:|---:|---:|---:|
| 1 | Enlarged Cardiomediastinum | 0.73024 | 0.67425 | 0.71080 | 0.72547 |
| 2 | Pneumothorax | 0.83516 | 0.84372 | 0.82765 | 0.86281 |
| 3 | Consolidation | 0.77991 | 0.79511 | 0.76285 | 0.79270 |
| 4 | Pneumonia | 0.72567 | 0.70804 | 0.72129 | 0.74199 |
| 5 | Edema | 0.85590 | 0.86013 | 0.86139 | 0.86628 |
| 6 | Cardiomegaly | 0.73066 | 0.75251 | 0.76349 | 0.75863 |
| 7 | Atelectasis | 0.77686 | 0.78783 | 0.76034 | 0.78496 |
| 8 | Lung Opacity | 0.69020 | 0.69974 | 0.71607 | 0.71271 |
| 9 | Pleural Effusion | 0.89307 | 0.90327 | 0.90354 | 0.90597 |
| 10 | No Finding | 0.86624 | 0.86929 | 0.86693 | 0.87522 |

Full-precision AUCs are in `class_auc.csv`. Pairwise class-wise probability correlations are in `ensemble_analysis.json`.
