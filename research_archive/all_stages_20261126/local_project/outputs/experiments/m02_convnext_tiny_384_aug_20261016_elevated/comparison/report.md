# M01 versus M02

All 859 Study IDs and all ten targets match exactly. Scores were independently recomputed from the best checkpoints' saved Study probabilities.
M01 Macro-AUC 0.79265; M02 Macro-AUC 0.79724; M02−M01 +0.00459 (paired patient 95% CI [-0.00793, +0.01772]).
M01 Macro-F1@0.5 0.28660; M02 0.29457.
Bootstrap: 2000/2000 valid draws from 228 patients.

| Class | M01 AUC | M02 AUC | Difference |
|---|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 0.65526 | 0.72547 | +0.07021 |
| 2 Pneumothorax | 0.87881 | 0.86100 | -0.01780 |
| 3 Consolidation | 0.77135 | 0.76194 | -0.00941 |
| 4 Pneumonia | 0.71627 | 0.71554 | -0.00073 |
| 5 Edema | 0.86639 | 0.87015 | +0.00376 |
| 6 Cardiomegaly | 0.76537 | 0.77118 | +0.00582 |
| 7 Atelectasis | 0.78222 | 0.79140 | +0.00918 |
| 8 Lung Opacity | 0.71445 | 0.70205 | -0.01241 |
| 9 Pleural Effusion | 0.90532 | 0.90495 | -0.00037 |
| 10 No Finding | 0.87110 | 0.86876 | -0.00234 |

Focused class 1/4/8 paired patient confidence intervals for M02−M01:
- 1 Enlarged Cardiomediastinum: [-0.03943, +0.21263]
- 4 Pneumonia: [-0.04849, +0.04592]
- 8 Lung Opacity: [-0.04266, +0.01100]

M01 best/stop epoch 3/9; M02 5/11.
Overfitting trend, training time and GPU peaks are in `comparison.json`.
The treatment is the combined augmentation strategy; this comparison cannot attribute a change to one operator.
