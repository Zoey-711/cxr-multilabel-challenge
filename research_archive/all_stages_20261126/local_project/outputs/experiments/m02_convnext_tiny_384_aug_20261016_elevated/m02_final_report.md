# M02 completed training and independent validation

Best epoch 5; stop epoch 11 (patience_exhausted); patience 6.
Best Study-level Macro-AUC 0.79724; Macro-F1@0.5 0.29457; 859 Studies.
M02−M01 Macro-AUC +0.00459; paired patient bootstrap 95% CI [-0.00793, +0.01772].
M01 best/stop epoch 3/9; M02 5/11.
M01 AUC decline best to stop 0.03271; M02 0.04688.
Epoch runtime sum 4.21 h; peak GPU allocated/reserved 1932/1992 MiB; board peak 4073 MiB; AMP skipped steps 9.
Independent validation reloaded best_model.pt and reproduced the logged 859-Study AUC. Image-level and Study-level probabilities are saved under validation_analysis/.
M02 changes the combined train augmentation strategy; attribution to a single operator is not supported.

| ID | Class | M01 AUC | M02 AUC | Difference |
|---:|---|---:|---:|---:|
| 1 | Enlarged Cardiomediastinum | 0.65526 | 0.72547 | +0.07021 |
| 2 | Pneumothorax | 0.87881 | 0.86100 | -0.01780 |
| 3 | Consolidation | 0.77135 | 0.76194 | -0.00941 |
| 4 | Pneumonia | 0.71627 | 0.71554 | -0.00073 |
| 5 | Edema | 0.86639 | 0.87015 | +0.00376 |
| 6 | Cardiomegaly | 0.76537 | 0.77118 | +0.00582 |
| 7 | Atelectasis | 0.78222 | 0.79140 | +0.00918 |
| 8 | Lung Opacity | 0.71445 | 0.70205 | -0.01241 |
| 9 | Pleural Effusion | 0.90532 | 0.90495 | -0.00037 |
| 10 | No Finding | 0.87110 | 0.86876 | -0.00234 |
