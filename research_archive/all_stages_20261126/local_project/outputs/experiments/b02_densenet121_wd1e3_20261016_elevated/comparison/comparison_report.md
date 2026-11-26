# B01/B02 paired validation comparison

All 859 validation Studies aligned by Subject_id and Study_id with identical targets.
B01 best epoch 5: Macro-AUC 0.78839, Macro-F1@0.5 0.28250.
B02 best epoch 5: Macro-AUC 0.78939, Macro-F1@0.5 0.29922.
B02 minus B01 Macro-AUC: +0.00100; paired patient-bootstrap 95% CI [-0.01151, +0.01372].
Paired bootstrap valid/attempted: 2000/2000; invalid excluded: 0.

| ID | Class | B01 AUC | B02 AUC | Delta |
|---:|---|---:|---:|---:|
| 1 | Enlarged Cardiomediastinum | 0.73024 | 0.67425 | -0.05599 |
| 2 | Pneumothorax | 0.83516 | 0.84372 | +0.00857 |
| 3 | Consolidation | 0.77991 | 0.79511 | +0.01520 |
| 4 | Pneumonia | 0.72567 | 0.70804 | -0.01763 |
| 5 | Edema | 0.85590 | 0.86013 | +0.00424 |
| 6 | Cardiomegaly | 0.73066 | 0.75251 | +0.02185 |
| 7 | Atelectasis | 0.77686 | 0.78783 | +0.01096 |
| 8 | Lung Opacity | 0.69020 | 0.69974 | +0.00954 |
| 9 | Pleural Effusion | 0.89307 | 0.90327 | +0.01019 |
| 10 | No Finding | 0.86624 | 0.86929 | +0.00305 |
