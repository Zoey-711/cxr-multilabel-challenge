# B01/B03 paired validation comparison

All 859 validation Studies aligned by Subject_id and Study_id with identical targets.
B01 best epoch 5: Macro-AUC 0.78839, Macro-F1@0.5 0.28250.
B03 best epoch 3: Macro-AUC 0.78944, Macro-F1@0.5 0.23182.
B03 minus B01 Macro-AUC: +0.00104; paired patient-bootstrap 95% CI [-0.01361, +0.01542].
Paired bootstrap valid/attempted: 2000/2000; invalid excluded: 0.

| ID | Class | B01 AUC | B03 AUC | Delta |
|---:|---|---:|---:|---:|
| 1 | Enlarged Cardiomediastinum | 0.73024 | 0.71080 | -0.01944 |
| 2 | Pneumothorax | 0.83516 | 0.82765 | -0.00751 |
| 3 | Consolidation | 0.77991 | 0.76285 | -0.01706 |
| 4 | Pneumonia | 0.72567 | 0.72129 | -0.00439 |
| 5 | Edema | 0.85590 | 0.86139 | +0.00549 |
| 6 | Cardiomegaly | 0.73066 | 0.76349 | +0.03283 |
| 7 | Atelectasis | 0.77686 | 0.76034 | -0.01652 |
| 8 | Lung Opacity | 0.69020 | 0.71607 | +0.02587 |
| 9 | Pleural Effusion | 0.89307 | 0.90354 | +0.01047 |
| 10 | No Finding | 0.86624 | 0.86693 | +0.00069 |
