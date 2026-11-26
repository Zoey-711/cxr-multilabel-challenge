# B03 preflight against actual B01 run

No training started during this audit.

| Field | B01 | B03 |
|---|---:|---:|
| `experiment.name` | b01_densenet121_384_local | b03_densenet121_lr3e5 |
| `training.backbone_lr` | 0.0001 | 3e-05 |
| `training.early_stopping_patience` | 8 | 6 |

Only the backbone learning rate and permitted early-stopping patience change, besides the experiment name.
Both runs use microbatch 16, accumulation 1, effective batch 16, and 1,897 scheduler steps per epoch.
Data: `F:\数据集\cxr_first.zip`; 30344 train images, 1422 validation images, 859 validation Studies.
The closed-world label encoding, deterministic preprocessing, model, scheduler, seed, and Study-level AUC protocol match B01.
ImageNet weight SHA-256 `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` matches B01 provenance.
Independent planned output: `E:\计挑赛\cxr-multilabel-challenge\outputs\experiments\b03_densenet121_lr3e5_20261016`; absent at audit.
Start without `--resume`; `train.py` will create a fresh ImageNet-pretrained model and new classifier.
