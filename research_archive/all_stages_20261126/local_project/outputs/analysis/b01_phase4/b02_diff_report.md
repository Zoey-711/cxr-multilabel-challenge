# B01/B02 configuration and launch preflight

B02 has not been started. B01 files were only read.

## Differences

| Field | B01 | B02 |
|---|---:|---:|
| `experiment.name` | b01_densenet121_384_local | b02_densenet121_wd1e3 |
| `training.early_stopping_patience` | 8 | 6 |
| `training.weight_decay` | 0.0001 | 0.001 |

Data archive: `F:\数据集\cxr_first.zip`; train images: 30344; official validation images: 1422, studies: 859.
Unlisted classes use the audited closed-world negative policy; validation fusion and ten-class Macro-AUC protocol match B01.
ImageNet weight SHA-256: `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`; matches B01 provenance.
Planned independent checkpoint directory: `E:\计挑赛\cxr-multilabel-challenge\outputs\experiments\b02_densenet121_wd1e3_20261016_elevated` (absent at preflight).
B02 starts from the same ImageNet weight file with a new ten-class head. Do not pass `--resume` or a B01 checkpoint.
Microbatch / accumulation / effective batch: B01 16 / 1 / 16; B02 16 / 1 / 16. Scheduler steps per epoch: 1897 in both runs.
Correction: B02 was previously planned as 4 x 4; its microbatch is now 16 and the accumulation field was removed to match B01 exactly.
Early stopping patience changes from 8 to 6 as authorized.
Estimated time: 7-9 minutes per epoch, roughly 1.3-4.5 hours for 11-30 epochs; no B02 timing has been measured.
