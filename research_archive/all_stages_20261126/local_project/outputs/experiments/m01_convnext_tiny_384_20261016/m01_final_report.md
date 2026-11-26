# M01 completed training and validation

Best epoch: 3; stopped after epoch 9.
Best Study-level Macro-AUC: 0.79265; Macro-F1@0.5: 0.28660.
Best-epoch training BCE: 0.27779.
Recorded epoch runtime: 1.16 hours; peak allocated GPU memory: 1932 MiB; AMP skipped optimizer steps: 9.
Best checkpoint size: 111378631 bytes.
Independent validation reloaded best_model.pt, covered all 859 Studies, and matched training AUC.
Training configuration, checkpoint, predictions, and comparison results are hashed in frozen_manifest.json.
The first post-training analysis used inference batch 16 and failed its exact AUC cross-check under FP16. Rerunning independent inference with the training validation batch 8 matched the recorded AUC exactly; the checkpoint and training run were unchanged.

| ID | Class | M01 AUC |
|---:|---|---:|
| 1 | Enlarged Cardiomediastinum | 0.65526 |
| 2 | Pneumothorax | 0.87881 |
| 3 | Consolidation | 0.77135 |
| 4 | Pneumonia | 0.71627 |
| 5 | Edema | 0.86639 |
| 6 | Cardiomegaly | 0.76537 |
| 7 | Atelectasis | 0.78222 |
| 8 | Lung Opacity | 0.71445 |
| 9 | Pleural Effusion | 0.90532 |
| 10 | No Finding | 0.87110 |

See comparison/report.md for the three prespecified complementarity schemes and paired bootstrap intervals.
