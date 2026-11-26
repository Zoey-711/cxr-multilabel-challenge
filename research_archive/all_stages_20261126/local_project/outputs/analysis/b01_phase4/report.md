# B01 frozen validation analysis

Source: existing validation `study_predictions.csv` and `metrics.jsonl`; no model inference or training.
Study count: 859; patient count: 228.
Best epoch: 5; Study-level Macro-AUC: 0.78839; patient-bootstrap 95% CI: [0.75144, 0.82058].
Bootstrap: 2000/2000 valid replicates; 0 excluded for a class missing positives or negatives.
All studies of a sampled patient stay together, including repeat studies; duplicates from resampling are retained.

## Threshold sensitivity

| Threshold | Macro-F1 | Macro-Precision | Macro-Recall | Zero-prediction Studies |
|---:|---:|---:|---:|---:|
| 0.2 | 0.39759 | 0.34168 | 0.56907 | 0/859 |
| 0.3 | 0.37162 | 0.36073 | 0.44265 | 8/859 |
| 0.4 | 0.33351 | 0.35036 | 0.32511 | 96/859 |
| 0.5 | 0.28250 | 0.39148 | 0.23552 | 230/859 |

Precision and Recall are unweighted means of ten per-class values; undefined precision from zero predictions is set to 0.
Each class's positive/negative count, threshold metrics, and probability p0/p10/p25/p50/p75/p90/p100 are in `threshold_by_class.csv`.

## Epoch trajectory

AUC peaks at epoch 5 (0.78839); the first lower AUC is epoch 6 (0.78716).
By epoch 7 AUC falls to 0.77417 while training loss falls from 0.25319 to 0.18334; this is the first clear sustained overfitting signal.
At epoch 13: AUC 0.74814, loss 0.03770.
See `epoch_curves.svg` and `epoch_metrics.csv` for every epoch.

Threshold results describe the validation set only. The competition's primary AUC does not depend on the threshold.
