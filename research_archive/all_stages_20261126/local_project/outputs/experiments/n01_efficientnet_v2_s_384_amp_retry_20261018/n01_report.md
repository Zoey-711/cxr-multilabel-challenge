# N01 EfficientNetV2-S: independent validation and fixed fusion

All scores below use the same 859 official validation Studies and 228 patients. The Phase 15 internal Holdout was not read or used.

| Scheme | Study Macro-AUC | Macro-AP | Macro-F1@0.5 |
|---|---:|---:|---:|
| N01 | 0.79258 | 0.41481 | 0.31591 |
| N01_EMA | 0.80491 | 0.42334 | 0.34880 |
| E03 | 0.80993 | 0.43778 | 0.27581 |
| E07 | 0.81152 | 0.43723 | 0.28039 |
| E08 | 0.80804 | 0.43085 | 0.29827 |
| E08_EMA | 0.81167 | 0.43800 | 0.32844 |

## Paired patient bootstrap

2000 valid draws; 0 invalid draws excluded. The same sampled patients and all their Studies were used for each paired difference.

- N01_EMA_minus_N01: +0.01232; 95% CI [+0.00372, +0.02270].
- E08_minus_E03: -0.00190; 95% CI [-0.00826, +0.00471].
- E08_minus_E07: -0.00348; 95% CI [-0.01210, +0.00457].
- E08_EMA_minus_E03: +0.00174; 95% CI [-0.00364, +0.00795].
- E08_EMA_minus_E07: +0.00015; 95% CI [-0.00789, +0.00782].

Raw and EMA variants share a training run and validation set. Their multiple comparisons are correlated, not independent significance tests.

## Training and deployment

Raw best epoch 3; EMA best epoch 5; stop epoch 9 (patience_exhausted). EMA successful-step updates 17066; total AMP skips 7.
Training epoch time 3.64 GPU hours; peak PyTorch allocated memory 2222 MiB.
E08 five inference checkpoints total 278,432,862 bytes (< 500,000,000 bytes).
Local RTX 4060 Laptop N01 single-image mean latency 80.8 ms; five-model E08 mean latency 156.7 ms. Neither is an official T4 measurement.

Ten-class AUC/AP and focused classes 1, 4 and 8 are in `comparison/per_class.csv`; error residual correlations are in `comparison/comparison.json`.

E03's leaderboard display 76.45 is user-reported; the exact uploaded file remains unverified locally. No hidden test inference or leaderboard upload was performed.
