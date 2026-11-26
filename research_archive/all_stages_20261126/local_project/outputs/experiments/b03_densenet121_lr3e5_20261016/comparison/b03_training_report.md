# B03 backbone learning-rate control: final report

## Frozen setup

B03 was copied from B01's saved runtime configuration. Apart from the experiment name, only the backbone learning rate changed from 0.0001 to 0.00003 and the authorized early-stopping patience changed from 8 to 6. Both used microbatch 16, gradient accumulation 1, effective batch 16, seed 42, DenseNet-121, the same ImageNet weights (SHA-256 `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`), new ten-class heads, 384×384 deterministic preprocessing, BCE, AdamW, the same cosine/warmup scheduler, weight decay 0.0001, and mean-probability Study fusion. No random augmentation was configured in either run. B03 did not resume from B01 or B02.

## Best checkpoint and paired result

- B03 ended normally at epoch 9 by patience-6 early stopping. Best epoch: 3.
- Independent best-checkpoint validation on all 859 Studies and 10 classes: Macro-AUC **0.7894350141**, Macro-F1@0.5 **0.2318212060**; saved training metrics agree.
- B01 best epoch 5: Macro-AUC 0.7883919646, Macro-F1@0.5 0.2825032402.
- B03 minus B01 Macro-AUC: **+0.0010430494**. Paired patient-cluster bootstrap across 228 patients, 2,000/2,000 valid replicates, 95% percentile CI **[-0.0136091520, +0.0154185585]**. The interval includes zero; this run does not establish an AUC improvement.
- B03 minus B01 Macro-F1@0.5: **-0.0506820342**. B03 has 247/859 zero-positive-prediction Studies at threshold 0.5, versus B01's 230/859.
- Best model file: 28,463,720 bytes (27.15 MiB); portable `state_dict`.

## Classes of interest

| Class | Positive Studies | B01 AUC | B03 AUC | Delta |
|---|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 24 | 0.73024 | 0.71080 | -0.01944 |
| 4 Pneumonia | 68 | 0.72567 | 0.72129 | -0.00439 |
| 6 Cardiomegaly | 177 | 0.73066 | 0.76349 | +0.03283 |
| 8 Lung Opacity | 214 | 0.69020 | 0.71607 | +0.02587 |

Class 1 has only 24 positive validation Studies; these individual class changes were not separately confidence-tested. The full ten-class comparison is in `per_class_auc_comparison.csv`.

## Training duration, memory, and overfitting

- Wall-clock B03 training time: 3,998.28 seconds (1 h 6 min 38 s) for 9 epochs. Recorded epoch durations total 3,992.36 seconds. B01 took 5,738.20 seconds (1 h 35 min 38 s) for 13 epochs; the total-time reduction is mostly from fewer epochs.
- B03 GPU whole-board memory usage from 132 running `nvidia-smi` samples: 4,971–5,031 MiB, maximum GPU utilization 99%. B01's sampled whole-board maximum was 4,578 MiB, but unrelated processes and different monitoring windows prevent a process-memory comparison. Both ran on the local RTX 4060 Laptop GPU, not the official T4.
- B03 AUC peaked at epoch 3 (0.78944). It first declined at epoch 4 (0.78734), then fell to 0.78277 at epoch 5 and 0.77137 at epoch 6 as training loss declined from 0.28710 to 0.20210. The decline is clear by epoch 6; final epoch 9 AUC was 0.75896, loss 0.10998.
- B01 AUC peaked at epoch 5, first declined at epoch 6, and showed a clear sustained drop by epoch 7. Lowering the backbone learning rate did not delay the visible onset of overfitting in this run.

Sources: `../config.yaml`, `../metrics.jsonl`, `../best_metrics.json`, `../validation_analysis/analysis.json`, `paired_comparison.json`, `per_class_auc_comparison.csv`, and `../gpu_monitor.csv`. The prelaunch difference audit is in `outputs/analysis/b03_preflight/`.
