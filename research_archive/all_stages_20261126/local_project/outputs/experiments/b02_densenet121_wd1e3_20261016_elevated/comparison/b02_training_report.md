# B02 training outcome

- Configuration check: B01 and B02 both used microbatch 16, accumulation 1, effective batch 16. B02 was corrected from the earlier 4 × 4 proposal before training. Only weight decay (0.0001 → 0.001), permitted patience (8 → 6), and experiment name differ.
- Initialization: fresh torchvision DenseNet121 ImageNet weights, SHA-256 `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`; no B01 checkpoint loaded.
- Completion: early stopped after epoch 11. Best epoch 5, training BCE loss 0.25272, independently revalidated Study-level Macro-AUC 0.78939 and Macro-F1@0.5 0.29922 across all 859 validation Studies and ten classes.
- At threshold 0.5, 234/859 Studies have no positive class prediction.
- Wall-clock training time: 5,017.63 seconds (1 h 23 min 38 s); sum of recorded epoch durations: 5,011.74 seconds.
- Best portable `state_dict`: 28,463,720 bytes (27.15 MiB).
- GPU: local NVIDIA RTX 4060 Laptop GPU. `nvidia-smi` board usage across 166 running samples was 4,520–5,160 MiB, maximum utilization 99%. This is whole-board usage, not a process-only peak; official T4 timing and memory were not tested.
- Overfitting: AUC peaked at epoch 5 (0.78939), then fell at epoch 6 (0.78058), epoch 7 (0.77810), and epoch 11 (0.74300) while training loss fell to 0.06231.
- B02 minus B01 Macro-AUC: +0.00100. Paired patient-cluster bootstrap 95% CI: [-0.01151, +0.01372], 2,000/2,000 valid resamples across 228 patients. The interval includes zero, so this run does not establish an AUC improvement.
- B02 minus B01 Macro-F1@0.5: +0.01672. The fixed threshold is a secondary metric; no threshold tuning or test prediction was performed.

Sources: `../metrics.jsonl`, `../best_metrics.json`, `../validation_analysis/analysis.json`, `paired_comparison.json`, `per_class_auc_comparison.csv`, and `../gpu_monitor.csv`.
