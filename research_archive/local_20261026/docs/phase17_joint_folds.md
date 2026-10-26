# Phase 17 joint patient-fold pilot

The official Train archive contains 11,055 patients, 18,513 Studies and 30,344 images. The fixed seed-42 five-fold index is at `outputs/analysis/phase17_patient_folds/`. Its patient, Study and image CSV SHA-256 hashes are recorded in `fold_sha256.json`; each patient has exactly one fold. Official Val is checked for overlap but does not determine fold assignments. Hidden Test is never read.

| Fold | Patients | Studies | Images | Class 1/4/8 positive patients |
|---|---:|---:|---:|---|
| 1 | 2,211 | 3,687 | 6,012 | 115 / 257 / 659 |
| 2 | 2,211 | 3,731 | 6,130 | 114 / 256 / 658 |
| 3 | 2,211 | 3,701 | 6,099 | 114 / 257 / 659 |
| 4 | 2,211 | 3,692 | 6,051 | 115 / 257 / 660 |
| 5 | 2,211 | 3,702 | 6,052 | 114 / 257 / 659 |

The complete 10-class positive Study and patient support table is in `fold_report.json`. `select_fold_rows` verifies all three index hashes, every source Train image key, Study label, patient assignment, archive member and expected count before returning train or OOF rows.

## Controlled Fold 1 models

D17-F1, C17-F1 and S17-F1 derive from the **actual saved runtime** configurations of B01, M02 and T01. The only model/config differences are the experiment name, frozen four-fold Train selector and an assertion of the same ImageNet weight SHA-256. Batch/accumulation are 16/1 for D17 and 8/2 for C17/S17. Early stopping uses official Val Study Macro-AUC, strict improvement, patience 8/6/6, at most 30 epochs. The current fold never selects an epoch. All outputs use image sigmoid probabilities followed by Study Mean.

Phase17 uses nonpersistent DataLoader workers for exact epoch-boundary recovery. Each `last_training.pt` includes model, AdamW, scheduler, GradScaler, early stopping state, Python/NumPy/PyTorch CPU/CUDA RNG and Train/Val DataLoader generator states. This worker lifecycle change affects data loading only; augmentation operators and ranges remain those of each control. The recovery guarantee is for the next completed epoch boundary. An interruption within an epoch restarts that epoch from the previous complete checkpoint.

Real two-step GPU preflights on the local RTX 4060 Laptop passed with finite loss/gradient and zero AMP skips. Peak PyTorch allocated memory was 3,145 MiB (D17), 1,932 MiB (C17), and 2,220 MiB (S17). A synthetic GPU two-epoch uninterrupted-versus-resumed test matched model weights, learning rate, GradScaler and early stopping state exactly. All three official model factories also passed an ImageNet-hash-checked `state_dict` checkpoint save/reload round trip with identical ten-class logits.

## Prespecified full-fold evaluation

If Fold 2–5 are later authorized, form D5/C5/S5 by equal Study-probability averaging within each backbone on official Val, then `E17 = 0.25 D5 + 0.25 C5 + 0.5 S5`. For each official Train patient, the OOF contribution from a backbone must come **only** from that patient's held-out fold model. Compare E17 with frozen E07 on the same 859 official Val Studies using paired patient bootstrap; no fusion-weight search. Fold 1 OOF is a pilot result, not a complete five-fold OOF estimate or a sufficient basis for picking a final ensemble. Phase15 Holdout remains closed to model selection.

## File-size and cost estimate before Fold 1 training

Observed frozen checkpoint sizes: DenseNet 28,463,720 bytes, ConvNeXt 111,378,631 bytes, Swin 110,405,322 bytes. Fifteen fold checkpoints would be approximately **1,251,238,365 bytes**, above the 500,000,000-byte deployment limit. Frozen E07 is **307,175,113 bytes**. Feasible *size-only* designs to evaluate later are D5+C1+S1 (~364,102,553 bytes), D5+C2+S1 (~475,481,184 bytes), or D5+C1+S2 (~474,507,875 bytes). Choosing which folds to retain requires a separately specified analysis and latency measurement; no Fold 1 ranking will be used to search combinations. If a full ensemble yields clear gains, knowledge distillation can be designed as a later controlled experiment using only approved Train data.

Historical median epoch times on the full Train were 7.3 minutes for B01, 22.3 for M02 and 22.1 for T01. Four-fold Train contains about 80% as many images, so initial Fold 1 estimates are roughly 6, 18 and 18 minutes per epoch plus validation. These are planning estimates and will be replaced with measured elapsed times. Five-fold deployment latency has not been measured on the official T4; the 100 ms figure is a competition reference.

Only the three Fold 1 runs are authorized. No hidden-test prediction or submission creation is part of this pipeline.
