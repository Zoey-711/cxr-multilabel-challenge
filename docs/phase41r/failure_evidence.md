# Phase41 failure evidence and limits

Source: `results/phase41_batch/reports/experiment_status.csv` at result commit `5d825ab199d23b54c1780af9692cc72ea24436d0`.

| Task | GPU | Reported epoch | Best epoch | Status |
| --- | ---: | ---: | ---: | --- |
| P41_BASE_F02 | 0 | N/A | 5 | DONE |
| P41_FUSION_F02 | 1 | 11 | 5 | FAILED |
| P41_BASE_F03 | 2 | 10 | 4 | FAILED |
| P41_FUSION_F03 | 3 | 10 | 4 | FAILED |
| P41_BASE_F04 | 4 | 11 | 5 | FAILED |
| P41_FUSION_F04 | 5 | 11 | 5 | FAILED |
| P41_BASE_F05 | 6 | 11 | 5 | FAILED |
| P41_FUSION_F05 | 7 | 11 | 5 | FAILED |

All seven status records end in `FloatingPointError: Nonfinite gradients` at `src/cxr/phase39_run.py:148` in the old code. The check is already **after** `GradScaler.unscale_`; the fault is not an early check before unscaling. The old code immediately raises before `GradScaler.step/update` can distinguish an FP16 backward overflow from a persistent invalid state. It also calls `clip_grad_norm_` before rejecting a nonfinite norm. These facts establish a recovery-handling defect. They do **not** prove FP16 overflow was the sole original trigger.

The synchronized lightweight result has only the BASE Fold2 completed experiment folder. It omits failed task full logs, last successful batch identifiers, and old `last_training.pt` files. Therefore the exact input/target/logit/loss values, GradScaler scale progression, and old checkpoint finiteness are UNKNOWN locally. `scripts/phase41r_audit_failures.py` checks the original server files without modifying them. `scripts/phase41r_diagnose.py` checks a bounded real-data replay and 32 optimizer groups on a single authorized GPU. Full repair confidence requires observing the new tasks past the original failure epoch, with skip events and update counts reviewed.

The repaired seven tasks start from the same frozen ImageNet initialization and patient folds in a new output root. Previous FAILED states are not imported. The one old DONE model is copied only after its original hash receipt is verified, and the new receipt records its historical identity. No learning rate, weight decay, resolution, augmentation, EMA decay, model structure, batch size or scheduler setting changes. AMP overflow handling and instrumentation change the numerical training protocol; compare repaired folds against frozen Fold1 with this qualification.
