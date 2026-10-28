# Phase41-R: numerical repair and independent seven-task recovery

## What is established

The published Phase41 status at result commit `5d825ab199d23b54c1780af9692cc72ea24436d0` records `P41_BASE_F02` DONE and seven FAILED tasks. All seven fail at the same `Nonfinite gradients` check around epochs 10–11; failures span BASE/FUSION and GPU 1–7. The lightweight result branch does not contain the seven complete queue logs or `last_training.pt` files, so the exact failing batches, gradient distributions, and checkpoint finiteness remain unverified. The common code path is the strongest supported explanation; a transient FP16 overflow is a hypothesis pending the real-data diagnostic.

The old implementation called `GradScaler.unscale_`, then clipped gradients and raised on nonfinite norm **before** allowing `GradScaler.step/update` to skip an overflowed update and reduce its scale. The repair checks the norm before modifying gradients; a scaler-confirmed overflow skips the optimizer, scheduler and EMA update, clears gradients, logs the event, and reduces the scale. A nonfinite loss, non-AMP nonfinite gradients, four consecutive skips or 20 skips in one epoch still fail. Learning rates, model structure, labels, augmentation, microbatch, accumulation, EMA decay, early stopping and fold assignments are unchanged. This is a numerical handling protocol revision, so report the repair version separately from frozen Fold1.

## Server steps

Use the `friend-gpu-batch` commit supplied with the delivery; do not use `main`. Copy `configs/phase41r_server.example.yaml` to `configs/phase41r_server.local.yaml` and set all authorized absolute paths, GPU IDs and a **new** GPU-hour budget. Keep `upload.enabled: false` unless the friend has already checked private repository permission and approved incremental result publishing. If publishing is approved, set it to `true` **before preflight** and leave that config unchanged throughout training. `recovery_source_root` is the original frozen Phase41 output; `output_root` must be a new empty Phase41-R directory. The original source is read only. The repair command imports `P41_BASE_F02` into its new output only after checking the original DONE receipt and every listed SHA-256; it does not train that task again. Seven failed tasks start from the approved ImageNet initialization in the new directory. Old failed optimizer states are never loaded. Within the new repair run, complete-epoch checkpoints can resume normally.

Run from the repository root using the same Python environment as the prior Phase41 batch:

```bash
python scripts/phase41r_audit_failures.py --config configs/phase41r_server.local.yaml
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action preflight
CUDA_VISIBLE_DEVICES=1 python scripts/phase41r_diagnose.py --config configs/phase41r_server.local.yaml --task P41_FUSION_F02 --steps 32
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action run
```

The bounded diagnostic uses real Fold2 training images for 32 optimizer groups; it writes `diagnostic/P41_FUSION_F02/result.json` and a reloadable probe checkpoint under the **new** root. It does not create a formal model. If it fails, do not run the batch. A 32-step pass confirms the repair path works on that sample; it cannot establish the late-epoch failure is gone. The batch continuously records any AMP skips and fails on persistent overflow.

`run` repeats the mandatory remote preflight checks and reuses its verified receipt. It refuses to launch any formal task until the matching 32-step real-data diagnostic says PASS. It also refuses an empty budget. For a cautious rollout, set `gpu_ids: [1]`, `max_concurrent: 1` and run the same process first; that config and preflight identity must stay fixed during a run. To use all eight GPUs, begin with a **new empty** output root and a matching config after that limited validation. Do not edit resource settings inside an active output root.

Status, graceful stop, and resume use:

```bash
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action status
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action stop
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action resume
```

The old seven failed runs remain in the historical folder. The audit report is `reports/failure_audit.json` in the new output. New run evidence is in `diagnostic/`, `preflight/`, `queue_logs/`, per-task `train.log`, `amp_events.jsonl`, and `nonfinite_diagnostic.json` on failure. Each completed model gets `DONE.json`, best weight, checkpoint, OOF/Val/Test probabilities, config, metrics, hashes and manifest. Full fivefold analysis is emitted only after all required folds are complete; until then the OOF is partial.

Optional incremental upload uses the **separate** `results/phase41r-recovery` branch and `phase41r-recovery-v1` release. Never point this repair run at `results/phase41-fivefold` or its old Release. Before enabling upload, confirm the private repository permission, that these names are unused, and that the data sharing rules permit the intended prediction files. If training ran with upload disabled, make a separate copy of the config with upload enabled **only for the upload action**. The manual upload command is:

```bash
python scripts/phase41_run.py --config configs/phase41r_server.local.yaml --action upload
```

Changing the training config after a run has started changes the authorization identity. Keep that config fixed. No original Phase41, local Phase17, E08 or leaderboard assets are modified by these commands.
