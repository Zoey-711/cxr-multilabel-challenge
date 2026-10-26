# Phase 15: independent-patient generalization protocol

Status (2026-10-18): **P15A formal training is running**. The Phase 15 sequential executor is waiting for P15A; it will independently validate and freeze P15A, then train/validate/freeze P15B, lock the fixed fusion, and score Internal Holdout once. The frozen Phase 5 split and all prior checkpoints/submissions remain untouched. The machine is an RTX 4060 Laptop with 8,188 MiB reported VRAM; results from it do not certify official T4 latency.

## Frozen population and leakage boundary

| Partition | Patients | Studies | Images | Use |
|---|---:|---:|---:|---|
| Internal Train | 9,949 | 16,667 | 27,301 | The only official Train images allowed to update weights |
| Internal Holdout | 1,106 | 1,846 | 3,043 | One final evaluation after model selection is locked |
| Official validation | 228 | 859 | 1,422 | Select each best checkpoint by Study Macro-AUC |

The frozen patient, Study and image indices have matching SHA-256 digests, no duplicate keys and no patient crossing Internal Train/Holdout or official validation. All ten classes have both positive and negative Studies in each internal partition. The rarest Holdout class, class 1, has 46 positive and 1,800 negative Studies; the complete counts are in `outputs/analysis/phase15_preflight_20261018/preflight.json`. Its 43 positive patients are recorded in the frozen Phase 13 audit. Split SHA-256 values and source ZIP path are also in that preflight JSON. Preflight checks compare every indexed image and its Study label to the original official Train CSV, then return only Internal Train rows to the DataLoader. The original ZIP is not changed or extracted.

The existing B01/M02 checkpoints saw Internal Holdout patients during their original full-Train runs. They must **never** be used to score this Holdout. No Holdout image or label enters gradient updates, early stopping, optimizer/scheduler state, threshold selection, architecture choice, augmentation choice or fusion-weight selection. The previously reused official validation set is used only for each checkpoint's selection under this protocol; this limits claims of broader generalization, even though the Internal Holdout patients are excluded from the new training runs.

## Two fresh retrains

| Run | Exact runtime control | Fresh ImageNet source SHA-256 | Micro / accumulation | Other training settings |
|---|---|---|---:|---|
| P15A DenseNet-121 | B01 `config.yaml` | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` | 16 / 1 | B01 augmentation, BCE, AdamW, LR, cosine + 2 warmup epochs, seed 42, patience 8 |
| P15B ConvNeXt-Tiny | M02 `config.yaml` | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` | 8 / 2 | M02 conservative CXR augmentation, BCE, AdamW, LR, cosine + 2 warmup epochs, seed 42, patience 6 |

Each training YAML copies its control's **actual saved runtime configuration**. The only differences are the new experiment name, the frozen Internal Train index and a SHA-256 assertion for the same original torchvision ImageNet file. B01 has no training augmentation; M02 has `conservative_cxr_v1`. That existing difference is retained as requested, so a direct P15A/P15B contrast does not isolate backbone architecture. Both use 384 aspect-preserving black padding, ImageNet normalization, image-level shuffle, ten BCE logits, Study Mean probability aggregation, and official validation Study Macro-AUC in max mode. No existing best checkpoint is loaded, and no new external medical data is used. There are 1,707 optimizer groups per internal-Train epoch for both models, including the final partial batch.

Formal outputs are `outputs/experiments/p15a_densenet121_internal_train_384/` and `outputs/experiments/p15b_convnext_tiny_internal_train_384/`. P15A exists and is running; P15B is created only after P15A completes and its independent validation succeeds. The training script refuses an occupied output directory. Both fresh runs are capped at 30 epochs. Each run saves the best official-validation Macro-AUC checkpoint, full logs, exact config, seed and pretrained-file provenance. Early stopping follows its own control (8 for A, 6 for B); it never reads the Internal Holdout.

## Locked final comparison

1. Complete both fresh runs. Independently reload both selected `best_model.pt` state dictionaries and recompute the official 859-Study validation metrics. Freeze model file hashes and deterministic inference settings before reading any Internal Holdout image.
2. Define per model: sigmoid each image's ten logits, then average probabilities across all images of the same `(Subject_id, Study_id)`. Fix the only planned ensemble as `0.5 × P15A_StudyMean + 0.5 × P15B_StudyMean`. Do not search weights, thresholds or aggregation rules.
3. Read the Internal Holdout **once** after the above choices are locked. Require exact 1,846-Study/1,106-patient coverage and identical key/label order for A, B and fusion. Report three Study Macro-AUC values, ten per-class AUCs and support counts; report AP and Macro-F1@0.5 as descriptive secondary metrics.
4. Use 2,000 paired patient-cluster bootstrap resamples with seed 42. Each draw resamples 1,106 Subject_ids with replacement and retains every Study of each drawn patient. Use the same draw for A, B and fusion. Report individual Macro-AUC percentile intervals and paired 95% difference intervals for B−A, fusion−A and fusion−B. Exclude and count any resample lacking a positive or negative Study in any class; never substitute an AUC value. Report valid/invalid draw counts.
5. Archive raw image probabilities, Study probabilities, class metrics, bootstrap settings/results and SHA-256 manifest under a new versioned analysis directory. Do not turn this Holdout into a new tuning set after reading its results.

## GPU feasibility and time budget

The two **one-optimizer-group** preflights used the real Internal Train filter, forward/backward FP16 AMP and exact micro/accumulation settings. Both passed without a checkpoint or validation run. Peak **PyTorch allocated** GPU memory was 3,088 MiB for A and 1,725 MiB for B on the 8,188 MiB RTX 4060 Laptop. These are measured training-step peaks, not complete-system VRAM totals or guarantees against later workstation load. The preflight records are in `outputs/analysis/phase15_preflight_20261018/gpu_preflight.json`.

Historical B01 mean epoch time was 7.35 min and M02 was 22.97 min on the same host, including official validation. Scaling the training-image portion from 30,344 to 27,301 images gives rough **lower** bounds of 6.61 and 20.67 min/epoch; the unscaled history is a conservative upper planning bound. If stopping at their original 13 and 11 epochs, two-model GPU time is roughly **5.22–5.80 hours**. At the 30-epoch ceiling for both, **13.64–15.16 hours**. Holdout inference, independent checkpoint verification and file I/O add time. These are forecasts, not measured Phase 15 training results; early-stop epochs may change.

## Later candidates, after the independent comparison

| Candidate | Training cost | Inference and storage cost | Main risk / decision |
|---|---|---|---|
| EMA weights | Same epoch count plus a parameter-average update each optimizer step and an extra model copy during training; exact overhead needs measurement. | Choosing EMA *instead of* raw weights gives one forward pass; retaining both roughly adds another model file. | Needs an online EMA during a **new** run; a frozen best checkpoint cannot reproduce it. DenseNet BatchNorm statistics require buffer handling or a post-training update. Fix decay/start and checkpoint-selection rule before running. |
| Second random seed | One full additional fresh run: on Internal Train roughly 1.4–1.6 h for DenseNet at the historical stop or 3.8–4.2 h for ConvNeXt; actual stop may differ. | Adds a full model file (about 28.5 MB or 111.4 MB) and one more image forward pass. | With the same ImageNet backbone initialization, seed changes mainly the fresh head, sample order and stochastic augmentation; diversity may be smaller than a fully independent random initialization. Gain is not guaranteed, and selecting seeds on the reused official validation set increases selection bias. |
| Two-view TTA | No new weight training; one extra validation inference pass to evaluate a predeclared transform. | Roughly one extra image forward pass for every model receiving TTA, with transform/decoding overhead. E03 already uses four models. | Geometric changes or crops may alter anatomy/markers. A predeclared, anatomy-preserving photometric view is a safer first candidate than flips or crops, but probability shifts and latency must be measured. |

PyTorch 2.10 documents `AveragedModel` for EMA, including buffer/BatchNorm handling ([official documentation](https://docs.pytorch.org/docs/2.10/generated/torch.optim.swa_utils.AveragedModel.html)). Independent initializations can provide diverse predictions ([Fort et al.](https://arxiv.org/abs/1912.02757)); a second seed still needs an actual local gain. TTA can turn some previously correct predictions into errors ([Shanmugam et al.](https://arxiv.org/abs/2011.11156)), so it requires a fixed transform and paired validation rather than a broad search.

**Priority:** authorize P15A/P15B independent-patient retraining first. After the one-time Holdout comparison, decide whether there is enough generalization evidence to justify a separate EMA experiment. A new-seed ensemble costs more GPU time and inference latency; TTA is last because the existing multi-model inference budget is already tight. No EMA, seed, or TTA experiment is launched in Phase 15 preparation.
