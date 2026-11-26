# CXR Multilabel Challenge

An installable `src`-layout project for ten-label, **study-level** chest X-ray classification. B01, B02, and B03 are completed and frozen; M01 is prepared for a separately authorized run. Real data, pretrained weights, and experiment outputs stay outside Git. In the preliminary round only `submission.csv` is submitted. Model and inference packaging for the final round awaits official instructions.

## Environment and setup

Official inference target: Python 3.12, PyTorch 2.10.0, torchvision 0.25.0, CUDA 12.8, cuDNN 9, NVIDIA T4 16 GB. Training and inference dependencies are separate in `requirements/train.txt` and `requirements/inference.txt`, and in the `train`/`inference` extras of `pyproject.toml`.

Run these commands from this project directory in a Python 3.12 environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements/inference.txt
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/check_model.py --config configs/b01_densenet121_384.yaml
python scripts/check_environment.py --config configs/b01_densenet121_384.yaml
```

`check_model.py` builds DenseNet-121 with `pretrained=False`, so it checks structure without downloading weights. Import packages using `cxr.*` after the editable install; scripts do not change `sys.path`.

Local environment check (2026-10-16): the project `.venv` has Python 3.12.4, torch 2.10.0+cu128, torchvision 0.25.0+cu128, and cuDNN 9. Driver 581.08 is installed; CUDA GPU operations pass. This workstation has an RTX 4060 Laptop GPU with 8 GB, so T4-specific latency and memory checks still need the official server. See [`docs/environment.md`](docs/environment.md).

## Phase 2: inspect the supplied archive

The real dataset is referenced by absolute path in [`configs/phase2_cxr_first.yaml`](configs/phase2_cxr_first.yaml): `F:/数据集/cxr_first.zip`. Scripts stream images from this archive and do not extract or modify it. The inspected CSVs are `cxr_first/train.csv`, `cxr_first/valid.csv`, and `cxr_first/test.csv`. The train and validation `Predict_class` field lists positive original IDs separated by semicolons. The organizer describes these as complete multilabel annotations, so the DataLoader treats unlisted IDs as 0 **for competition targets** and returns an all-valid mask. This does not assert clinical absence. An optional `unlisted_policy="unknown"` retains the conservative masked behavior. The test split has no target or mask.

Use the existing project `.venv` from this directory:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --config configs\phase2_cxr_first.yaml --require-cuda
.\.venv\Scripts\python.exe scripts\phase2_audit.py --config configs\phase2_cxr_first.yaml --workers 4
.\.venv\Scripts\python.exe scripts\smoke_dataloader.py --config configs\phase2_cxr_first.yaml
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs\audits\pytest_tmp
```

The audit writes `data_audit.json`, `class_distribution.csv`, `study_distribution.csv`, and `label_cooccurrence.csv` under `outputs/audits`. It checks every referenced image by reading the ZIP member, validating its CRC, decoding the JPG, and comparing dimensions to metadata. The smoke test writes `dataloader_test.json` in the same directory. `class_distribution.csv` reports competition target positives and negatives at study and image level, plus the fact that no explicit negative/uncertainty code occurs. `study_distribution.csv` has one row per study. The cooccurrence matrix counts study-level positive pairs. The DataLoader uses deterministic aspect-preserving resize with black padding and ImageNet normalization; training augmentation has not been added.

Study-level validation helpers align fused image probabilities with complete competition targets from `study_targets_from_rows`. The earlier baseline weight download and training were separately authorized; the M01 ImageNet weight has also been downloaded for its requested preflight.

## Baseline configuration

[`configs/b01_densenet121_384.yaml`](configs/b01_densenet121_384.yaml) records DenseNet-121, ImageNet as the *planned* pretrained source, 384×384 input, 10 outputs, batch size 16, 30 epochs, AdamW with backbone LR 1e-4 and classifier LR 3e-4, weight decay 1e-4, BCE with logits, cosine schedule with two warmup epochs, FP16 AMP, gradient clipping 1.0, early stopping patience 8, seed 42, and four workers. None of these values has been validated on the competition data. The model factory defaults to `pretrained=False` to prevent an implicit download.

## Remaining decisions and future inference

The supplied ZIP contains all available images and metadata; no additional data files are expected. The official complete-annotation statement resolves the competition label mapping: listed class IDs are 1, unlisted class IDs are 0. Every labeled study has a nonblank list, and `No Finding` never co-occurs with a disease class. The CSV has no explicit uncertainty code. See the [audit report](outputs/audits/audit_report.md) for the evidence and clinical interpretation limit. This label rule does not authorize a training run.

The older `scripts/audit_data.py` supports an extracted image-level CSV manifest after its field mapping is filled. It does not read the supplied archive. For this dataset use `scripts/phase2_audit.py` above. Example for a future extracted manifest:

```powershell
python scripts/audit_data.py --config configs/MANIFEST_CONFIG.yaml --split train
```

The manifest audit command requires a separately configured manifest. The completed archive audit includes cross-split checks and full image validation.

The organizer permits multiple rows per Study. The project now defaults to `inference.submission_mode: threshold` with a configurable 0.5 threshold, so one row is emitted per selected class. Whether all ten probabilities may be submitted regardless of threshold, and how to represent a Study with no selected class, remain unresolved. The writer fails instead of silently omitting such a Study. See [`docs/rules.md`](docs/rules.md).

After a `state_dict` checkpoint is available, `predict.py` reads the original ZIP test split directly with the local config and generates the preliminary-round CSV with the threshold policy. It does not read test labels. An explicit `--policy all_classes` remains available only for a separately confirmed rule; it is not the default. Threshold mode rejects a Study with no selected class until the official empty-study rule is known.

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\b01_densenet121_384_local.yaml --checkpoint outputs\experiments\RUN\best_model.pt --output outputs\submissions\submission.csv
.\.venv\Scripts\python.exe scripts\predict.py --config configs\b01_densenet121_384_local.yaml --checkpoint outputs\experiments\RUN\best_model.pt --threshold 0.6 --output outputs\submissions\submission.csv
```

The script requires CUDA, loads only a portable `state_dict`, applies deterministic 384×384 ImageNet normalization, fuses images by study, maps model index 0–9 to class ID 1–10, checks the checkpoint is at most 500 MB, and writes the required CSV columns. Paths to data and checkpoint are supplied at runtime; the checkpoint stores no absolute paths.

On a CUDA machine, measure decode + preprocessing + model forward + probability transfer for one image:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_inference.py --config configs\b01_densenet121_384_local.yaml --checkpoint outputs\experiments\RUN\best_model.pt --archive-split test --warmup 10 --runs 100
```

Only a run on the official T4 can establish compliance with the 100 ms latency limit.

## Baseline training

The user authorized the first ImageNet weight download and Baseline run on 2026-10-16. The local run uses [`configs/b01_densenet121_384_local.yaml`](configs/b01_densenet121_384_local.yaml) and the official torchvision `DenseNet121_Weights.IMAGENET1K_V1` weight file. The experiment directory records a copied config, seed, environment, pretrained-weight URL and SHA-256, per-epoch Study-level Macro-AUC/F1, best portable `state_dict`, and resumable training state. Training reads only train images; validation aggregates probabilities across all images of each validation Study. Test labels are never read. On this 8 GB RTX 4060 Laptop GPU, a real-data two-step preflight passed at batch size 16 with about 3,145 MiB peak allocated GPU memory. B01 ended by early stopping at epoch 13; its best checkpoint is epoch 5 with validation Study-level Macro-AUC 0.78839 and Macro-F1 0.28250. Full per-class diagnostics are in `outputs/experiments/b01_densenet121_384_20261016/validation_analysis/`.

Run or resume from this project directory; on Windows, four DataLoader workers require multiprocessing access:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\b01_densenet121_384_local.yaml --run-id b01_densenet121_384_20261016
.\.venv\Scripts\python.exe scripts\train.py --config configs\b01_densenet121_384_local.yaml --run-id b01_densenet121_384_20261016 --resume
```

The preflight option runs a requested number of training batches, writes only environment and provenance records, and saves no model checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\b01_densenet121_384_local.yaml --run-id b01_preflight_example --preflight-steps 2
```

While B01 runs, `monitor_training.py` only reads the existing status/log and samples GPU board usage via `nvidia-smi`; it appends `gpu_monitor.csv` without changing the model or config. After `status.json` reaches `complete`, `analyze_validation.py` independently rereads validation images, fuses probabilities by Study, cross-checks the best saved AUC, and writes class diagnostics and the zero-prediction Study count. It refuses to use GPU while training is still running.

```powershell
.\.venv\Scripts\python.exe scripts\monitor_training.py --config configs\b01_densenet121_384_local.yaml --run-dir outputs\experiments\b01_densenet121_384_20261016 --interval 30
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\b01_densenet121_384_local.yaml --run-dir outputs\experiments\b01_densenet121_384_20261016
.\.venv\Scripts\python.exe scripts\analyze_when_complete.py --config configs\b01_densenet121_384_local.yaml --run-dir outputs\experiments\b01_densenet121_384_20261016 --poll-seconds 30
```

## Phase 4: frozen B01 analysis and B02 preparation

The B01 Phase 4 analysis reads the existing per-Study validation predictions and epoch log. It writes only to `outputs/analysis/b01_phase4`, outside the frozen B01 experiment directory. It computes threshold metrics, patient-cluster bootstrap confidence intervals, and an SVG curve for all epochs. The B02 preflight compares its configuration with B01, verifies the data archive and ImageNet weight hash, and checks that the planned output directory is absent. Neither command starts training.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_b01.py --config configs\b01_densenet121_384_local.yaml --run-dir outputs\experiments\b01_densenet121_384_20261016 --output-dir outputs\analysis\b01_phase4 --bootstrap-replicates 2000 --seed 42

.\.venv\Scripts\python.exe scripts\preflight_b02.py --config configs\b02_densenet121_wd1e3.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --planned-run-id b02_densenet121_wd1e3_20261016 --output-dir outputs\analysis\b01_phase4
```

The final B01 runtime configuration and log confirm microbatch 16, accumulation 1, and effective batch 16. `configs/b02_densenet121_wd1e3.yaml` was corrected from the earlier 4-by-4 plan to the same 16-by-1 setting. Weight decay changes to 0.001, and early stopping patience changes to 6. All other training and validation settings match the actual B01 run. B02 starts from the same cached torchvision ImageNet weights with a fresh classifier; it does not resume from B01.

The authorized B02 run writes to `outputs/experiments/b02_densenet121_wd1e3_20261016_elevated`. The first attempted directory without the `_elevated` suffix contains only a failed sandbox startup log; no batch was trained there. The training and post-training commands are:

B02 completed by early stopping at epoch 11. Its best epoch was 5, independently revalidated Study-level Macro-AUC was 0.78939, and Macro-F1@0.5 was 0.29922. The paired patient-bootstrap Macro-AUC difference versus B01 was +0.00100 (95% CI [-0.01151, +0.01372]); see the run's `comparison/b02_training_report.md` for the full record.

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\b02_densenet121_wd1e3.yaml --run-id b02_densenet121_wd1e3_20261016_elevated

.\.venv\Scripts\python.exe scripts\monitor_training.py --config configs\b02_densenet121_wd1e3.yaml --run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --interval 30

.\.venv\Scripts\python.exe scripts\finalize_b02.py --config configs\b02_densenet121_wd1e3.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b02-run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --poll-seconds 30

.\.venv\Scripts\python.exe scripts\compare_b01_b02.py --config configs\b02_densenet121_wd1e3.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b02-run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --output-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated\comparison --bootstrap-replicates 2000 --seed 42
```

## B03 backbone learning-rate control

B03 uses a copy of B01's actual saved runtime configuration. The preflight report in `outputs/analysis/b03_preflight/` confirms that only the backbone learning rate (0.0001 to 0.00003), permitted early-stopping patience (8 to 6), and experiment name differ. Both runs use microbatch 16, accumulation 1, the same ImageNet weight hash, and the same 859-Study validation protocol. B03 starts from fresh ImageNet weights with a new classifier.

B03 completed by early stopping at epoch 9; the best epoch was 3. Independent validation Macro-AUC was 0.78944 and Macro-F1@0.5 was 0.23182. The paired patient-bootstrap Macro-AUC difference versus B01 was +0.00104 (95% CI [-0.01361, +0.01542]). See `outputs/experiments/b03_densenet121_lr3e5_20261016/comparison/b03_training_report.md`.

```powershell
.\.venv\Scripts\python.exe scripts\preflight_b03.py --config configs\b03_densenet121_lr3e5.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --planned-run-id b03_densenet121_lr3e5_20261016 --output-dir outputs\analysis\b03_preflight

.\.venv\Scripts\python.exe scripts\train.py --config configs\b03_densenet121_lr3e5.yaml --run-id b03_densenet121_lr3e5_20261016

.\.venv\Scripts\python.exe scripts\monitor_training.py --config configs\b03_densenet121_lr3e5.yaml --run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --interval 30

.\.venv\Scripts\python.exe scripts\finalize_b03.py --config configs\b03_densenet121_lr3e5.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b03-run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --poll-seconds 30

.\.venv\Scripts\python.exe scripts\compare_b01_b03.py --config configs\b03_densenet121_lr3e5.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b03-run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --output-dir outputs\experiments\b03_densenet121_lr3e5_20261016\comparison --bootstrap-replicates 2000 --seed 42
```

## Frozen B01/B02/B03 analysis and M01 preparation

The B01/B02/B03 run directories are frozen. `analyze_frozen_ensemble.py` reads their existing 859-Study validation predictions, verifies exact Study keys and targets, evaluates the sole prespecified three-model equal-probability fusion, and computes patient-level paired bootstrap confidence intervals. Its output is under `outputs/analysis/frozen_ensemble/`, including `frozen_manifest.json` with SHA-256 hashes of the frozen configs, metrics, best checkpoints, and validation predictions. The saved predictions have one row per Study, so image-level Mean versus Max cannot be recomputed from them.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_frozen_ensemble.py --config configs\b01_densenet121_384_local.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b02-run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --b03-run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --output-dir outputs\analysis\frozen_ensemble --bootstrap-replicates 2000 --seed 42
```

M01 uses official `torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`. The cached weight is recorded with its URL and SHA-256 in the preflight output. The M01 config retains B01's data, target encoding, ImageNet normalization, 384 by 384 inputs, BCE, optimizer, scheduler, seed, and Study-level validation. Based on the RTX 4060 Laptop 8 GB preflight, it uses microbatch 8 and accumulation 2 for effective batch 16. The training learning rates are an initial proposal, not a measured optimum.

`preflight_m01.py` reads eight real training images and tests FP16 forward/backward for batch 2, 4, and 8. It takes no optimizer step, saves a structure-only `state_dict` checkpoint, and verifies exact reload. Preflight results and the M01 preparation report are in `outputs/experiments/m01_convnext_tiny_384_preflight_20261016_elevated/`. The official T4 latency constraint remains untested.

```powershell
.\.venv\Scripts\python.exe scripts\preflight_m01.py --config configs\m01_convnext_tiny_384.yaml --output-dir outputs\experiments\m01_convnext_tiny_384_preflight_NEW
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs\audits\pytest_tmp
```

The authorized M01 run uses an independent run ID. `metrics.jsonl` records each epoch's Study-level scores, optimizer steps, AMP skipped steps, elapsed time, and GPU peak memory. The passive GPU monitor writes board-level memory samples. After training completes, `analyze_validation.py` loads the best checkpoint independently and saves both image-level and Study-level validation probabilities. `compare_m01.py` evaluates exactly the three requested M01 schemes using aligned Study IDs and patient-level paired bootstrap.

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\m01_convnext_tiny_384.yaml --run-id m01_convnext_tiny_384_20261016
.\.venv\Scripts\python.exe scripts\monitor_training.py --config configs\m01_convnext_tiny_384.yaml --run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --interval 30
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\m01_convnext_tiny_384.yaml --run-dir outputs\experiments\m01_convnext_tiny_384_20261016
.\.venv\Scripts\python.exe scripts\compare_m01.py --config configs\m01_convnext_tiny_384.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b02-run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --b03-run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --output-dir outputs\experiments\m01_convnext_tiny_384_20261016\comparison --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\finalize_m01.py --config configs\m01_convnext_tiny_384.yaml --b01-run-dir outputs\experiments\b01_densenet121_384_20261016 --b02-run-dir outputs\experiments\b02_densenet121_wd1e3_20261016_elevated --b03-run-dir outputs\experiments\b03_densenet121_lr3e5_20261016 --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --poll-seconds 30
```

`finalize_m01.py` waits for training completion, then runs the independent validation and comparison commands above, writes a concise final report, and records SHA-256 hashes for the M01 artifacts. Run it instead of calling those two post-training commands separately if automatic finalization is desired.

M01 finished by early stopping at epoch 9. Its best checkpoint is epoch 3 with independently reproduced validation Macro-AUC 0.79265 and Macro-F1@0.5 0.28660 across all 859 Studies. The three-model frozen ensemble remains 0.80267; the prespecified equal blend of M01 with that ensemble reached 0.80344, with a patient-level paired 95% delta interval that includes zero. The run saved 1,422 validation image probability vectors and 859 Study vectors, plus epoch logs and GPU memory records. See `outputs/experiments/m01_convnext_tiny_384_20261016/m01_final_report.md` and its `comparison/report.md`.

## Phase 5: M01 multiview analysis and M02 preparation

Phase 5 reads the frozen M01 image predictions and official training/validation metadata. It does not load model weights or train a model. Its outputs are in `outputs/analysis/phase5_m01_multiview/`, `outputs/analysis/phase5_m01_generalization/`, `outputs/analysis/phase5_internal_split/`, and `outputs/experiments/m02_convnext_tiny_384_aug_preparation_20261016/`.

The 859-Study M01 Macro-AUC is 0.79265 with Mean, 0.78722 with Max, and 0.79054 with an equal Mean/Max blend. Paired patient bootstrap (228 patients, 2,000 replicates) estimates Max minus Mean at -0.00543, 95% CI [-0.00934, -0.00241]. There are 391 single-image and 468 multi-image validation Studies. Multi-image Studies are 54.0% of training Studies but 72.0% of image-level training samples, so the current image-level BCE gives them more weight. This is an objective weighting difference, while the single/multi validation AUC gap may also reflect different cases and views.

M01 had no stochastic training augmentation. M02 adds only small train-only rotation (±3°), translation (±2%), brightness (±5%), and contrast (±5%), each group independently applied with probability 0.5. It keeps the same deterministic M01 preprocessing, model, data split, seed, learning rates, effective batch 16, BCE, optimizer, scheduler, and Study-level validation. The existing M01 preprocessing uses `ImageOps.pad`, which fits the complete rectangular image inside 384×384 and adds black padding. M02 has an independent configuration and preparation directory; no formal M02 run exists yet.

The internal fixed holdout is selected once from sorted official training Subject IDs with seed 42. It contains 1,106 patients, 1,846 Studies, and 3,043 images; the remaining 9,949 patients, 16,667 Studies, and 27,301 images form the internal training part. Index CSVs refer to the original ZIP and do not copy images. All patients remain in one part and neither part overlaps official validation patients. This is a future robustness check, not an additional training run.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_m01_multiview.py --config configs\m01_convnext_tiny_384.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --output-dir outputs\analysis\phase5_m01_multiview --bootstrap-replicates 2000 --seed 42

.\.venv\Scripts\python.exe scripts\audit_m01_generalization.py --config configs\m01_convnext_tiny_384.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --output-dir outputs\analysis\phase5_m01_generalization

.\.venv\Scripts\python.exe scripts\create_internal_split.py --config configs\m01_convnext_tiny_384.yaml --output-dir outputs\analysis\phase5_internal_split --seed 42 --holdout-fraction 0.1

.\.venv\Scripts\python.exe scripts\prepare_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --m01-config configs\m01_convnext_tiny_384.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --output-dir outputs\experiments\m02_convnext_tiny_384_aug_preparation_20261016

.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs\audits\pytest_phase5_all_tmp
```

M01's logged peak allocated/reserved GPU memory was 1,932/1,992 MiB on the local RTX 4060 Laptop; its passive GPU monitor saw 3,506 MiB peak board use. M02's projected time is roughly 8–11 minutes per epoch, or 4–5.5 hours for all 30 epochs; the actual GPU peak and early-stop duration need measurement in its future run. These local measurements do not establish official T4 latency compliance.

## Phase 6: preprocessing audit and M02 formal run

`audit_m01_preprocessing.py` selects extreme and near-square aspect ratios from both train and validation, writes six original/processed image pairs and a contact sheet under `outputs/analysis/phase6_preprocessing/`, and checks the processed tensor against the frozen M01 operation sequence. Visual inspection confirmed that PIL `ImageOps.pad` **fits the entire image inside 384×384 and adds black padding**. It applies no nonuniform resize or geometric crop. The earlier Phase 5 crop warning was incorrect and has been corrected in the analysis outputs. Complete fit resize plus padding is already the M01 protocol, so that same operation alone is not a distinct M03 experiment.

`preflight_m02_phase6.py` checks the frozen M01 config and hashes, M02's augmentation-only config difference, official ImageNet weight hash, independent output directory, image/label/Study alignment, augmented pixel finiteness, and unchanged validation/test preprocessing. The M02 comparison tests the combined augmentation strategy; it cannot isolate the contribution of rotation, translation, brightness, or contrast separately. Once its checks pass, the authorized formal run uses batch 8, accumulation 2, patience 6, and Mean Study fusion. The first local launch under the restricted Windows sandbox failed before any optimizer update because DataLoader worker pipes were denied; that failed directory is retained. The formal run was restarted with the same configuration in the independent `_elevated` directory.

```powershell
.\.venv\Scripts\python.exe scripts\audit_m01_preprocessing.py --config configs\m01_convnext_tiny_384.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --output-dir outputs\analysis\phase6_preprocessing

.\.venv\Scripts\python.exe scripts\preflight_m02_phase6.py --config configs\m02_convnext_tiny_384_aug.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --run-id m02_convnext_tiny_384_aug_20261016_elevated --output-dir outputs\analysis\phase6_m02_preflight_elevated

.\.venv\Scripts\python.exe scripts\train.py --config configs\m02_convnext_tiny_384_aug.yaml --run-id m02_convnext_tiny_384_aug_20261016_elevated

.\.venv\Scripts\python.exe scripts\monitor_training.py --config configs\m02_convnext_tiny_384_aug.yaml --run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated --interval 30

.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\m02_convnext_tiny_384_aug.yaml --run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated

.\.venv\Scripts\python.exe scripts\compare_m01_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --m02-run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated --output-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\comparison --bootstrap-replicates 2000 --seed 42

.\.venv\Scripts\python.exe scripts\freeze_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated

.\.venv\Scripts\python.exe scripts\finalize_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --m01-run-dir outputs\experiments\m01_convnext_tiny_384_20261016 --m02-run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated --poll-seconds 30

.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs\audits\pytest_phase6_tmp
```

The final paired M01/M02 analysis aligns all 859 Study IDs and ten labels, evaluates the original Mean aggregation, and resamples the same 228 patients for both models. Its report is stored inside M02's independent `comparison/` directory. `finalize_m02.py` waits for training completion and runs the three post-training commands above; run it instead of those three commands if automatic finalization is desired. No M03/M04 or test inference runs are part of Phase 6.

## Early stopping policy for future experiments

Completed B01/B02/B03/M01 run configs and outputs remain frozen. For a new run, omit `training.early_stopping_patience` to use the model default: **4 for DenseNet-121 optimization** and **6 for a newly trained backbone**. An explicit patience in a config is honored, so historical configs retain their original 8 or 6. Optional `training.early_stopping_min_delta` defaults to `0.0`.

The monitored quantity is Study-level `macro_auc`, with `max` mode. With `min_delta: 0.0`, only a strictly higher AUC resets patience. With a positive `min_delta`, patience resets when AUC improves by at least that amount over the last patience reset. Any strict new global AUC maximum still saves `best_model.pt`, even if the gain is below `min_delta`. This keeps the best checkpoint tied to the highest observed validation AUC. Equal AUC never replaces it.

Every future run writes best epoch, stop epoch, patience, min_delta, and stop reason (`patience_exhausted` or `max_epochs_reached`) to `status.json` and `metrics.jsonl`. The resumable checkpoint also stores the patience reference and best epoch. Resume rejects a changed config or a completed frozen run.

## Versioned E02 test submission candidate

**Cleanup update (2026-10-16):** The historical threshold-filtered directories `E02_20261016` and `E02_20261016_v2` were deleted at the user's request after the organizer required all ten classes per Study. Commands and descriptions below for those two directories document the earlier workflow and no longer apply to existing files. The valid E02 candidate is `outputs/submissions/E02_20261016_full10/submission.csv`; E01 is also retained. Valid per-image and per-Study source probabilities were hash-verified and archived in `outputs/analysis/e02_test_probability_provenance_20261016/`. Deletion is recorded in `submission_history.csv` and `submission_status_events.csv`.

`generate_e02_submission.py` verifies the frozen B01/B02/B03/M01 best-checkpoint SHA-256 hashes, runs GPU inference on all 2,218 original test ZIP images, mean-aggregates each model's image sigmoid probabilities to the same 1,414 Study IDs, and applies `0.5 × ((B01+B02+B03)/3) + 0.5 × M01`. It writes a new, non-overwritable version directory with the 1,414×10 Study probability matrix, each model's image and Study probabilities, `submission.csv`, the fusion config, a prediction check, zero-prediction Study IDs, and `manifest.json`. Files are marked read-only after generation. `outputs/submissions/submission_history.csv` records candidates separately from manually reported upload status and official score.

The candidate uses the organizer's default threshold of 0.5 with exact CSV columns `Subject_id,Study_id,Predict_class,Probability` and original class IDs 1–10. If a Study has no class above threshold, the candidate contains no row for it. The official rules supplied so far **do not say whether a zero-prediction Study may be omitted**. Both `E02_20261016` and `E02_20261016_v2` contain the same CSV with 560 omitted Studies, so their effective format status is blocked pending clarification. The second version was briefly marked ready based on an incorrect statement; that status was retracted in `submission_history.csv` and the append-only `submission_status_events.csv`, while its sealed files remain unchanged. The generator does not force No Finding, choose an argmax class, change the threshold, or upload the file. A new version name is required for any later candidate.

```powershell
.\.venv\Scripts\python.exe scripts\generate_e02_submission.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E02_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe scripts\verify_submission_bundle.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E02_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe scripts\reissue_confirmed_submission.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --source-version E02_20261016 --new-version E02_NEW_VERSION --output-root outputs\submissions --official-rule-reference "OFFICIAL_NOTICE_EXPLICITLY_ALLOWING_OMISSION"

.\.venv\Scripts\python.exe scripts\retract_submission_format_status.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E02_20261016_v2 --output-root outputs\submissions --reason "Official rules do not specify whether zero-prediction Studies may be omitted"

.\.venv\Scripts\python.exe scripts\verify_submission_bundle.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E02_20261016_v2 --output-root outputs\submissions

.\.venv\Scripts\python.exe -m pytest -q tests\test_versioned_submission.py --basetemp outputs\audits\pytest_e02_tmp
```

Run GPU inference outside the restricted Windows sandbox when DataLoader worker pipes are denied. **Neither candidate is currently confirmed compliant for upload**. The full 1,414×10 matrix and four model probability files remain available in both directories. Do not rerun the same version directory; choose a new version identifier instead. Actual upload status and official score must be entered by the user in `submission_history.csv`. Only run the reissue command above if the official organizer explicitly confirms the empty-Study rule; it was run once based on an incorrect statement and then retracted.

## Current modules

## E01 official result and label-free test shift audit

M02 completed on 2026-10-17 under the already approved augmentation-only configuration. It stopped at epoch 11 on Study Macro-AUC patience 6, with best epoch 5. Independent 859-Study validation reproduced Macro-AUC 0.79724427 and Macro-F1@0.5 0.29456875. The frozen best checkpoint is `outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/best_model.pt`; the complete run report is `m02_final_report.md` in that directory. The separate `outputs/analysis/m02_vs_e01_20261016/comparison.json` compares M02 with frozen M01 and E01 validation predictions, aligned by all 859 Study IDs. E01 remains at 0.80267447; M02 minus E01 is -0.00543020 with patient-cluster paired 95% CI [-0.01952105, 0.00574815]. No test labels, threshold search, new submission, or leaderboard upload were used.

E01 was manually submitted on 2026-10-16. The leaderboard displayed `75.51`; the organizer defines the primary ranking metric as Study-level Macro-AUC, but the leaderboard display scale has not been independently confirmed. If it is a 0–100 percentage display, the corresponding AUC is 0.7551. The original display score and this conditional interpretation are recorded in `outputs/submissions/E01_20261016_official_result.json`; `submission_history.csv` marks E01 `submitted`. Its frozen CSV, matrix, and checkpoints were not changed.

`audit_test_shift.py` reads split CSV metadata and samples ten decoded images per split from the original ZIP. It compares unlabeled image views, orientation, dimensions, image count per Study, patient counts, file coverage, and split overlap. Its report distinguishes measured differences from unverified explanations of leaderboard performance. It does not inspect hidden test labels or tune model weights/thresholds.

```powershell
.\.venv\Scripts\python.exe scripts\audit_test_shift.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --output-dir outputs\analysis\test_shift_20261016 --report docs\test_shift_audit.md

.\.venv\Scripts\python.exe scripts\verify_submission_bundle.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E01_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe scripts\compare_e01_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --m02-run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated --output-dir outputs\analysis\m02_vs_e01_20261016 --bootstrap-replicates 2000

.\.venv\Scripts\python.exe scripts\finalize_e01_m02.py --config configs\m02_convnext_tiny_384_aug.yaml --m02-run-dir outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated --output-dir outputs\analysis\m02_vs_e01_20261016 --poll-seconds 30
```

Run `compare_e01_m02.py` only after the M02 finalizer has completed independent validation. It aligns the same 859 Study IDs, reconstructs E01 from frozen validation probabilities, and reports M01/M02/E01 Study Macro-AUC and a patient-level paired confidence interval for M02 minus E01.
`finalize_e01_m02.py` is the optional wait wrapper for that same comparison; run one of these commands, not both.

## E01 all-class submission rule (2026-10-16 organizer staff clarification)

### E02 full-ten reissue

`reissue_e02_full10.py` validates the sealed original E02 bundle, all four frozen best-checkpoint SHA-256 digests, every source probability file, full test Study coverage, and the E02 fusion formula. It then copies the verified 1,414-by-10 matrix into a new version and writes all ten class probabilities for every Study. It does not load a model, run GPU inference, filter at 0.5, add `Gate`, or upload. The new directory contains `submission.csv`, `probability_matrix.csv`, `validation_report.json`, and `manifest.json`; the latter records source file and checkpoint hashes. The directory is sealed after creation. The original E02 directories remain unchanged.

```powershell
.\.venv\Scripts\python.exe scripts\verify_submission_bundle.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E02_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe scripts\reissue_e02_full10.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --source-version E02_20261016 --version E02_20261016_full10 --output-root outputs\submissions
```

The generator refuses to overwrite an existing version. `submission_history.csv` marks the new file `generated`; only the user can record a leaderboard upload and official score.

Staff clarified that **every test Study must have all ten original class IDs, one row per class**, in `Subject_id,Study_id,Predict_class,Probability` order. A `Gate` column is needed only if a nondefault classification threshold is used. E01 writes every probability without a threshold filter and uses no `Gate` column. It still reports how many Studies have no probability at least 0.5 for diagnostics. The two sealed E02 CSVs above are byte-identical threshold-filtered candidates and do **not** meet the newly clarified ten-row rule; their files are retained unchanged.

E01 uses only the SHA-256-verified B01/B02/B03 best checkpoints, predicts every test image with each model's frozen deterministic preprocessing, Mean-aggregates within Study, then takes `(B01+B02+B03)/3`. It writes a new immutable version directory with per-model image and Study probabilities, the complete 1,414-by-10 matrix, a 14,140-row `submission.csv`, the fusion config, validation check, and manifest. The history row has `generated` status until the user records an actual upload.

```powershell
.\.venv\Scripts\python.exe scripts\generate_e01_submission.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E01_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe scripts\verify_submission_bundle.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --version E01_20261016 --output-root outputs\submissions

.\.venv\Scripts\python.exe -m pytest -q tests\test_versioned_submission.py --basetemp outputs\audits\pytest_e01_tmp
```

The generator refuses an existing version directory. It never trains or uploads.



- `cxr.data.audit`: schema-driven audit of image-level CSV rows; counts subjects, studies, repeated images, missing files, and raw label values.
- `cxr.models.factory`: DenseNet-121 and ConvNeXt-Tiny factories with ten outputs and official torchvision ImageNet weight metadata.
- `cxr.engine.train` and `scripts/train.py`: masked BCE primitives and the GPU training loop, including optional gradient accumulation and Study-level validation.
- `cxr.engine.validate` and `cxr.engine.metrics`: align fused study predictions with study labels and calculate ten-class Study-level AUC/F1.
- `cxr.engine.analysis`: independently calculates per-class validation diagnostics and the number of studies with no class above threshold.
- `cxr.inference.aggregate`: deterministic mean of image probabilities by `(subject_id, study_id)`.
- `cxr.inference.predict`: deterministic image loading and inference with an explicit metadata mapping.
- `cxr.inference.submission`: explicit CSV row policies; no default policy is assumed.
- `cxr.utils.checkpoint`: portable `state_dict` save/load and 500 MB model-file check.
- `cxr.utils`: configuration and canonical class order.

Project directories and their roles are documented in [`docs/structure.md`](docs/structure.md). Raw data, model weights, and outputs are ignored by Git.

## Phase 7: fixed A02 fusion and M03 preparation

The A02 analysis reads SHA-verified frozen validation probabilities, reconstructs E01 from B01/B02/B03, aligns all 859 Studies and 228 patients, and evaluates only the three prespecified schemes. The M03 audit compares real training/validation images and tests a new ImageNet mean border fill. M02 already uses aspect-preserving fit and padding, so M03 changes the padding color while preserving geometry. These commands perform no training or test inference:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_a02_fusion.py --config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --output-dir outputs\analysis\a02_fixed_fusions_20261017 --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\prepare_m03.py --config configs\m03_convnext_tiny_meanpad_384.yaml --m02-config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --output-dir outputs\analysis\m03_preprocessing_20261017
.\.venv\Scripts\python.exe -m pytest tests\test_archive_dataset.py tests\test_config.py tests\test_model.py -q --basetemp outputs\analysis\m03_preprocessing_20261017\pytest_tmp
```

The output directories must be new when these scripts run. After explicit approval to train M03, use:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\m03_convnext_tiny_meanpad_384.yaml
```

## Phase 8A: E03 candidate and Study-balanced sampling preparation

E03 reuses the sealed E01 Study probability matrix and only runs the frozen M02 model on test images. The new version directory is immutable after generation. It contains the full 1,414-by-10 probability matrix, the 14,140-row all-class `submission.csv`, M02 image/Study probabilities, E01 source hashes, a manifest, and validation checks. Generation marks history as `generated`; upload and official score entry are manual.

```powershell
.\.venv\Scripts\python.exe scripts\generate_e03_submission.py --config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --version E03_20261017 --output-root outputs\submissions
.\.venv\Scripts\python.exe scripts\verify_e03_submission.py --config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --version E03_20261017 --output-root outputs\submissions
```

The S01 preparation script audits Study image counts, checks the frozen patient split, and compares its config against M02. It does not train. The sampler keeps 30,344 image draws and the existing number of optimizer steps per epoch while giving each Study one or two draws.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_study_sampling.py --config configs\s01_convnext_tiny_study_balanced_384.yaml --baseline-config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --output-dir outputs\analysis\phase8a_study_sampling_20261017
.\.venv\Scripts\python.exe -m pytest tests\test_study_sampling.py tests\test_archive_dataset.py -q --basetemp outputs\analysis\phase8a_study_sampling_20261017\pytest_tmp
```

Both output-producing commands require new directories when rerun. E03's four checkpoint files total about 188 MiB, below the 500 MB limit; official T4 end-to-end latency has not been measured.

S01 has completed and is frozen as a negative Study-balanced sampling result. Its command is retained for reproducibility; do not rerun it or change E03's `generated` history state:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\s01_convnext_tiny_study_balanced_384.yaml
```

S01 records actual per-Study draw counts in `sampling_epoch_XX.csv.gz` and rejects an epoch whose completed batches do not cover every Study at the expected one-or-two-draw frequency. M02's planned 1,897 accumulation groups per epoch are retained; AMP skips can change the count of successful optimizer updates.

After the S01 training process completes, the authorized postprocessing sequence independently reloads its best checkpoint, aligns all 859 validation Studies, compares S01/M02 and the fixed E03/E04 fusions with paired patient bootstrap, then freezes the S01 artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\s01_convnext_tiny_study_balanced_384.yaml --run-dir outputs\experiments\s01_convnext_tiny_study_balanced_384_20261017
.\.venv\Scripts\python.exe scripts\compare_s01.py --config configs\s01_convnext_tiny_study_balanced_384.yaml --run-dir outputs\experiments\s01_convnext_tiny_study_balanced_384_20261017 --output-dir outputs\experiments\s01_convnext_tiny_study_balanced_384_20261017\comparison --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\freeze_s01.py --config configs\s01_convnext_tiny_study_balanced_384.yaml --run-dir outputs\experiments\s01_convnext_tiny_study_balanced_384_20261017
```

For the current run, `scripts/finalize_s01_pipeline.py` monitors completion and executes that sequence once. It writes progress to `outputs/analysis/s01_pipeline_20261017/status.json`. The frozen M02, E01, and E03 files are read only during this sequence.

## Official inference environment

See [`docs/environment.md`](docs/environment.md) for the organizer's server specifications and local Python 3.12 setup. `scripts/check_environment.py --config configs/b01_densenet121_384.yaml --require-cuda` checks the local GPU software stack; use `--require-official-host` on the official T4 host. No Docker build or image upload is needed for the preliminary round.

Exact installation and server check commands (run in a clean Python 3.12 environment; choose the channel appropriate to the machine):

```powershell
& 'D:\users\anaconda3\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
Copy-Item docs/windows_sitecustomize.py.txt .venv/Lib/site-packages/sitecustomize.py
& '.\.venv\Scripts\python.exe' -m pip install -r requirements/inference.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements/train.txt
& '.\.venv\Scripts\python.exe' scripts/check_environment.py --config configs/b01_densenet121_384.yaml --require-cuda
& '.\.venv\Scripts\python.exe' scripts/check_environment.py --config configs/b01_densenet121_384.yaml --require-official-host
& '.\.venv\Scripts\python.exe' -m pytest -q
```

The `Copy-Item` step is only needed for this workstation's Anaconda-based Python, whose bundled older Visual C++ DLLs prevent direct torch import. Use the CUDA wheel channel; do not install CPU-only wheels for GPU work. `--require-official-host` will fail on this workstation because its GPU is not a T4; only the official T4 server can validate the 100 ms inference limit.

## Phase 9: E03 error analysis and L01 ASL preparation

E03 remains the best validation submission candidate and keeps `generated` status pending manual upload. [`outputs/analysis/phase9_e03_errors_20261017/report.md`](outputs/analysis/phase9_e03_errors_20261017/report.md) analyzes its frozen validation fusion, with detailed JSON, subgroup CSV, and a reconstructed 859×10 validation probability table. [`docs/phase9_asl_plan.md`](docs/phase9_asl_plan.md) describes the controlled BCE versus ASL experiment. Both scripts refuse to overwrite their output; choose a new output path for a rerun.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_phase9_errors.py --config outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated\config.yaml --output-dir outputs\analysis\phase9_e03_errors_20261017
.\.venv\Scripts\python.exe scripts\preflight_asl.py --config configs\l01_convnext_tiny_asl_384.yaml --output outputs\analysis\phase9_asl_preflight_20261017.json
.\.venv\Scripts\python.exe -m pytest tests\test_asymmetric_loss.py tests\test_phase9_metrics.py -q
```

At the end of Phase 9, L01 had not been trained. Its prepared command was:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\l01_convnext_tiny_asl_384.yaml --run-id l01_convnext_tiny_asl_384
```

## Phase 10: authorized L01 ASL control

The final preflight compares the elementwise loss and gradients against the ASL author formula, then measures BCE and ASL initial loss/gradient norms on the same pretrained ConvNeXt forward pass without an optimizer update. It writes `outputs/analysis/phase10_asl_final_audit_20261017.json`.

```powershell
.\.venv\Scripts\python.exe scripts\audit_asl_final.py --config configs\l01_convnext_tiny_asl_384.yaml
.\.venv\Scripts\python.exe scripts\train.py --config configs\l01_convnext_tiny_asl_384.yaml --run-id l01_convnext_tiny_asl_384_20261017_elevated
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated
.\.venv\Scripts\python.exe scripts\compare_l01.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated --output-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated\comparison --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\report_l01.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated
.\.venv\Scripts\python.exe scripts\freeze_l01.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated
```

On this workstation the first sandboxed L01 launch failed before the first training batch because Windows denied DataLoader multiprocessing pipe creation. That failed directory is retained for diagnosis. The authorized run uses the same config and an independent `_elevated` output directory; its training process runs with access to the DataLoader workers. Independent validation and comparison must wait for the run to finish. Do not rerun the training command while it is active.

The current background postprocessing watcher waits for `status.json` to become `complete`, then runs the three postprocessing commands above in order. It stops on any failure and records each step's stdout/stderr under `outputs/analysis/phase10_l01_pipeline_20261017`. It does not launch or resume training.

```powershell
.\.venv\Scripts\python.exe scripts\finalize_l01_pipeline.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated --output-dir outputs\analysis\phase10_l01_pipeline_20261017
.\.venv\Scripts\python.exe scripts\wait_l01_report.py --config configs\l01_convnext_tiny_asl_384.yaml --run-dir outputs\experiments\l01_convnext_tiny_asl_384_20261017_elevated --pipeline-dir outputs\analysis\phase10_l01_pipeline_20261017 --output outputs\analysis\phase10_l01_final_report_20261017.md
```

## Phase 11: R01 512×512 resolution control

R01 copies the frozen M02 runtime config and changes only `experiment.name` and `data.input_size` from 384 to 512. The data loader now accepts 384 or 512 and keeps the same aspect preserving resize, centered black padding, ImageNet normalization and training augmentation. The GPU preflight passed with micro-batch 8 and accumulation 2, so effective batch 16, 30,344 training images and 1,897 optimizer groups per epoch remain unchanged. No batch fallback was needed. The preflight JSON is `outputs/analysis/phase11_r01_preflight_b8_20261017.json`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_archive_dataset.py -q --basetemp outputs\analysis\phase11_r01_pytest_tmp
.\.venv\Scripts\python.exe scripts\preflight_r01.py --config configs\r01_convnext_tiny_512.yaml --batch-size 8 --output outputs\analysis\phase11_r01_preflight_b8_20261017.json
.\.venv\Scripts\python.exe scripts\train.py --config configs\r01_convnext_tiny_512.yaml --run-id r01_convnext_tiny_512_20261017_elevated
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated
.\.venv\Scripts\python.exe scripts\compare_r01.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated --output-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated\comparison --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\benchmark_r01.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated --output outputs\experiments\r01_convnext_tiny_512_20261017_elevated\inference_benchmark.json
.\.venv\Scripts\python.exe scripts\report_r01.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated
.\.venv\Scripts\python.exe scripts\freeze_r01.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated
```

The authorized R01 run completed at Epoch 10 through patience-based early stopping and is frozen. Its postprocessing watcher executed independent validation, fixed E06 comparison, local validation-image latency testing, report generation and freezing in order. The watcher does not train and refuses to overwrite its output. R01 had lower validation Macro-AUC than M02, and E06 had lower validation Macro-AUC than E03; see `outputs/experiments/r01_convnext_tiny_512_20261017_elevated/comparison/report.md`. Do not rerun the completed training command.

```powershell
.\.venv\Scripts\python.exe scripts\finalize_r01_pipeline.py --config configs\r01_convnext_tiny_512.yaml --run-dir outputs\experiments\r01_convnext_tiny_512_20261017_elevated --output-dir outputs\analysis\phase11_r01_pipeline_20261017
```

The latency script uses validation images and the local RTX 4060 Laptop GPU. It does not establish compliance with the official T4 100 ms limit. No hidden test inference or submission generation is part of R01.
# Phase 12: frozen M02 Study feature pooling

Run from the repository root with the installed Python 3.12 environment and a CUDA GPU. The commands only read the official Train/Val ZIP and the frozen M02 experiment; they create new output directories and fail if an output directory already exists.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_study_pool.py tests\test_study_feature_cache.py -q
.\.venv\Scripts\python.exe -u scripts\extract_m02_features.py --config configs\v00_v01_v02_m02_features.yaml
.\.venv\Scripts\python.exe -u scripts\train_study_heads.py --config configs\v00_v01_v02_m02_features.yaml
.\.venv\Scripts\python.exe scripts\report_phase12.py --config configs\v00_v01_v02_m02_features.yaml
.\.venv\Scripts\python.exe scripts\freeze_phase12.py --config configs\v00_v01_v02_m02_features.yaml
```

The feature cache is `outputs/features/m02_convnext_tiny_384_deterministic/`; the V00/V01/V02 runs and comparison are in `outputs/experiments/v00_v01_v02_m02_frozen_features/`. Cached training features use M02's deterministic validation preprocessing and exclude its random training augmentation. The aggregation heads receive official training labels only; validation labels are used for checkpoint selection and reporting. The 30 epoch head budget is fixed across all three architectures, with the best validation Study Macro-AUC checkpoint retained for each.

If image extraction finished but the final metadata check was interrupted before writing `manifest.json`, validate and finalize the existing cache without repeating GPU image inference:

```powershell
.\.venv\Scripts\python.exe scripts\finalize_m02_features.py --config configs\v00_v01_v02_m02_features.yaml
```

## Phase 13: V03 residual head and independent patient protocol

V03 uses the frozen Phase 12 M02 feature/logit cache. Its correction is zero by construction for a single-image Study. The following commands train only the small V03 head, audit the existing internal patient split, measure local validation Study latency, and freeze the new outputs. They do not retrain the backbone, run test inference, or create a submission.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_study_residual.py tests\test_study_pool.py tests\test_study_feature_cache.py -q --basetemp=outputs\analysis\phase13_pytest_tmp
.\.venv\Scripts\python.exe -u scripts\train_v03_residual.py --config configs\v03_m02_multiview_residual.yaml
.\.venv\Scripts\python.exe scripts\audit_phase13_internal_protocol.py --config configs\phase13_internal_retrain_protocol.yaml
.\.venv\Scripts\python.exe scripts\report_phase13.py --config configs\v03_m02_multiview_residual.yaml
.\.venv\Scripts\python.exe scripts\freeze_phase13.py --config configs\v03_m02_multiview_residual.yaml
.\.venv\Scripts\python.exe scripts\audit_phase13_ap.py --config configs\v03_m02_multiview_residual.yaml
.\.venv\Scripts\python.exe scripts\freeze_phase13_ap.py --config configs\v03_m02_multiview_residual.yaml
```

If the protocol audit is being deliberately rerun before freezing its outputs, add `--refresh` to its command. The V03 result is in `outputs/experiments/v03_m02_multiview_residual/`; the no-training patient protocol is in `outputs/analysis/phase13_internal_protocol/`. The former M02 checkpoint has seen every official Train patient, including the internal Holdout. The independent protocol requires a fresh ImageNet initialization trained only on internal Train patients, with the official validation set used for checkpoint selection and the internal Holdout evaluated once at the end. No internal retraining is authorized by these commands.

The supplemental `outputs/analysis/phase13_ap_tieaware/` recomputes AP after grouping equal probability scores. Use this table for final per-class AP reporting; the difference from earlier AP calculations does not affect AUC or patient bootstrap results.

## Phase 14: T01 Swin-Tiny backbone

T01 is controlled against the frozen M02 runtime configuration. The only training configuration changes are experiment identity and `model.name: swin_t`; the official torchvision `Swin_T_Weights.IMAGENET1K_V1` cache file is `swin_t-704ceda3.pth` (SHA-256 `704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b`). The local 8x2 preflight passed, so effective batch remains 16. No external medical training data is used.

```powershell
.\.venv\Scripts\python.exe scripts\check_model.py --config configs\t01_swin_tiny_384.yaml
.\.venv\Scripts\python.exe scripts\preflight_t01.py --config configs\t01_swin_tiny_384.yaml --output outputs\experiments\t01_swin_tiny_384_preflight_b8x2_20261018\preflight_report.json
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\t01_swin_tiny_384.yaml --run-id t01_swin_tiny_384_20261018
.\.venv\Scripts\python.exe -u scripts\monitor_t01_early_stopping.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018 --output outputs\analysis\t01_early_stopping_20261018.log
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018
.\.venv\Scripts\python.exe scripts\compare_t01.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018 --output-dir outputs\experiments\t01_swin_tiny_384_20261018\comparison
.\.venv\Scripts\python.exe scripts\benchmark_t01.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018 --output outputs\experiments\t01_swin_tiny_384_20261018\inference_benchmark.json
.\.venv\Scripts\python.exe scripts\report_t01.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018
.\.venv\Scripts\python.exe scripts\freeze_t01.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018
```

The read-only early-stopping monitor can run while training is active. It prints one line per completed epoch with `Early Stop Counter=x/6` and checks strict best-AUC updates, checkpoint decisions, and stop conditions against `metrics.jsonl`. The implementation calls the monitored metric `study_level_macro_auc`; this is the validation Study Macro-AUC requested as `val_study_macro_auc`. The independent validation and fixed E07 comparison run only after training reaches `complete`. To perform the final five commands automatically in that order, run the following watcher once against a new output directory:

```powershell
.\.venv\Scripts\python.exe -u scripts\finalize_t01_pipeline.py --config configs\t01_swin_tiny_384.yaml --run-dir outputs\experiments\t01_swin_tiny_384_20261018 --output-dir outputs\analysis\t01_pipeline_20261018
```

E07 is fixed as `0.5*E03 + 0.5*T01`, with 859 validation Studies aligned by Subject_id and Study_id. The local inference benchmark includes ZIP decoding and deterministic preprocessing on an RTX 4060 Laptop GPU. It does not establish official T4 latency compliance. These commands do not run hidden-test inference, create `submission.csv`, or upload to the leaderboard.

## Phase 15: independent-patient retrain preparation

The frozen patient split and complete, one-time Holdout protocol are described in [docs/phase15_patient_generalization.md](docs/phase15_patient_generalization.md). `prepare_phase15.py` checks the exact B01/M02 runtime-config differences, frozen index and ImageNet hashes, Train-only DataLoader mapping, ten-class support, output isolation and training-time forecast. The two `--preflight-steps 1` commands each execute one real optimizer group for GPU memory verification, then exit without validation or checkpoint. They are **not** formal retraining. Run each preflight only once under a new run ID; existing directories are rejected.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_phase15.py --config configs\phase15_patient_generalization.yaml
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\p15a_densenet121_internal_train_384.yaml --run-id p15a_internal_preflight_20261018 --preflight-steps 1
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\p15b_convnext_tiny_internal_train_384.yaml --run-id p15b_internal_preflight_20261018 --preflight-steps 1
.\.venv\Scripts\python.exe scripts\verify_phase15_gpu_preflight.py --config configs\phase15_patient_generalization.yaml --a-run-dir outputs\experiments\p15a_internal_preflight_20261018 --b-run-dir outputs\experiments\p15b_internal_preflight_20261018
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs\analysis\phase15_pytest_manual tests\test_internal_split.py tests\test_early_stopping.py
```

Phase 15 formal retraining was approved and P15A started on 2026-10-18. Each run starts from the cached original torchvision ImageNet file, verifies its SHA-256, updates weights using only 27,301 Internal Train images, and selects the best checkpoint on the official validation set. The Internal Holdout is evaluated once only after both independent best-checkpoint validations, SHA-256 manifests and the fixed 0.5/0.5 fusion are locked. Do not rerun the following commands against an occupied output directory.

```powershell
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\p15a_densenet121_internal_train_384.yaml --run-id p15a_densenet121_internal_train_384
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\p15b_convnext_tiny_internal_train_384.yaml --run-id p15b_convnext_tiny_internal_train_384
```

The sequential executor waits for the already running P15A process, validates and freezes it, then starts P15B. It validates and freezes P15B, checks the fixed official-validation fusion, writes `selection_lock.json`, and invokes the one-time Holdout evaluator. Its actual invocation uses P15A PID 56624 and the independent output directory below:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_phase15_pipeline.py --config configs\phase15_patient_generalization.yaml --p15a-pid 56624 --output-dir outputs\analysis\phase15_formal_20261018\pipeline
```

The executor invokes the following analysis commands in order; they are listed for reproducibility, **not** as instructions to rerun the locked Holdout. The Holdout evaluator rejects an existing output directory. No command reads the hidden test set or creates a submission.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\p15a_densenet121_internal_train_384.yaml --run-dir outputs\experiments\p15a_densenet121_internal_train_384
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\p15b_convnext_tiny_internal_train_384.yaml --run-dir outputs\experiments\p15b_convnext_tiny_internal_train_384
.\.venv\Scripts\python.exe scripts\evaluate_phase15_holdout.py --config configs\phase15_patient_generalization.yaml --lock outputs\analysis\phase15_formal_20261018\pipeline\selection_lock.json --output-dir outputs\analysis\phase15_formal_20261018\pipeline\holdout
.\.venv\Scripts\python.exe -m pytest -q tests\test_phase15_holdout_analysis.py tests\test_internal_split.py
```

## Phase 16: N01 EfficientNetV2-S with fixed EMA comparison

The approved N01 experiment and its evidence boundary are in [docs/phase16_n01.md](docs/phase16_n01.md). `prepare_phase16.py` checks the frozen Phase 15 hashes, E03's generated CSV/history and every N01 change from M02. E03 stays `generated` until its platform upload record and file identity can be verified. N01 starts from torchvision's official ImageNet checkpoint and uses only the official training split for gradient updates; Phase 15 Holdout is excluded from all N01 decisions.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_phase16.py --config configs\n01_efficientnet_v2_s_384.yaml --output-dir outputs\analysis\phase16_preflight_20261018
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\n01_efficientnet_v2_s_384.yaml --run-id n01_efficientnet_v2_s_384_preflight_20261018 --preflight-steps 5
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\n01_efficientnet_v2_s_384.yaml --run-id n01_efficientnet_v2_s_384_20261018
.\.venv\Scripts\python.exe -u scripts\finalize_n01_pipeline.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018 --output-dir outputs\analysis\n01_pipeline_20261018\steps
```

The N01 formal run and its sequential postprocessing watcher are active. The watcher calls these commands after training completes. Do not rerun them against their occupied output paths:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018 --checkpoint-kind raw
.\.venv\Scripts\python.exe scripts\analyze_validation.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018 --checkpoint-kind ema
.\.venv\Scripts\python.exe scripts\compare_n01.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018 --output-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018\comparison --bootstrap-replicates 2000 --seed 42
.\.venv\Scripts\python.exe scripts\benchmark_n01.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018 --output outputs\experiments\n01_efficientnet_v2_s_384_20261018\inference_benchmark.json
.\.venv\Scripts\python.exe scripts\freeze_n01.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_20261018
.\.venv\Scripts\python.exe -m pytest -q tests\test_n01_ema.py
```

## E07 versioned submission candidate

`E07 = 0.5*E03 + 0.5*T01` uses the sealed E03 1,414-Study probability matrix and fresh deterministic inference from the frozen T01 best checkpoint. Both models' test data remain unlabeled. The candidate is written to a new read-only version directory, includes all ten original class IDs for each Study (14,140 data rows), and is **not uploaded**. `verify_e07_submission.py` checks the source hashes, all image-to-Study Mean calculations, fusion formula, CSV content and history state independently.

```powershell
.\.venv\Scripts\python.exe -u scripts\generate_e07_submission.py --config outputs\experiments\t01_swin_tiny_384_20261018\config.yaml --version E07_20261018 --output-root outputs\submissions
.\.venv\Scripts\python.exe scripts\verify_e07_submission.py --config outputs\experiments\t01_swin_tiny_384_20261018\config.yaml --version E07_20261018 --output-root outputs\submissions
```

The first N01 formal attempt stopped at epoch 1, optimizer group 423 because a new finite-gradient assertion ran before the standard FP16 GradScaler overflow skip. Its failed directory and logs are retained. N01's training hyperparameters are unchanged; the AMP safety check now allows skipped updates, records their step indices, and stops only after 20 consecutive skips. A separate 500-step preflight must pass before a fresh formal run is launched under a new run ID.

```powershell
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\n01_efficientnet_v2_s_384.yaml --run-id n01_efficientnet_v2_s_384_amp_preflight_20261018 --preflight-steps 500
```

The 500-group preflight passed: one AMP skip at group 423, 499 successful optimizer/EMA updates, no sustained nonfinite gradients, peak PyTorch allocated memory 2,222 MiB. A new formal attempt and watcher were started using the exact same N01 YAML; the failed run and watcher logs remain in their original directories.

```powershell
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\n01_efficientnet_v2_s_384.yaml --run-id n01_efficientnet_v2_s_384_amp_retry_20261018
.\.venv\Scripts\python.exe -u scripts\finalize_n01_pipeline.py --config configs\n01_efficientnet_v2_s_384.yaml --run-dir outputs\experiments\n01_efficientnet_v2_s_384_amp_retry_20261018 --output-dir outputs\analysis\n01_pipeline_amp_retry_20261018\steps
```
# Phase 17 joint patient-fold pilot (D17-F1, C17-F1, S17-F1)

The fixed five-fold indices and the three Fold 1 training controls are described in [docs/phase17_joint_folds.md](docs/phase17_joint_folds.md). Run commands from the project root. The joint pipeline runs **only Fold 1** in D17, C17, S17 order, independently recomputes official validation and OOF probabilities, then freezes each experiment and analyzes the fixed `0.25 D17-F1 + 0.25 C17-F1 + 0.5 S17-F1` pilot. It never reads hidden-test labels or creates a submission.

```powershell
.\.venv\Scripts\python.exe scripts\create_phase17_folds.py --config configs\b01_densenet121_384_local.yaml --output-dir outputs\analysis\phase17_patient_folds --seed 42
.\.venv\Scripts\python.exe scripts\prepare_phase17_configs.py --config outputs\experiments\b01_densenet121_384_20261016\config.yaml --fold-dir outputs\analysis\phase17_patient_folds --output-dir configs
.\.venv\Scripts\python.exe scripts\prepare_phase17_joint_configs.py --config configs\f17_densenet121_fold1_384.yaml --fold-dir outputs\analysis\phase17_patient_folds --output-dir configs
.\.venv\Scripts\python.exe -m pytest -q tests\test_patient_folds.py tests\test_training_resume.py --basetemp outputs\phase17_pytest
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\f17_densenet121_fold1_384.yaml --run-id d17_f1_preflight_20261019 --preflight-steps 2
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\c17_fold1_384.yaml --run-id c17_f1_preflight_20261019 --preflight-steps 2
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\s17_fold1_384.yaml --run-id s17_f1_preflight_20261019 --preflight-steps 2
.\.venv\Scripts\python.exe scripts\preflight_phase17_checkpoint.py --config configs\f17_densenet121_fold1_384.yaml --output outputs\analysis\phase17_preflight\d17_checkpoint.json
.\.venv\Scripts\python.exe scripts\preflight_phase17_checkpoint.py --config configs\c17_fold1_384.yaml --output outputs\analysis\phase17_preflight\c17_checkpoint.json
.\.venv\Scripts\python.exe scripts\preflight_phase17_checkpoint.py --config configs\s17_fold1_384.yaml --output outputs\analysis\phase17_preflight\s17_checkpoint.json
.\.venv\Scripts\python.exe -u scripts\run_phase17_fold1.py --config configs\f17_densenet121_fold1_384.yaml --output-dir outputs\analysis\phase17_fold1_pipeline_20261019
```

The generated indices, config files, preflight outputs and formal run IDs are versioned by directory. Generation and training commands refuse existing output directories. On Linux/SSH, use the same relative fold-index path and set the archive path in a copied config to the mounted official ZIP location. `--resume` on `scripts/train.py` resumes a partially completed Fold 1 run from its last complete epoch, restoring optimizer, scheduler, GradScaler and RNG states.

To audit the frozen Fold 1–3 fixed fusion without running training or Test inference, run from the project root:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\analyze_phase17_three_fold_ensemble.py --config configs\f17_densenet121_fold3_384.yaml --output-dir outputs\analysis\phase17_three_fold_ensemble_recheck
```

The approved local Fold 4 sequence runs D17, C17, then S17 with their frozen configs. It resumes an interrupted model from its last complete epoch, freezes each model, and computes the fixed Fold 4 fusion and Fold 1–4 partial OOF. It does not start Fold 5 or create Test predictions:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -u scripts\run_phase17_fold4.py --config configs\f17_densenet121_fold4_384.yaml
```
