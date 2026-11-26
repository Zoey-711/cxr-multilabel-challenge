# S01 Study-balanced sampling plan

M02 trained on 30344 images from 18513 Studies and 11055 patients.
Multi-image Studies: 10006 (54.0%) of Studies, but 72.0% of image-level training exposures.
Original Study exposures per epoch range from 1 to 7 images. Proposed S01 exposures: {1: 6682, 2: 11831}; epoch still has 30344 image draws and 1897 optimizer steps.

Here 1,897 is the number of planned accumulation groups. M02 had occasional FP16 GradScaler skips, so its **successful** optimizer updates varied from 1,895 to 1,897 by epoch. S01 preserves the planned group count and scheduler rule; exact successful updates cannot be guaranteed in advance and will be reported from each epoch log.

The sampler draws every Study once, samples distinct Studies for the remaining draws, and chooses one source image uniformly on each draw. It uses seed+epoch so resume is reproducible. Subjects with multiple Studies still contribute once per Study; patient balancing is a separate question.

Only algorithmic config change from frozen M02: `data.train_sampling: study_balanced_v1`. Run name is bookkeeping. BCE, ConvNeXt-Tiny, LR, augmentation, batch/accumulation, seed, 384px preprocessing, official validation and Mean aggregation remain the same.

Validation groups: one image vs two or more images per Study. Official val: {'single_image_studies': 391, 'multi_image_studies': 468}; internal val: {'single_image_studies': 835, 'multi_image_studies': 1011}. Report group counts, ten per-class AUCs, Macro-AUC only if each class has both signs, and paired patient bootstrap of S01-M02 within each group. Keep 0.5 F1 as secondary. The official 859-Study validation remains the model selection set.

Generalization check: reuse the frozen patient split index (9949/1106 patients; 16667/1846 Studies), with no Subject_id overlap. A future internal rerun must train both M02 control and S01 candidate only on internal_train; the frozen full-train M02 checkpoint cannot evaluate internal_val without leakage. No such rerun is authorized now.

Training is **not started**. Proposed command after approval: `.\.venv\Scripts\python.exe scripts\train.py --config configs\s01_convnext_tiny_study_balanced_384.yaml`.

The S01 training loop will write `sampling_epoch_XX.csv.gz` with the actual per-Study draws observed in completed batches, and will reject an epoch unless all 18,513 Studies are present with exactly 6,682 single draws and 11,831 double draws. Epoch logs also summarize draws originating from one-image versus multi-image Studies.
