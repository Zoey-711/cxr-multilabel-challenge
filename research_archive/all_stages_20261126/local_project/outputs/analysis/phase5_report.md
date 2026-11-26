# Phase 5: M01 multiview and M02 preparation

All 1,422 saved image probability vectors align to the 859 official validation Studies and 228 patients. Reaggregated Mean probabilities exactly equal the frozen M01 Study predictions. The M01 checkpoint, config, and image prediction SHA-256 values match its frozen manifest after this analysis.

## Study-level aggregation

| Method | Macro-AUC | Delta vs Mean | Paired patient 95% CI for delta |
|---|---:|---:|---:|
| Mean | 0.79265 | 0 | reference |
| Max | 0.78722 | -0.00543 | [-0.00934, -0.00241] |
| 0.5 Mean + 0.5 Max | 0.79054 | -0.00211 | [-0.00430, -0.00045] |

The bootstrap resamples the same 228 patients for both methods, retains all their Studies, and excludes draws where any class lacks a positive or negative Study. All 2,000 draws were valid. The ten class AUCs are in `phase5_m01_multiview/per_class_auc.csv`.

Validation image counts per Study: 1 image 391 Studies, 2 images 380, 3 images 81, 4 images 7. Distinct ViewPosition counts per Study: 1 view 464, 2 views 393, 3 views 2. Image-level ViewPosition counts: AP 584, PA 364, LATERAL 284, LL 151, blank 39. The common Study combinations are AP 442, PA+LATERAL 193, PA+LL 122, and AP+LATERAL 57.

Mean Macro-AUC is 0.71883 for 391 single-image Studies and 0.83800 for 468 multi-image Studies. Max changes the multi-image score to 0.83186 and cannot change single-image predictions. Class frequencies and views differ between the groups; this performance gap does not isolate a causal multi-image effect.

## Generalization audit

M01 used no stochastic training augmentation. Its image-level loader repeats each Study target for each image. Multi-image Studies are 54.0% of training Studies but 72.0% of training image samples. Class 1 positive rate is 3.36% per Study versus 2.95% per image; class 8 is 24.35% versus 22.09%; No Finding is 36.10% versus 41.69%. This establishes a training objective weighting difference, though it does not prove which part caused the validation gap.

At threshold 0.5, class 1 (Enlarged Cardiomediastinum) has 24 positives, AUC 0.65526, 0 TP and 24 FN. Class 4 (Pneumonia) has 68 positives, AUC 0.71627, 0 TP, 1 FP and 68 FN. Class 8 (Lung Opacity) has 214 positives, AUC 0.71445, 9 TP, 11 FP and 205 FN. Their positive probability medians are 0.0522, 0.1447 and 0.2628, respectively. Detailed error rows and single/multi strata are in `phase5_m01_generalization/`.

The source loop divides the sum of masked BCE over each gradient-accumulation group by its actual valid label count. The last group has one 8-image microbatch and 80 valid labels. The scheduler advances once per successful optimizer update, and every logged epoch-end learning rate matches the successful update count. There were nine AMP skipped steps: epoch 2 one, epoch 3 one, epoch 4 two, and epochs 5–9 one each; epoch 1 had none. Training losses were finite and every epoch had successful updates. Logs do not include individual gradient values or the exact skipped batch locations, so they cannot prove every gradient was finite.

M01 uses `ImageOps.pad`, which fits complete rectangular images inside 384×384 and adds black padding. The earlier crop warning was corrected by visual inspection of actual preprocessed images in Phase 6. It is kept unchanged in M02 so the planned comparison isolates training augmentation.

## M02 and internal holdout

`configs/m02_convnext_tiny_384_aug.yaml` differs from the frozen M01 runtime config only by its experiment name and the new training augmentation. Each train image has independent 0.5 probabilities for (a) rotation within ±3° plus translation within ±2%, and (b) brightness and contrast changes within ±5%. No flip, random crop, elastic transform, cutout, external data or validation augmentation is used. Two real train and two real validation images passed the DataLoader smoke check with finite 2×3×384×384 tensors and ten targets. The separate preparation directory is `outputs/experiments/m02_convnext_tiny_384_aug_preparation_20261016/`; no formal M02 training has started.

The fixed seed-42 internal patient split uses only the official training ZIP: internal training has 9,949 patients / 16,667 Studies / 27,301 images; internal validation has 1,106 patients / 1,846 Studies / 3,043 images. Patient overlap is zero, including against the official validation subjects. All ten classes have positives and negatives in both parts. Indices and class counts are under `phase5_internal_split/`.

M01's local RTX 4060 Laptop peak GPU allocated/reserved memory was 1,932/1,992 MiB and passive board peak was 3,506 MiB. M02 is projected at 8–11 minutes per epoch; about 1.2–1.7 hours if it stops at nine epochs, or 4–5.5 hours at the full 30. These are estimates; M02 GPU peak and run duration have not been measured.

Full project unit suite: 25 passed, 2 PyTorch pin-memory deprecation warnings. No new training, test inference, or leaderboard submission was performed.
