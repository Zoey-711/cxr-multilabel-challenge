# M01 generalization audit

M01 used no stochastic train image augmentation. Train and validation both used deterministic ImageOps.pad and ImageNet normalization.
The train loader sampled 30344 images from 18513 Studies. Multi-image Studies were 54.0% of Studies but 72.0% of image-level samples.
This gives each multi-image Study a weight proportional to its image count in the image-level BCE objective.
All 9 recorded epoch learning rates match actual successful optimizer-step counts; there were 9 AMP skips by epoch: {'1': 0, '2': 1, '3': 1, '4': 2, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1}.
All logged training losses are finite. Exact skipped batches and gradient values were not saved, so a stronger claim about every gradient is unavailable.
The last accumulation group has one 8-image microbatch; training divides its BCE sum by its actual 80 valid labels.

| Class | Studies + | AUC | TP | FP | FN | Positive p50 | Negative p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 24 | 0.65526 | 0 | 0 | 24 | 0.0522 | 0.0390 |
| 4 Pneumonia | 68 | 0.71627 | 0 | 1 | 68 | 0.1447 | 0.0741 |
| 8 Lung Opacity | 214 | 0.71445 | 9 | 11 | 205 | 0.2628 | 0.1705 |

Class-specific image-versus-Study sampling prevalence is in `train_sampling_by_class.csv`. View and image-count strata for focused errors are in `focus_class_errors.csv` and `error_cases.csv`.
The current preprocessing fits the complete image with black padding; it remains unchanged for the M01/M02 augmentation-only comparison.
