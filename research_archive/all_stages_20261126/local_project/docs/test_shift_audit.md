# Train / Val / Test label-free shift audit

Source: ZIP split CSV metadata and ten decoded images per split. No test labels were read or inferred.

## Measured distribution

| Measure | Train | Val | Test |
|---|---:|---:|---:|
| Patients | 11055 | 228 | 136 |
| Studies | 18513 | 859 | 1414 |
| Images | 30344 | 1422 | 2218 |
| Studies per patient median | 1 | 1.0 | 6.0 |
| Multi-image Studies (%) | 54.049 | 54.482 | 45.191 |
| Height median (px) | 3043.0 | 2960.0 | 2778.0 |
| Width median (px) | 2544.0 | 2544.0 | 2544.0 |
| Width/height median | 0.832 | 0.832 | 0.832 |
| Missing ZIP images | 0 | 0 | 0 |
| Invalid dimension metadata | 0 | 0 | 0 |

## ViewPosition

| Value | Train | Val | Test |
|---|---:|---:|---:|
| <blank> | 1202 | 39 | 139 |
| AP | 12152 | 584 | 1090 |
| LATERAL | 6554 | 284 | 370 |
| LL | 2831 | 151 | 217 |
| PA | 7604 | 364 | 402 |
| PA LLD | 1 | 0 | 0 |

## PatientOrientation

| Value | Train | Val | Test |
|---|---:|---:|---:|
| <blank> | 3307 | 135 | 369 |
| Erect | 23651 | 1088 | 1587 |
| Recumbent | 3386 | 199 | 262 |

## Images per Study

| Value | Train | Val | Test |
|---|---:|---:|---:|
| 1 | 8507 | 391 | 775 |
| 2 | 8369 | 380 | 496 |
| 3 | 1465 | 81 | 122 |
| 4 | 160 | 7 | 20 |
| 5 | 9 | 0 | 1 |
| 6 | 2 | 0 | 0 |
| 7 | 1 | 0 | 0 |

## Dimension quantiles

| Measure | Train p10 / median / p90 | Val | Test |
|---|---:|---:|---:|
| image_height_rows | 2021.0 / 3043.0 / 3056.0 | 2021.0 / 2960.0 / 3056.0 | 2021.0 / 2778.0 / 3056.0 |
| image_width_columns | 1769.0 / 2544.0 / 3056.0 | 1763.1 / 2544.0 / 3056.0 | 1760.0 / 2544.0 / 3056.0 |
| aspect_ratio_width_over_height | 0.832 / 0.832 / 1.201 | 0.832 / 0.832 / 1.201 | 0.832 / 0.832 / 1.201 |
| image_pixels | 3631512.0 / 7774464.0 / 7774464.0 | 3578965.1 / 7774464.0 / 7774464.0 | 3653903.8 / 7774464.0 / 7774464.0 |

## Confirmed findings

- Patient and Study overlaps: {'train_val': {'shared_subjects': 0, 'shared_studies': 0}, 'train_test': {'shared_subjects': 0, 'shared_studies': 0}, 'val_test': {'shared_subjects': 0, 'shared_studies': 0}}.
- AP images: Val 41.1% versus Test 49.1% (+8.1 percentage points); PA: Val 25.6% versus Test 18.1% (-7.5 points). Blank ViewPosition: Val 2.7% versus Test 6.3%.
- Multi-image Studies: Val 54.5% versus Test 45.2%. Test has 1414 Studies from 136 patients; Val has 859 from 228.
- Median image height: Val 2960px versus Test 2778px. Median width and width/height ratio are the same in both splits.
- E01 test inference and validation use the same dataset preprocessor and Study Mean aggregation. Training-only augmentation is absent from validation and test. The source code maps class ID 1 to output index 0 and writes IDs 1–10 in the CSV.
- E01 frozen CSV SHA-256 matches its manifest. Full image-to-Study Mean and E01 fusion were independently verified by `scripts/verify_submission_bundle.py`.
- The first ten decoded images in each split agree with Rows/Columns metadata: True.

## Hypotheses, not established causes

- Differences in view, size, and Study image counts may contribute to a validation-to-leaderboard gap, but no test labels or per-subgroup test scores are available to establish a cause.
- Test patients contribute more repeat Studies on average. Because the competition scores Studies, those repeated patients may influence the test metric differently from validation; the direction and magnitude of any effect cannot be determined without hidden labels.
- A 75.51 leaderboard display would equal 0.7551 only if the display uses Macro-AUC percentage units; then the difference from local E01 validation AUC 0.8026744715463039 would be -0.04757. The display scale has not been independently confirmed.
- The test set may have different disease prevalence or acquisition sites, but those properties cannot be determined from these metadata alone.

No threshold tuning, fusion-weight search, or hidden test labels were used.
