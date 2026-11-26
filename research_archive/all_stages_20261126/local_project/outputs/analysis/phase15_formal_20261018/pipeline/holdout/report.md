# Phase 15: independent-patient Holdout evaluation

Internal Holdout: 1106 patients, 1846 Studies, 3043 images.
These patients were excluded from both new training runs. Model selection used only the official 859-Study validation set. This Holdout was scored once after checkpoint and fusion choices were locked.

| Scheme | Holdout Macro-AUC | Patient bootstrap 95% CI | Macro-AP | Macro-F1@0.5 |
|---|---:|---:|---:|---:|
| A | 0.80576 | [0.79296, 0.81880] | 0.41949 | 0.29511 |
| B | 0.80510 | [0.79171, 0.81896] | 0.42706 | 0.28297 |
| fusion | 0.81388 | [0.80117, 0.82701] | 0.43617 | 0.29155 |

Bootstrap: 2000 valid paired patient draws; 0 invalid draws excluded.

- fusion_minus_A: +0.00812; 95% CI [+0.00499, +0.01125].
- fusion_minus_B: +0.00878; 95% CI [+0.00300, +0.01459].
- B_minus_A: -0.00066; 95% CI [-0.00874, +0.00760].

Ten-class AUC/AP, positive patient support, and single/multi-image Study results are in `result.json`. Subgroup comparisons are descriptive because class composition may differ.

## Official validation used for model selection

P15A Macro-AUC 0.78855; P15B Macro-AUC 0.79714; fixed fusion Macro-AUC 0.80007. These are selection-set results and must not be described as internal Holdout estimates.

## Existing leaderboard result

E01 previously received displayed official score 75.51. No Phase 15 model was run on hidden test data or submitted. This leaderboard score is distinct from both local official-validation and internal-Holdout metrics.
