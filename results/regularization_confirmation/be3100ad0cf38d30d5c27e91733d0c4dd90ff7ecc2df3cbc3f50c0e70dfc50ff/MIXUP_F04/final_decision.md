# Fixed regularization confirmation

{
  "scope": "OUTER_FOLD4_HOLDOUT_EXCLUDED_EXPLORATORY",
  "challenger": "MIXUP",
  "base_auc": 0.8034703682694712,
  "challenger_auc": 0.802555410242135,
  "delta": -0.0009149580273362101,
  "ci95": [
    -0.00399057221441209,
    0.0019146381490017673
  ],
  "valid_bootstraps": 2000,
  "invalid_bootstraps": 0,
  "invalid_policy": "discard replicates missing either label for any class; no resampling to reach 2000",
  "studies": 3265,
  "patients": 1977,
  "decision": "COMPLETED_NO_GAIN",
  "further_training_started": false
}

No official Val/Test selection. No automatic confirmation-fold launch. Historical folds are not a blind test.
