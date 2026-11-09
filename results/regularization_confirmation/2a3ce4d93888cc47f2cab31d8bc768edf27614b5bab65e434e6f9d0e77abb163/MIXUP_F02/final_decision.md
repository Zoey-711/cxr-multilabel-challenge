# Fixed regularization confirmation

{
  "scope": "OUTER_FOLD2_HOLDOUT_EXCLUDED_EXPLORATORY",
  "challenger": "MIXUP",
  "base_auc": 0.8003662970612769,
  "challenger_auc": 0.8041487704848901,
  "delta": 0.003782473423613153,
  "ci95": [
    0.00037899333649172613,
    0.007008034830519254
  ],
  "valid_bootstraps": 2000,
  "invalid_bootstraps": 0,
  "invalid_policy": "discard replicates missing either label for any class; no resampling to reach 2000",
  "studies": 3388,
  "patients": 2008,
  "decision": "PROMISING_PILOT",
  "further_training_started": false
}

No official Val/Test selection. No automatic confirmation-fold launch. Historical folds are not a blind test.
