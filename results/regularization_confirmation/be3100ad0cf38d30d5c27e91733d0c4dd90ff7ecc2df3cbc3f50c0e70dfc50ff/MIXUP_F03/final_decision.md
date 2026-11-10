# Fixed regularization confirmation

{
  "scope": "OUTER_FOLD3_HOLDOUT_EXCLUDED_EXPLORATORY",
  "challenger": "MIXUP",
  "base_auc": 0.7998572774795281,
  "challenger_auc": 0.8029566019814313,
  "delta": 0.0030993245019032,
  "ci95": [
    -0.0004950200796817611,
    0.0064504564867938485
  ],
  "valid_bootstraps": 2000,
  "invalid_bootstraps": 0,
  "invalid_policy": "discard replicates missing either label for any class; no resampling to reach 2000",
  "studies": 3360,
  "patients": 1992,
  "decision": "PROMISING_PILOT",
  "further_training_started": false
}

No official Val/Test selection. No automatic confirmation-fold launch. Historical folds are not a blind test.
