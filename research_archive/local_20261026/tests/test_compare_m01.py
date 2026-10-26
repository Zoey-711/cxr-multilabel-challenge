import numpy as np

from scripts.compare_m01 import paired_patient_bootstrap, score


def test_identical_study_predictions_have_zero_paired_delta():
    targets = np.tile(np.asarray([0, 1] * 10, dtype=np.int8)[:, None], (1, 10))
    probabilities = np.where(targets == 1, 0.8, 0.2).astype(np.float64)
    scored = score(probabilities, targets)
    assert scored["macro_auc"] == 1.0
    assert scored["macro_f1_at_0_5"] == 1.0
    assert scored["zero_prediction_studies_at_0_5"] == 10

    names = ("B01", "Frozen_equal", "M01", "M01_B01_equal", "M01_Frozen_equal_half")
    schemes = {name: probabilities.copy() for name in names}
    subjects = [f"subject_{i // 2}" for i in range(len(targets))]
    bootstrap = paired_patient_bootstrap(schemes, targets, subjects, replicates=200, seed=42)
    assert bootstrap["patient_count"] == 10
    assert bootstrap["valid_replicates"] >= 100
    for references in bootstrap["delta_ci_95"].values():
        for interval in references.values():
            assert interval == {"lower": 0.0, "upper": 0.0}
