import numpy as np

from scripts.compare_m01_m02 import measures, patient_bootstrap


def test_paired_patient_bootstrap_and_ten_class_scoring():
    targets = np.tile(np.array([[0] * 10, [1] * 10], dtype=int), (4, 1))
    probabilities = np.where(targets == 1, 0.8, 0.2).astype(float)
    subjects = [f"subject_{i}" for i in range(4) for _ in range(2)]
    scored = measures(probabilities, targets)
    assert scored["macro_auc"] == 1.0
    assert scored["macro_f1_at_0_5"] == 1.0
    assert len(scored["per_class"]) == 10
    result = patient_bootstrap(probabilities, probabilities, targets, subjects,
                               replicates=200, seed=42)
    assert result["patient_count"] == 4
    assert result["valid_replicates"] == 200
    assert result["ci_95_lower"] == result["ci_95_upper"] == 0.0
    assert len(result["per_class_difference_ci_95"]) == 10
