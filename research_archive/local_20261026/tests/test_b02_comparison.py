import numpy as np

from scripts.compare_b01_b02 import paired_patient_bootstrap


def test_paired_patient_bootstrap_uses_same_subject_resample():
    subjects = [f"p{i}" for i in range(20)]
    targets = np.asarray([[1] * 10 if i % 2 else [0] * 10 for i in range(20)])
    b01 = np.where(targets == 1, 0.8, 0.2)
    b02 = b01.copy()
    result = paired_patient_bootstrap(b01, b02, targets, subjects, replicates=200, seed=42)
    assert result["patient_count"] == 20
    assert result["difference_ci_95_lower"] == result["difference_ci_95_upper"] == 0.0
