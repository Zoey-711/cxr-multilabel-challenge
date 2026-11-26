import numpy as np

from cxr.engine.b01_analysis import binary_auc, patient_bootstrap_auc, threshold_analysis


def test_threshold_metrics_and_no_positive_prediction_count():
    labels = np.asarray([[1] * 10, [0] * 10])
    probabilities = np.asarray([[0.45] * 10, [0.10] * 10])
    summaries, classes = threshold_analysis(probabilities, labels)
    assert [row["zero_predicted_class_studies"] for row in summaries] == [1, 1, 1, 2]
    assert summaries[0]["macro_f1"] == 1.0
    assert summaries[-1]["macro_f1"] == 0.0
    assert classes[0]["positive_studies"] == classes[0]["negative_studies"] == 1


def test_patient_bootstrap_excludes_one_class_resamples_without_imputing_auc():
    subjects = ["patient_0", "patient_0"] + [f"patient_{i}" for i in range(1, 10)]
    targets = np.asarray([[1] * 10] * 2 + [[0] * 10] * 9)
    probabilities = np.where(targets == 1, 0.9, 0.1)
    assert binary_auc(np.zeros(3, dtype=int), np.arange(3)) is None
    result = patient_bootstrap_auc(probabilities, targets, subjects, replicates=300, seed=42)
    assert result["patient_count"] == 10
    assert result["invalid_replicates_excluded"] > 0
    assert result["valid_replicates"] + result["invalid_replicates_excluded"] == 300
    assert result["point_macro_auc"] == result["ci_95_percentile_lower"] == result["ci_95_percentile_upper"] == 1.0
