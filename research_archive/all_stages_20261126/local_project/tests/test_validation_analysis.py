from cxr.engine.analysis import analyze_validation
from cxr.engine.validate import StudyTarget
from cxr.inference.aggregate import ImagePrediction


def test_class_diagnostics_and_zero_prediction_study_count():
    predictions = [
        ImagePrediction("p1", "s1", "a", (0.7,) * 10),
        ImagePrediction("p1", "s1", "b", (0.9,) * 10),
        ImagePrediction("p2", "s2", "c", (0.1,) * 10),
    ]
    targets = [StudyTarget("p1", "s1", (1.0,) * 10), StudyTarget("p2", "s2", (0.0,) * 10)]
    analysis, studies = analyze_validation(predictions, targets)
    assert analysis["macro_auc"] == analysis["macro_f1"] == 1.0
    assert analysis["zero_predicted_class_studies"] == 1
    assert analysis["per_class"][0]["precision"] == analysis["per_class"][0]["recall"] == 1.0
    assert analysis["per_class"][0]["predicted_positive_rate"] == 0.5
    assert analysis["per_class"][0]["probability_quantiles"]["p50"] == 0.45
    assert studies[0]["Study_id"] == "s1" and studies[0]["image_count"] == 2
