import pytest

from cxr.engine.metrics import binary_auc, study_multilabel_metrics
from cxr.engine.validate import StudyTarget
from cxr.inference.aggregate import ImagePrediction


def test_binary_auc_ties_and_order():
    assert binary_auc([0, 1], [0.2, 0.8]) == 1.0
    assert binary_auc([0, 1], [0.8, 0.2]) == 0.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        binary_auc([1, 1], [0.2, 0.8])


def test_study_metrics_fuse_images_before_scoring():
    predictions = [
        ImagePrediction("p1", "s1", "a", (0.8,) * 10),
        ImagePrediction("p1", "s1", "b", (0.6,) * 10),
        ImagePrediction("p2", "s2", "c", (0.1,) * 10),
    ]
    targets = [StudyTarget("p1", "s1", (1.0,) * 10), StudyTarget("p2", "s2", (0.0,) * 10)]
    result = study_multilabel_metrics(predictions, targets)
    assert result["study_count"] == 2
    assert result["macro_auc"] == result["macro_f1"] == 1.0
    assert result["per_class"][0]["positive_studies"] == 1
