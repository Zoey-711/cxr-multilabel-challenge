from pathlib import Path

from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES, original_class_id


def test_baseline_config_loads_without_real_data():
    path = Path(__file__).resolve().parents[1] / "configs" / "b01_densenet121_384.yaml"
    config = load_config(path)
    assert config["data"]["train_manifest"] is None
    assert config["data"]["columns"]["study_id"] is None
    assert config["model"]["num_classes"] == len(CLASS_NAMES) == 10
    assert (original_class_id(0), original_class_id(9)) == (1, 10)


def test_package_imports():
    import cxr.data.audit
    import cxr.engine.train
    import cxr.engine.validate
    import cxr.inference.aggregate
    import cxr.inference.submission
    import cxr.models.factory

    assert cxr.models.factory.create_model is not None
