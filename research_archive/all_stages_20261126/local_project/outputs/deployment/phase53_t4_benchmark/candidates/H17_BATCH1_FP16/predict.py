"""Frozen H17 batch=1 FP16 entry."""
from pathlib import Path
from frozen_predict import load_config, load_models as _load, predict_image as _predict
HERE = Path(__file__).resolve().parent
PRECISION = "PyTorch FP16 autocast"
def load_models():
    config = load_config(HERE / "config/deployment_config.json")
    return config, _load(config)
def predict_image(path: Path, loaded):
    config, models = loaded
    return _predict(path, models, config)
