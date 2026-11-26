"""Frozen E08 five-model FP16 entry."""
from pathlib import Path
from frozen_predict import load_models as _load, predict_image as _predict
PRECISION = "PyTorch FP16 autocast"
def load_models():
    return _load("E08_EMA")
def predict_image(path: Path, models):
    return _predict(path, models, candidate="E08_EMA")
