"""Portable wrapper of the frozen candidate's inference implementation."""
from pathlib import Path
from frozen_predict import load_models as _load, predict_image as _predict
PRECISION = 'ONNX CUDA FP32'
def load_models():
    return _load()
def predict_image(path: Path, models):
    return _predict(path, models)
