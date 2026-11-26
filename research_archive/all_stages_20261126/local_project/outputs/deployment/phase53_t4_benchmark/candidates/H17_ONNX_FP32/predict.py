"""Independent H17 ONNX CUDA FP32 entry."""
from pathlib import Path
from frozen_predict import load_model as _load, predict_image as _predict
PRECISION = "ONNX CUDA FP32"
def load_models():
    return _load()
def predict_image(path: Path, model):
    return _predict(path, model)
