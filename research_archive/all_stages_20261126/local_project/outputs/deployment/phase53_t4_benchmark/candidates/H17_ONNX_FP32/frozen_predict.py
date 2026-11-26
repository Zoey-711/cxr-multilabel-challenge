"""Phase49 H17 ONNX CUDA FP32 candidate; independent of the frozen FP16 package."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

HERE = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    path = HERE / cfg["model_file"]
    if not path.resolve().is_relative_to(HERE.resolve()) or sha256(path) != cfg["model_sha256"]:
        raise ValueError("Deployment model path/hash mismatch")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA execution unavailable: {session.get_providers()}")
    return session


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    padded = ImageOps.pad(rgb, (384, 384), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    tensor = TF.normalize(TF.to_tensor(padded), MEAN, STD)
    return tensor.unsqueeze(0).contiguous().numpy()


def predict_image(image: str | Path | Image.Image, model) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = preprocess(image)
    else:
        with Image.open(image) as opened:
            array = preprocess(opened)
    logits = model.run(None, {"image": array})[0][0].astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if probs.shape != (10,) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid ten-class probabilities")
    return probs


def predict_studies(image_rows: list[tuple[str, str, Path]], model):
    grouped = defaultdict(list)
    for subject, study, path in image_rows:
        grouped[(subject, study)].append(predict_image(path, model))
    return {key: np.mean(parts, axis=0) for key, parts in grouped.items()}


def write_submission(studies: dict, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if len(studies) != 1414:
        raise ValueError(f"Expected 1414 official Test Studies, got {len(studies)}")
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability"])
        for (subject, study), probs in sorted(studies.items()):
            if len(probs) != 10 or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
                raise ValueError("Invalid Study probabilities")
            for j, value in enumerate(probs, 1):
                writer.writerow([subject, study, j, format(float(value), ".17g")])
