"""Phase45 E08 ONNX CUDA FP32 inference, independent of PyTorch checkpoints.

The five ONNX graphs are the only loaded model files. Original source checkpoint
hashes remain in the export manifest for provenance, but are not runtime assets.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE
PHASE44 = HERE
sys.path.insert(0, str(HERE / "src"))

import numpy as np
import onnxruntime as ort
from PIL import Image
from cxr.data.archive import image_member, read_split_rows
from cxr.data.dataset import preprocess_image

NAMES = ("B01", "B02", "B03", "M02", "N01_EMA")
PROBS = [f"Prob_{i}" for i in range(1, 11)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_models() -> dict:
    """Fail closed if CUDA EP is unavailable or a graph differs from Phase44."""
    ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    exported = json.loads((PHASE44 / "onnx_export_manifest.json").read_text(encoding="utf-8"))
    sessions = {}
    for name in NAMES:
        graph = PHASE44 / "onnx_models" / f"{name}_fp32.onnx"
        if sha(graph) != exported[name]["fp32_sha256"]:
            raise ValueError(f"ONNX graph SHA-256 mismatch: {name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(str(graph), sess_options=options, providers=["CUDAExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider fallback: {name}")
        sessions[name] = session
    return sessions


def from_image(image: Image.Image, models: dict) -> np.ndarray:
    tensor = preprocess_image(image, 384, preprocessing="black_pad_v1").unsqueeze(0).numpy()
    parts = {}
    for name in NAMES:
        logits = models[name].run(None, {"image": tensor})[0][0].astype(np.float64)
        parts[name] = 1 / (1 + np.exp(-logits))
    e01 = (parts["B01"] + parts["B02"] + parts["B03"]) / 3
    fused = .5 * (.5 * e01 + .5 * parts["M02"]) + .5 * parts["N01_EMA"]
    if fused.shape != (10,) or not np.isfinite(fused).all() or (fused < 0).any() or (fused > 1).any():
        raise ValueError("Invalid E08 probability")
    return fused


def predict_image(path: Path, models: dict) -> np.ndarray:
    with Image.open(path) as image:
        return from_image(image, models)


def validation(archive: Path, output: Path, models: dict) -> dict:
    rows = read_split_rows(archive, "val")
    if len(rows) != 1422:
        raise ValueError("Unexpected Val image count")
    grouped = {}
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            with Image.open(io.BytesIO(source.read(image_member(row, "val")))) as image:
                probability = from_image(image, models)
            grouped.setdefault((row["Subject_id"], row["Study_id"]), []).append(probability)
    if len(grouped) != 859:
        raise ValueError("Unexpected Val Study count")
    with output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject_id", "Study_id", *PROBS])
        for key, values in sorted(grouped.items()):
            writer.writerow([*key, *[format(float(x), ".17g") for x in np.mean(values, axis=0)]])
    return {"images": len(rows), "studies": len(grouped), "sha256": sha(output)}
