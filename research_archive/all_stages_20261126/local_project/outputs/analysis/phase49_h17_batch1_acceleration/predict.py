"""Independent H17-02 Phase49 inference; frozen PyTorch package stays read only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"
sys.path[:0] = [str(ROOT / "outputs/analysis/phase44_e08_acceleration/isolated_deps"), str(PACKAGE / "src")]

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from cxr.data.archive import image_member, read_split_rows, split_archive_path
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, preprocess_image, resize_pad_image, study_targets_from_rows
from cxr.engine.metrics import binary_auc

spec = importlib.util.spec_from_file_location("frozen_h17_package_predict", PACKAGE / "predict.py")
frozen = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(frozen)

BACKENDS = ("torch_frozen", "torch_inplace", "onnx_fp32", "onnx_fp16", "onnx_fp32_iobinding", "onnx_fp16_iobinding")


def sha(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preprocess_inplace(image: Image.Image):
    padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
    return TF.normalize(TF.to_tensor(padded), IMAGENET_MEAN, IMAGENET_STD, inplace=True)


def load_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(name)
    config = frozen.load_config(PACKAGE / "config/deployment_config.json")
    if name.startswith("torch"):
        model = frozen.load_models(config)["H17_DISTILL_02"]
        return dict(name=name, model=model, config=config, providers=["PyTorch CUDA"])
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    model = HERE / "onnx_models" / f"h17_raw_{'fp16' if 'fp16' in name else 'fp32'}.onnx"
    manifest = json.loads((HERE / "onnx_export_manifest.json").read_text())
    kind = "fp16" if "fp16" in name else "fp32"
    if sha(model) != manifest[kind]["sha256"] or sha(PACKAGE / config["model_file"]) != manifest["source_weight_sha256"]:
        raise ValueError("ONNX or frozen source hash mismatch")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CUDAExecutionProvider"])
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX CUDA provider unavailable: {providers}")
    return dict(name=name, model=session, config=config, providers=providers, model_file=str(model))


def predict_tensor(tensor: torch.Tensor, backend: dict):
    name, model = backend["name"], backend["model"]
    if name.startswith("torch"):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(tensor.unsqueeze(0).to("cuda"))
            prob = torch.sigmoid(logits).squeeze(0).float().cpu().numpy().astype(np.float64)
    else:
        array = tensor.unsqueeze(0).contiguous().numpy()
        if "iobinding" in name:
            binding = model.io_binding()
            binding.bind_cpu_input("image", array)
            binding.bind_output("logits", "cpu")
            model.run_with_iobinding(binding)
            logits = binding.copy_outputs_to_cpu()[0][0].astype(np.float64)
        else:
            logits = model.run(None, {"image": array})[0][0].astype(np.float64)
        prob = 1 / (1 + np.exp(-logits))
    if prob.shape != (10,) or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Invalid ten-class output")
    return prob


def predict_image(image: str | Path | Image.Image, backend: dict):
    if isinstance(image, Image.Image):
        pixels = image
        tensor = preprocess_image(pixels, 384, preprocessing="black_pad_v1") if backend["name"] != "torch_inplace" else preprocess_inplace(pixels)
    else:
        with Image.open(image) as pixels:
            tensor = preprocess_image(pixels, 384, preprocessing="black_pad_v1") if backend["name"] != "torch_inplace" else preprocess_inplace(pixels)
    return predict_tensor(tensor, backend)


def validate_preprocessing(paths: list[Path]):
    max_diff = 0.; images = 0
    for path in paths:
        with Image.open(path) as im:
            original = preprocess_image(im, 384, preprocessing="black_pad_v1")
            candidate = preprocess_inplace(im)
        max_diff = max(max_diff, float((original - candidate).abs().max()))
        images += 1
    return dict(images=images, max_abs_tensor_diff=max_diff, exact=max_diff == 0.)


def validation(backend: dict, archive: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows = read_split_rows(archive, "val")
    if len(rows) != 1422:
        raise ValueError("Official Val image count changed")
    grouped = defaultdict(list)
    with zipfile.ZipFile(split_archive_path(archive, "val")) as source:
        for row in rows:
            with Image.open(io.BytesIO(source.read(image_member(row, "val")))) as image:
                grouped[(row["Subject_id"], row["Study_id"])].append(predict_image(image, backend))
    if len(grouped) != 859:
        raise ValueError("Official Val Study count changed")
    values = {key: np.mean(parts, axis=0) for key, parts in grouped.items()}
    targets = {(s.subject_id, s.study_id): s.labels for s in study_targets_from_rows(rows)}
    if set(targets) != set(values):
        raise ValueError("Val Study IDs/labels changed")
    keys = sorted(values)
    aucs = [binary_auc(np.array([targets[k][j] for k in keys], dtype=np.int8),
                       np.array([values[k][j] for k in keys])) for j in range(10)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)])
        for key in keys:
            writer.writerow([*key, *[format(float(x), ".17g") for x in values[key]]])
    return dict(backend=backend["name"], studies=len(keys), images=len(rows), macro_auc=float(np.mean(aucs)),
                per_class_auc=aucs, csv_sha256=sha(output), providers=backend["providers"])


def benchmark(backend: dict, measured: list[Path], warmup: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(output)
    for path in warmup:
        predict_image(path, backend)
        torch.cuda.synchronize()
    output.mkdir(parents=True)
    elapsed = {p.name: [] for p in measured}
    with (output / "cost_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "image"] + [f"Prob_{i}" for i in range(1, 11)])
        for round_id in range(1, 4):
            for path in measured:
                torch.cuda.synchronize()
                start = time.perf_counter()
                prob = predict_image(path, backend)
                writer.writerow([round_id, path.name, *[format(float(x), ".17g") for x in prob]])
                stream.flush()
                torch.cuda.synchronize()
                elapsed[path.name].append((time.perf_counter() - start) * 1000)
    per_image = [statistics.mean(x) for x in elapsed.values()]
    result = dict(backend=backend["name"], gpu=torch.cuda.get_device_name(), providers=backend["providers"],
                  sample="fixed 100 official Val single-image Studies", official_sample=False,
                  precision="FP16 autocast" if backend["name"].startswith("torch") else backend["name"],
                  batch=1, warmup=5, images=100, rounds=3,
                  scope="file read through final CSV write, CUDA synchronized; model load/warmup excluded",
                  mean_ms=float(np.mean(per_image)), p50_ms=float(np.percentile(per_image, 50)),
                  p95_ms=float(np.percentile(per_image, 95)), max_ms=float(np.max(per_image)),
                  predictions_sha256=sha(output / "cost_predictions.csv"), target_status="T4_UNVERIFIED")
    (output / "cost_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def stage_breakdown(backend: dict, paths: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(output)
    records = []
    for path in paths:
        torch.cuda.synchronize(); start = time.perf_counter(); raw = path.read_bytes()
        file_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            decode_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            tensor = preprocess_inplace(im) if backend["name"] == "torch_inplace" else preprocess_image(im, 384, preprocessing="black_pad_v1")
        preprocess_ms = (time.perf_counter() - start) * 1000
        # Separate H2D only for PyTorch; ORT session.run includes its own H2D.
        if backend["name"].startswith("torch"):
            start = time.perf_counter(); gpu_tensor = tensor.unsqueeze(0).to("cuda"); torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                prob = torch.sigmoid(backend["model"](gpu_tensor)).squeeze(0).float().cpu().numpy()
            torch.cuda.synchronize(); forward_ms = (time.perf_counter() - start) * 1000
        else:
            h2d_ms = None
            start = time.perf_counter(); prob = predict_tensor(tensor, backend)
            torch.cuda.synchronize(); forward_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        import io as _io
        sink = _io.StringIO(); csv.writer(sink).writerow([path.name, *prob.tolist()])
        write_ms = (time.perf_counter() - start) * 1000
        records.append(dict(image=path.name, file_read_ms=file_ms, decode_ms=decode_ms,
                            preprocess_ms=preprocess_ms, h2d_ms=h2d_ms,
                            forward_and_cpu_result_ms=forward_ms, write_result_ms=write_ms))
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return records
