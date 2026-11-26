"""Phase56 E08 original weights: CUDA Graph safe experiment."""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import tensorrt as trt

from frozen_predict import (
    load_model_state,
    preprocess_image,
)


from PIL import ImageOps
from torchvision.transforms import functional as TF
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from cxr.prediction_consistency import sha256

_original_preprocess_image = preprocess_image

def gray_first_preprocess(image, size=384, *, preprocessing="black_pad_v1"):
    if image.mode != "L" or preprocessing != "black_pad_v1":
        return _original_preprocess_image(
            image, size, preprocessing=preprocessing
        )

    if size not in (384, 512):
        raise ValueError("Unsupported image size")

    padded = ImageOps.pad(
        image,
        (size, size),
        method=Image.Resampling.BILINEAR,
        color=0,
    )

    return TF.normalize(
        TF.to_tensor(padded.convert("RGB")),
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

preprocess_image = gray_first_preprocess

PRECISION = "PyTorch FP16 autocast + CUDA Graph"
NAMES = ("B01", "B02", "B03", "M02", "N01_EMA", "S17_F3")
GRAPH_NAMES = ("B01", "B03", "N01_EMA", "S17_F3")


def load_models():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "manifest_sha256.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        source = root / relative
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Frozen deployment asset mismatch: {relative}")
    model_dir = Path(__file__).resolve().parent / "model"
    models = {
        name: load_model_state(
            model_dir / f"{name}.pt",
            device="cuda",
        )
        for name in GRAPH_NAMES
    }

    assert set(models) == {"B01", "B03", "N01_EMA", "S17_F3"}
    print("PyTorch models:", sorted(models), flush=True)

    static_x = torch.empty(
        (1, 3, 384, 384),
        dtype=torch.float32,
        device="cuda",
    )

    def forward(name):
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                return models[name](static_x)

    # Warm up on a separate stream before capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(warmup_stream):
        for name in GRAPH_NAMES:
            for _ in range(20):
                forward(name)

    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graphs = {}
    graph_outputs = {}

    for name in GRAPH_NAMES:
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            logits = forward(name)

        graphs[name] = graph
        graph_outputs[name] = logits

        print(f"{name}: CUDA Graph captured", flush=True)

    torch.cuda.synchronize()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engines = {}

    for name in ("B02", "M02"):
        engine_dir = Path(__file__).resolve().parent.parent / "tensorrt_engines"
        path = engine_dir / f"{name}_fp32.engine"

        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"{name}: TensorRT engine加载失败")

        ctx = engine.create_execution_context()
        if ctx is None:
            raise RuntimeError(f"{name}: TensorRT context创建失败")

        if tuple(ctx.get_tensor_shape("input")) != (1, 3, 384, 384):
            raise RuntimeError(f"{name}: 输入shape错误")

        if tuple(ctx.get_tensor_shape("logits")) != (1, 10):
            raise RuntimeError(f"{name}: 输出shape错误")

        output = torch.empty((1, 10), device="cuda", dtype=torch.float32)

        if not ctx.set_tensor_address("logits", output.data_ptr()):
            raise RuntimeError(f"{name}: 输出绑定失败")

        engines[name] = (engine, ctx, output)
        print(f"{name}: TensorRT loaded", flush=True)

    return {
        "models": models,
        "static_x": static_x,
        "graphs": graphs,
        "graph_outputs": graph_outputs,
        "trt_runtime": runtime,
        "trt_engines": engines,
    }


def predict_image(path: Path, state):
    with Image.open(path) as source:
        tensor = preprocess_image(
            source,
            384,
            preprocessing="black_pad_v1",
        )

    x = state["static_x"]
    models = state["models"]

    probabilities = {}

    with torch.inference_mode():
        x.copy_(tensor.unsqueeze(0), non_blocking=False)

        for name in NAMES:
            if name in state["graphs"]:
                state["graphs"][name].replay()
                logits = state["graph_outputs"][name]
            elif name in state["trt_engines"]:
                engine, ctx, output = state["trt_engines"][name]

                if not ctx.set_tensor_address("input", x.data_ptr()):
                    raise RuntimeError(f"{name}: 输入绑定失败")

                stream = torch.cuda.current_stream().cuda_stream

                if not ctx.execute_async_v3(stream):
                    raise RuntimeError(f"{name}: TensorRT执行失败")

                logits = output
            else:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = models[name](x)

            probabilities[name] = (
                torch.sigmoid(logits)
                .squeeze(0)
                .float()
            )

        stacked = torch.stack(
            [probabilities[n] for n in NAMES]
        )
        values = stacked.cpu().numpy().astype(np.float64)

    probs = dict(zip(NAMES, values))

    # Original E08 ensemble arithmetic, unchanged.
    e01 = (
        probs["B01"]
        + probs["B02"]
        + probs["B03"]
    ) / 3.0

    e03 = 0.5 * e01 + 0.5 * probs["M02"]
    e08 = 0.5 * e03 + 0.5 * probs["N01_EMA"]
    result = 0.75 * e08 + 0.25 * probs["S17_F3"]

    if (
        result.shape != (10,)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or np.any(result > 1)
    ):
        raise RuntimeError("Invalid E08 prediction")

    return result


_original_predict_image = predict_image

def predict_image(path: Path, state):
    stream = state.setdefault("infer_stream", torch.cuda.Stream())

    caller = torch.cuda.current_stream()
    stream.wait_stream(caller)

    with torch.cuda.stream(stream):
        result = _original_predict_image(path, state)

    caller.wait_stream(stream)

    return result
