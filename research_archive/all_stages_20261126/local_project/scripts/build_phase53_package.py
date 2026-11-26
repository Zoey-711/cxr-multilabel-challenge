"""Build an isolated, hash-checked T4 benchmark artifact from frozen local assets."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/deployment/phase53_t4_benchmark"
CAND = OUT / "candidates"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if expected and sha(source) != expected:
        raise ValueError(f"Source SHA mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != sha(source):
            raise FileExistsError(f"Different file already present: {destination}")
    else:
        shutil.copy2(source, destination)
    return sha(destination)


def write(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = body.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing generated file differs: {path}")
    else:
        path.write_bytes(data)


def dependency_source(dest):
    source = ROOT / "src/cxr"
    modules = ["__init__.py", "data/__init__.py", "data/archive.py", "data/dataset.py",
               "data/sampling.py", "engine/__init__.py", "engine/validate.py",
               "inference/__init__.py", "inference/aggregate.py", "models/__init__.py",
               "models/factory.py", "utils/__init__.py", "utils/checkpoint.py"]
    for module in modules:
        copy(source / module, dest / "src/cxr" / module)


def wrapper(candidate, source, loader, predictor, precision):
    write(candidate / "predict.py", f'''"""Portable wrapper of the frozen candidate's inference implementation."""
from pathlib import Path
from frozen_predict import {loader} as _load, {predictor} as _predict
PRECISION = {precision!r}
def load_models():
    return _load()
def predict_image(path: Path, models):
    return _predict(path, models)
''')


def cost(candidate, name):
    write(candidate / "cost.py", f'''"""Official-style cost check for {name}; uses predict.py unchanged."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import predict
from benchmark_core import run
if __name__ == "__main__":
    run(predict, {name!r})
''')


def source_with_replacements(source, destination, replacements):
    text = Path(source).read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise ValueError(f"Expected source fragment absent: {old}")
        text = text.replace(old, new)
    write(destination, text)


def build():
    CAND.mkdir(parents=True, exist_ok=True)
    e08_manifest = json.loads((ROOT / "outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json").read_text())
    onnx_manifest = json.loads((ROOT / "outputs/analysis/phase44_e08_acceleration/onnx_export_manifest.json").read_text())
    h17_config_source = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"
    h17_onnx_source = ROOT / "outputs/analysis/phase49_h17_batch1_acceleration/deployment_candidate"
    phase45 = ROOT / "outputs/submissions/E08_ONNX_FP32_20261028"
    names = ("B01", "B02", "B03", "M02", "N01_EMA")
    requirements_onnx = "numpy==2.5.2\nPillow==12.3.0\nonnxruntime-gpu==1.23.2\ntorch==2.10.0\ntorchvision==0.25.0\n"
    requirements_torch = "numpy==2.5.2\nPillow==12.3.0\ntorch==2.10.0\ntorchvision==0.25.0\n"

    # A: independent ONNX FP32 E08 version with the corresponding submitted CSV.
    a = CAND / "E08_ONNX_FP32"
    source_with_replacements(phase45 / "predict.py", a / "frozen_predict.py", [
        ('ROOT = HERE.parents[2]', 'ROOT = HERE'),
        ('PHASE44 = ROOT / "outputs/analysis/phase44_e08_acceleration"', 'PHASE44 = HERE'),
        ('sys.path[:0] = [str(PHASE44 / "isolated_deps"), str(ROOT / "src")]',
         'sys.path.insert(0, str(HERE / "src"))'),
        ('options.intra_op_num_threads = 2',
         'options.intra_op_num_threads = 2\n        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")')])
    sanitized = {name: {key: onnx_manifest[name][key] for key in
                   ("source_sha256", "fp32_bytes", "fp32_sha256", "input_shape", "opset")}
                 for name in names}
    write(a / "onnx_export_manifest.json", json.dumps(sanitized, indent=2))
    for name in names:
        copy(ROOT / "outputs/analysis/phase44_e08_acceleration/onnx_models" / f"{name}_fp32.onnx",
             a / "onnx_models" / f"{name}_fp32.onnx", onnx_manifest[name]["fp32_sha256"])
    wrapper(a, None, "load_models", "predict_image", "ONNX CUDA FP32")
    copy(phase45 / "submission.csv", a / "reference_submission.csv",
         json.loads((phase45 / "sha256_manifest.json").read_text())["submission.csv"]["sha256"])
    dependency_source(a)
    write(a / "requirements.txt", requirements_onnx)
    cost(a, "E08_ONNX_FP32")

    # B: frozen H17 batch=1 FP16 implementation and exact submitted CSV.
    b = CAND / "H17_BATCH1_FP16"
    copy(h17_config_source / "predict.py", b / "frozen_predict.py")
    copy(h17_config_source / "config/deployment_config.json", b / "config/deployment_config.json")
    copy(h17_config_source / "model/H17_02.pth", b / "model/H17_02.pth",
         "35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69")
    write(b / "predict.py", '''"""Frozen H17 batch=1 FP16 entry."""
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
''')
    copy(ROOT / "outputs/submissions/H17_02_BATCH1_20261024/submission.csv",
         b / "reference_submission.csv", "00e093b43fecc115708568cc0e47c667863e6963ab1fa4d0147f8fbed5930aac")
    dependency_source(b)
    write(b / "requirements.txt", requirements_torch)
    cost(b, "H17_BATCH1_FP16")

    # C: original five PyTorch checkpoints, with the Phase30 FP16 path localized.
    c = CAND / "E08_PYTORCH_FP16"
    source_with_replacements(ROOT / "outputs/analysis/phase30_deployment/candidate_deployment/predict.py",
        c / "frozen_predict.py", [
        ('ROOT = Path(os.environ.get("CXR_PROJECT_ROOT", HERE.parents[3])).resolve()', 'ROOT = HERE'),
        ('ASSET_ROOT = Path(os.environ["CXR_DEPLOY_ASSET_ROOT"]).resolve() if os.environ.get("CXR_DEPLOY_ASSET_ROOT") else None',
         'ASSET_ROOT = HERE / "model"')])
    copy(ROOT / "outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json",
         c / "frozen_e08_manifest.json")
    for name in names:
        meta = e08_manifest["models"][name]
        copy(ROOT / "outputs/experiments" / meta["run_id"] / meta["checkpoint"],
             c / "model" / f"{name}.pt", meta["checkpoint_sha256"])
    write(c / "predict.py", '''"""Frozen E08 five-model FP16 entry."""
from pathlib import Path
from frozen_predict import load_models as _load, predict_image as _predict
PRECISION = "PyTorch FP16 autocast"
def load_models():
    return _load("E08_EMA")
def predict_image(path: Path, models):
    return _predict(path, models, candidate="E08_EMA")
''')
    # The historical FP16 CSV is numerically different from this batch=1 entry.
    # It remains frozen in the source project and is deliberately absent here.
    dependency_source(c)
    write(c / "requirements.txt", requirements_torch)
    cost(c, "E08_PYTORCH_FP16")

    # D: ONNX FP32 H17 candidate has no matching leaderboard submission.
    d = CAND / "H17_ONNX_FP32"
    source_with_replacements(h17_onnx_source / "predict.py", d / "frozen_predict.py", [
        ('options.intra_op_num_threads = 2',
         'options.intra_op_num_threads = 2\n    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")')])
    copy(h17_onnx_source / "config.json", d / "config.json")
    copy(h17_onnx_source / "h17_raw_fp32.onnx", d / "h17_raw_fp32.onnx",
         "2575fd4798e1639c959bceddd1447c8b38e1def1ade17056889b5fa227c60099")
    write(d / "predict.py", '''"""Independent H17 ONNX CUDA FP32 entry."""
from pathlib import Path
from frozen_predict import load_model as _load, predict_image as _predict
PRECISION = "ONNX CUDA FP32"
def load_models():
    return _load()
def predict_image(path: Path, model):
    return _predict(path, model)
''')
    dependency_source(d)
    write(d / "requirements.txt", requirements_onnx)
    cost(d, "H17_ONNX_FP32")

    for candidate in CAND.iterdir():
        if candidate.is_dir():
            files = {str(path.relative_to(candidate)).replace("\\", "/"):
                     {"sha256": sha(path), "bytes": path.stat().st_size}
                     for path in sorted(candidate.rglob("*")) if path.is_file() and path.name != "manifest_sha256.json"}
            model_files = [path for path in candidate.rglob("*") if path.suffix.lower() in {".pt", ".pth", ".onnx"}]
            necessary = [path for path in candidate.rglob("*") if path.suffix == ".json" and path.name != "manifest_sha256.json"]
            deployed_bytes = sum(path.stat().st_size for path in model_files + necessary)
            write(candidate / "manifest_sha256.json", json.dumps({"candidate": candidate.name,
                "files": files, "actual_model_and_config_bytes": deployed_bytes,
                "under_500000000_bytes": deployed_bytes <= 500000000,
                "t4_latency": "T4_UNVERIFIED"}, indent=2))
    print(json.dumps({p.name: json.loads((p / "manifest_sha256.json").read_text())["actual_model_and_config_bytes"]
                      for p in CAND.iterdir() if p.is_dir()}))


if __name__ == "__main__":
    build()
