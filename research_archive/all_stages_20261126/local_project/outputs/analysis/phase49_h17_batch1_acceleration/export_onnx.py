"""Export the frozen raw H17-02 inference state to independent ONNX files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"
sys.path[:0] = [str(ROOT / "outputs/analysis/phase44_e08_acceleration/isolated_deps"),
                str(PACKAGE), str(PACKAGE / "src")]

import onnx
import torch
from onnxconverter_common import float16
from predict import load_config, load_models


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    cfg = load_config(PACKAGE / "config/deployment_config.json")
    source = PACKAGE / cfg["model_file"]
    if sha(source) != cfg["model_sha256"]:
        raise ValueError("Frozen H17 source weight changed")
    model = load_models(cfg, device="cpu")["H17_DISTILL_02"]
    output = HERE / "onnx_models"
    output.mkdir(parents=True, exist_ok=True)
    fp32, fp16 = output / "h17_raw_fp32.onnx", output / "h17_raw_fp16.onnx"
    if fp32.exists() or fp16.exists():
        raise FileExistsError("Phase49 export already exists; never overwrite")
    with torch.inference_mode():
        torch.onnx.export(model.eval(), torch.zeros(1, 3, 384, 384), str(fp32),
                          input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(fp32))
    onnx.checker.check_model(graph)
    half = float16.convert_float_to_float16(graph, keep_io_types=True, disable_shape_infer=True)
    onnx.save(half, str(fp16))
    onnx.checker.check_model(str(fp16))
    record = dict(source_weight_sha256=sha(source), source_config_sha256=sha(PACKAGE / "config/deployment_config.json"),
                  fp32=dict(path=str(fp32), bytes=fp32.stat().st_size, sha256=sha(fp32)),
                  fp16=dict(path=str(fp16), bytes=fp16.stat().st_size, sha256=sha(fp16)),
                  input_shape=[1, 3, 384, 384], opset=17, source_parameters_unchanged=True)
    (HERE / "onnx_export_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k]["bytes"] for k in ("fp32", "fp16")}))


if __name__ == "__main__":
    main()
