"""Inspect ONNX operator placement on the fixed Val sample, without fitting parameters."""
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

import predict
import onnxruntime as ort

here = Path(__file__).resolve().parent
sample = next((predict.ROOT / "outputs/analysis/phase29_deployment/benchmark_images/pictures_100").glob("*.jpg"))
with Image.open(sample) as im:
    tensor = predict.preprocess_image(im, 384, preprocessing="black_pad_v1")
options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = str(here / "onnx_operator_trace")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(str(here / "onnx_models/h17_raw_fp32.onnx"),
                               sess_options=options, providers=["CUDAExecutionProvider"])
session.run(None, {"image": tensor.unsqueeze(0).numpy()})
trace = Path(session.end_profiling())
events = json.loads(trace.read_text())
by_provider = defaultdict(lambda: dict(nodes=0, duration_us=0))
by_op = defaultdict(lambda: dict(nodes=0, duration_us=0))
for event in events:
    args = event.get("args", {})
    if event.get("cat") != "Node":
        continue
    provider = args.get("provider", "UNKNOWN")
    op = args.get("op_name", "UNKNOWN")
    for table, key in ((by_provider, provider), (by_op, op)):
        table[key]["nodes"] += 1
        table[key]["duration_us"] += event.get("dur", 0)
result = dict(trace=str(trace), providers=dict(by_provider),
              top_ops=sorted((dict(op=k, **v) for k, v in by_op.items()),
                             key=lambda x: x["duration_us"], reverse=True)[:20],
              note="One profiled graph pass; operator timings are not end-to-end wall clock.")
(here / "onnx_operator_profile.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
