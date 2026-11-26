"""Independent Phase47 read-only model benchmark and preprocessing study.

Only writes into its own directory. The fixed 100 Val images are not an
official T4 cost sample. No Test input is read by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import io
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P44 = ROOT / "outputs/analysis/phase44_e08_acceleration"
P30 = ROOT / "outputs/analysis/phase30_deployment/candidate_deployment"
SAMPLE = ROOT / "outputs/analysis/phase29_deployment/benchmark_images"
sys.path[:0] = [str(P44 / "isolated_deps"), str(ROOT / "src")]

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, preprocess_image, resize_pad_image
from cxr.data.archive import image_member, read_split_rows
from cxr.engine.metrics import binary_auc


def paths():
    warm = sorted((SAMPLE / "warmup_5").glob("*.jpg"))
    measured = sorted((SAMPLE / "pictures_100").glob("*.jpg"))
    if len(warm) != 5 or len(measured) != 100:
        raise ValueError("Fixed Val 5/100-image timing set incomplete")
    return warm, measured


def write_csv(name, fields, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return {"mean_ms": float(statistics.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def model_backend(name):
    if name == "pytorch_fp16":
        sys.path.insert(0, str(P30))
        from predict import load_models, predict_image
        models = load_models("E08_EMA", device="cuda")
        return models, lambda path: predict_image(path, models, candidate="E08_EMA")
    sys.path.insert(0, str(P44))
    from predict import load_models, predict_image
    precision = "fp32" if name.endswith("fp32") else "fp16"
    models = load_models(provider="CUDAExecutionProvider", precision=precision)
    return models, lambda path: predict_image(path, models)


def benchmark(name):
    warm, measured = paths()
    models, infer = model_backend(name)
    for path in warm:
        infer(path)
        torch.cuda.synchronize()
    by_image = {p.name: [] for p in measured}
    out = HERE / f"{name}_cost_predictions.csv"
    with out.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "image", *[f"Prob_{i}" for i in range(1, 11)]])
        for round_id in range(3):
            for index, path in enumerate(measured, 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                p = infer(path)
                writer.writerow([round_id, path.name, *[format(float(v), ".17g") for v in p]])
                f.flush()
                torch.cuda.synchronize()
                by_image[path.name].append((time.perf_counter() - start) * 1000)
            print(name, "round", round_id + 1, "done", flush=True)
    per_image = [statistics.mean(v) for v in by_image.values()]
    record = {"candidate": name, "device": torch.cuda.get_device_name(), "provider":
              "CUDAExecutionProvider" if name.startswith("onnx") else "PyTorch CUDA AMP FP16",
              "batch": 1, "warmup": 5, "images": 100, "rounds": 3,
              "scope": "read image file through flushed result write; sync CUDA; model load and warmup excluded",
              "sample": "fixed Val pictures_100, not official organizer Test timing sample",
              **stats(per_image), "cost_predictions_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
              "t4_status": "T4_UNVERIFIED"}
    (HERE / f"{name}_benchmark.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


def preprocess_a(image, target):
    padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
    tensor = TF.to_tensor(padded)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=tensor.dtype).view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def preprocess_c(image, target):
    padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
    array = np.array(padded, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=tensor.dtype).view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)


def preprocess_d(image, target):
    # Preallocated output, but every image still receives its own padded pixels.
    padded = resize_pad_image(image, 384, preprocessing="black_pad_v1")
    target.copy_(TF.to_tensor(padded))
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=target.dtype).view(-1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=target.dtype).view(-1, 1, 1)
    return target.sub_(mean).div_(std)


def preprocessing_study():
    _, measured = paths()
    candidates = {"A_inplace_normalize": preprocess_a,
                  "C_numpy_contiguous": preprocess_c,
                  "D_preallocated_output": preprocess_d}
    try:
        import cv2
        cv2_version = cv2.__version__
    except ImportError:
        cv2_version = None
    rows = []
    for name, function in [("original", lambda image, target: preprocess_image(image, 384)), *candidates.items()]:
        elapsed = []
        max_diff = 0.0
        mean_diff = []
        exact = True
        for path in measured:
            with Image.open(path) as image:
                reference = preprocess_image(image, 384)
                buffer = torch.empty((3, 384, 384), dtype=torch.float32)
                start = time.perf_counter()
                actual = function(image, buffer)
                elapsed.append((time.perf_counter() - start) * 1000)
                if actual.shape != reference.shape or actual.dtype != reference.dtype:
                    raise ValueError("Tensor shape or dtype changed")
                diff = (actual-reference).abs()
                max_diff = max(max_diff, float(diff.max()))
                mean_diff.append(float(diff.mean()))
                exact &= bool(torch.equal(actual, reference))
        rows.append({"candidate": name, "images": len(measured), "mean_preprocess_ms": statistics.mean(elapsed),
                     "p50_preprocess_ms": np.percentile(elapsed, 50), "p95_preprocess_ms": np.percentile(elapsed, 95),
                     "max_abs_tensor_diff": max_diff, "mean_abs_tensor_diff": statistics.mean(mean_diff),
                     "bitwise_equal": exact, "opencv_available": cv2_version or "NO",
                     "full_val_status": "PENDING" if name != "original" and exact else ("REFERENCE" if name == "original" else "REJECT_DIFFERENT_TENSOR")})
    if cv2_version:
        # OpenCV bilinear interpolation is not assumed identical to Pillow.
        rows.append({"candidate": "B_opencv_resize", "images": 0, "mean_preprocess_ms": "N/A",
                     "p50_preprocess_ms": "N/A", "p95_preprocess_ms": "N/A", "max_abs_tensor_diff": "N/A",
                     "mean_abs_tensor_diff": "N/A", "bitwise_equal": False, "opencv_available": cv2_version,
                     "full_val_status": "BLOCKED_UNVERIFIED_INTERPOLATION"})
    else:
        rows.append({"candidate": "B_opencv_resize", "images": 0, "mean_preprocess_ms": "N/A",
                     "p50_preprocess_ms": "N/A", "p95_preprocess_ms": "N/A", "max_abs_tensor_diff": "N/A",
                     "mean_abs_tensor_diff": "N/A", "bitwise_equal": False, "opencv_available": "NO",
                     "full_val_status": "BLOCKED_NO_OPENCV"})
    write_csv("cpu_preprocessing_benchmarks.csv", list(rows[0]), rows)
    print(json.dumps(rows, default=float), flush=True)


def binding_probe():
    warm, measured = paths()
    graph = P44 / "onnx_models" / "B01_fp32.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA fallback")
    output_name = session.get_outputs()[0].name
    tensors = []
    for path in warm + measured:
        with Image.open(path) as image:
            tensors.append(preprocess_image(image, 384).unsqueeze(0).numpy())
    for value in tensors[:5]:
        session.run(None, {"image": value})
    rows = []
    for name in ("session_run", "io_binding_cpu_input_output"):
        times = []
        max_diff = 0.0
        for value in tensors[5:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if name == "session_run":
                result = session.run(None, {"image": value})[0]
            else:
                io = session.io_binding()
                io.bind_cpu_input("image", value)
                io.bind_output(output_name, "cpu")
                session.run_with_iobinding(io)
                result = io.copy_outputs_to_cpu()[0]
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            if name != "session_run":
                reference = session.run(None, {"image": value})[0]
                max_diff = max(max_diff, float(np.max(np.abs(result-reference))))
        rows.append({"candidate": name, "graph": "B01_fp32", "images": 100,
                     **stats(times), "max_abs_logit_diff": max_diff,
                     "scope": "one graph only; preprocessed tensor, excludes file read and Study fusion"})
    write_csv("onnx_graph_optimization.csv", list(rows[0]), rows)
    print(json.dumps(rows), flush=True)


def validate_a():
    """Bitwise tensor, full Val probability/AUC, and same-sample wall-clock audit."""
    phase45 = ROOT / "outputs/submissions/E08_ONNX_FP32_20261028"
    spec = importlib.util.spec_from_file_location("phase47_reference_predict", phase45 / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.preprocess_image
    def optimized(image, size=384, *, preprocessing="black_pad_v1"):
        if size != 384 or preprocessing != "black_pad_v1":
            raise ValueError("Frozen E08 preprocessing expected")
        return preprocess_a(image, None)
    module.preprocess_image = optimized
    models = module.load_models()
    archive = Path(r"F:\数据集\cxr_first.zip")
    rows = read_split_rows(archive, "val")
    if len(rows) != 1422:
        raise ValueError("Val image count differs")
    exact = True
    max_input_diff = 0.0
    grouped = {}
    with zipfile.ZipFile(archive) as z:
        for i, row in enumerate(rows, 1):
            with Image.open(io.BytesIO(z.read(image_member(row, "val")))) as image:
                ref = source(image, 384, preprocessing="black_pad_v1")
                candidate = optimized(image)
                delta = float((ref-candidate).abs().max())
                max_input_diff = max(max_input_diff, delta)
                exact &= bool(torch.equal(ref, candidate))
                p = module.from_image(image, models)
            grouped.setdefault((row["Subject_id"], row["Study_id"]), []).append(p)
            if i % 300 == 0 or i == len(rows):
                print(f"A full Val {i}/{len(rows)}", flush=True)
    if len(grouped) != 859 or not exact:
        raise ValueError("A tensor changed or Val coverage differs")
    actual = {key: np.mean(values, axis=0) for key, values in grouped.items()}
    reference = {}
    with (phase45 / "val_predictions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reference[row["Subject_id"], row["Study_id"]] = np.array(
                [float(row[f"Prob_{i}"]) for i in range(1, 11)])
    if set(actual) != set(reference):
        raise ValueError("Study IDs differ")
    max_probability_diff = max(float(np.max(np.abs(actual[k]-reference[k]))) for k in actual)
    if max_probability_diff > 1e-6:
        raise ValueError("A probabilities changed")
    labels = {}
    with (ROOT / "batch_assets/val_labels.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["Subject_id"], row["Study_id"]] = np.array(
                [int(row[f"Target_{i}"]) for i in range(1, 11)])
    if set(labels) != set(actual):
        raise ValueError("Val labels differ")
    order = sorted(actual)
    y = np.stack([labels[k] for k in order])
    p = np.stack([actual[k] for k in order])
    auc = [binary_auc(y[:, i], p[:, i]) for i in range(10)]
    per_class = [{"class_id": i+1, "auc": auc[i], "positive_studies": int(y[:,i].sum()),
                  "baseline_fp32_auc": auc[i], "delta": 0.0} for i in range(10)]
    write_csv("per_class_auc.csv", list(per_class[0]), per_class)
    warm, measured = paths()
    for path in warm:
        module.predict_image(path, models)
        torch.cuda.synchronize()
    elapsed = {path.name: [] for path in measured}
    for round_id in range(3):
        for path in measured:
            torch.cuda.synchronize()
            start = time.perf_counter()
            module.predict_image(path, models)
            torch.cuda.synchronize()
            elapsed[path.name].append((time.perf_counter()-start)*1000)
        print("A benchmark round", round_id+1, "done", flush=True)
    cost = stats([statistics.mean(v) for v in elapsed.values()])
    result = {"candidate": "A_inplace_normalize_ONNX_FP32", "images_val": len(rows),
              "studies_val": len(actual), "input_bitwise_equal": exact, "max_input_diff": max_input_diff,
              "max_val_probability_diff_vs_phase45_fp32": max_probability_diff,
              "val_macro_auc": float(np.mean(auc)), "patient_bootstrap_difference_ci95": [0, 0],
              "bootstrap_note": "Identical Study probabilities imply paired difference exactly zero in every valid resample",
              "benchmark": cost, "sample": "fixed 100 Val images; 5 warmup, 3 rounds; 4060 only",
              "t4_status": "T4_UNVERIFIED"}
    (HERE / "preprocess_a_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def paired_probe(kind):
    """Alternate two variants on each fixed image under one loaded model set."""
    sys.path.insert(0, str(P44))
    from predict import load_models
    models = load_models(provider="CUDAExecutionProvider", precision="fp32")
    warm, measured = paths()
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32).view(-1,1,1)
    std = torch.as_tensor(IMAGENET_STD, dtype=torch.float32).view(-1,1,1)
    def predict(path, variant):
        with Image.open(path) as image:
            if kind == "preprocess" and variant == "candidate":
                tensor = preprocess_a(image,None).unsqueeze(0).numpy()
            else:
                tensor = preprocess_image(image,384).unsqueeze(0).numpy()
        parts={}
        for name,session in models.items():
            if kind == "binding" and variant == "candidate":
                io = session.io_binding()
                io.bind_cpu_input("image",tensor)
                io.bind_output(session.get_outputs()[0].name,"cpu")
                session.run_with_iobinding(io)
                logits=io.copy_outputs_to_cpu()[0][0].astype(np.float64)
            else:
                logits=session.run(None,{"image":tensor})[0][0].astype(np.float64)
            parts[name]=1/(1+np.exp(-logits))
        e01=(parts['B01']+parts['B02']+parts['B03'])/3
        return .5*(.5*e01+.5*parts['M02'])+.5*parts['N01_EMA']
    for path in warm:
        predict(path,"baseline")
        predict(path,"candidate")
    elapsed={name:{p.name:[] for p in measured} for name in ("baseline","candidate")}
    max_diff=0.0
    with (HERE/f"paired_{kind}_predictions.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["variant","round","image",*[f"Prob_{i}" for i in range(1,11)]])
        for round_id in range(3):
            for i,path in enumerate(measured):
                results={}
                ordering=("baseline","candidate") if (round_id+i)%2==0 else ("candidate","baseline")
                for variant in ordering:
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    output=predict(path,variant)
                    writer.writerow([variant,round_id,path.name,*[format(float(x),'.17g') for x in output]])
                    f.flush()
                    torch.cuda.synchronize()
                    elapsed[variant][path.name].append((time.perf_counter()-start)*1000)
                    results[variant]=output
                max_diff=max(max_diff,float(np.max(np.abs(results['baseline']-results['candidate']))))
            print("paired",kind,"round",round_id+1,"done",flush=True)
    baseline=[statistics.mean(v) for v in elapsed['baseline'].values()]
    candidate=[statistics.mean(v) for v in elapsed['candidate'].values()]
    differences=[a-b for a,b in zip(baseline,candidate)]
    report={"kind":kind,"sample":"fixed 100 Val images; 5 warmup, 3 rounds; CUDA wall clock",
            "baseline":stats(baseline),"candidate":stats(candidate),
            "paired_mean_saving_ms":statistics.mean(differences),
            "paired_p95_saving_ms":float(np.percentile(differences,95)),
            "max_abs_probability_diff":max_diff,"t4_status":"T4_UNVERIFIED"}
    (HERE/f"paired_{kind}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


def stage_probe():
    sys.path.insert(0, str(P44))
    from predict import load_models, predict_image
    models = load_models(provider="CUDAExecutionProvider", precision="fp32")
    warm, measured = paths()
    for path in warm:
        predict_image(path, models)
    observations = {}
    for path in measured:
        stage = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = predict_image(path, models, stage)
        with io.StringIO() as sink:
            writer = csv.writer(sink)
            writer.writerow([format(float(x), ".17g") for x in value])
            sink.flush()
        torch.cuda.synchronize()
        stage["end_to_end_with_memory_csv"] = (time.perf_counter()-start)*1000
        for name, duration in stage.items():
            observations.setdefault(name, []).append(duration)
    rows = []
    for name, durations in observations.items():
        rows.append({"stage": name, **stats(durations), "images": len(measured),
                     "sample": "fixed 100 Val images, one instrumented round, RTX4060",
                     "timer": "wall clock; ORT forward includes implicit H2D and D2H, no isolated GPU event timing"})
    write_csv("latency_breakdown.csv", list(rows[0]), rows)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("baseline", "preprocessing", "binding", "validate_a", "paired_a", "paired_binding", "stages"))
    parser.add_argument("--candidate", choices=("pytorch_fp16", "onnx_fp32", "onnx_fp16"))
    args = parser.parse_args()
    if args.task == "baseline":
        if not args.candidate:
            parser.error("--candidate required")
        benchmark(args.candidate)
    elif args.task == "preprocessing":
        preprocessing_study()
    elif args.task == "binding":
        binding_probe()
    elif args.task == "validate_a":
        validate_a()
    elif args.task in ("paired_a", "paired_binding"):
        paired_probe("preprocess" if args.task == "paired_a" else "binding")
    else:
        stage_probe()
