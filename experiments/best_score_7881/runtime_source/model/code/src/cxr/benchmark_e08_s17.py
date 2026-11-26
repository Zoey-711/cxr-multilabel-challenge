"""Paired, official-style RTX 4060 reference benchmark for E08 + local S17-F3."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from cxr.data.dataset import preprocess_image
from cxr.utils.checkpoint import load_model_state


class CombinedEntry:
    PRECISION = "PyTorch FP16 autocast, batch=1"

    def __init__(self, e08_module, s17_checkpoint, s17_sha256):
        self.e08_module = e08_module
        self.s17_checkpoint = s17_checkpoint
        self.s17_sha256 = s17_sha256

    def load_models(self):
        with self.s17_checkpoint.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != self.s17_sha256:
            raise ValueError("S17-F3 checkpoint SHA-256 mismatch")
        models = self.e08_module.load_models()
        models["S17_F3"] = load_model_state(self.s17_checkpoint, device="cuda")
        return models

    def predict_image(self, path, models):
        # Same RGB conversion, black_pad_v1 transform and FP16 forward semantics
        # as the frozen E08 entry, with one shared image decode and tensor.
        with Image.open(path) as image:
            tensor = preprocess_image(image, 384, preprocessing="black_pad_v1")
        tensor = tensor.unsqueeze(0).to("cuda")
        p = {}
        with torch.inference_mode():
            for name, model in models.items():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p[name] = torch.sigmoid(model(tensor)).squeeze(0).float().cpu().numpy().astype(np.float64)
        e01 = (p["B01"] + p["B02"] + p["B03"]) / 3.0
        e03 = 0.5 * e01 + 0.5 * p["M02"]
        e08 = 0.5 * e03 + 0.5 * p["N01_EMA"]
        combined = 0.75 * e08 + 0.25 * p["S17_F3"]
        if combined.shape != (10,) or not np.isfinite(combined).all() or np.any((combined < 0) | (combined > 1)):
            raise ValueError("Invalid fused probabilities")
        return combined


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    candidate = Path(cfg["candidate"])
    package = Path(cfg["e08_package"])
    samples = Path(cfg["val_samples"])
    checkpoint = Path(cfg["s17_checkpoint"])
    output = Path(cfg["output"])
    expected_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    if expected_manifest["source_files"]["S17_checkpoint_sha256"] != cfg["s17_sha256"]:
        raise ValueError("Candidate manifest checkpoint differs")
    if cfg["candidate_submission_sha256"] != expected_manifest["files"]["submission.csv"]["sha256"]:
        raise ValueError("Candidate submission differs")
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    # Existing predict.py imports frozen_predict using its own candidate folder.
    sys.path.insert(0, str(package))
    e08 = load_module(package / "predict.py", "benchmark_e08_frozen_entry")
    core = load_module(package.parents[1] / "benchmark_core.py", "benchmark_core")
    combined = CombinedEntry(e08, checkpoint, cfg["s17_sha256"])
    output.mkdir(parents=True)
    for name, entry in (("e08_baseline", e08), ("e08_s17_combined", combined)):
        params = ["benchmark_core", "--pictures-100", str(samples / "pictures_100"),
                  "--pictures-5", str(samples / "pictures_5"), "--output", str(output / name)]
        sys.argv = params
        core.run(entry, name.upper())
    a = json.loads((output / "e08_baseline/report.json").read_text(encoding="utf-8"))
    b = json.loads((output / "e08_s17_combined/report.json").read_text(encoding="utf-8"))
    result = {"baseline_mean_ms": a["per_image_then_overall_mean_ms"],
              "combined_mean_ms": b["per_image_then_overall_mean_ms"],
              "increment_ms": b["per_image_then_overall_mean_ms"] - a["per_image_then_overall_mean_ms"],
              "gpu": b["gpu"], "precision": b["precision"], "batch": 1,
              "warmup": 5, "timed_images": 100, "rounds": 3,
              "sample_kind": "Phase53 fixed official Val images, not official T4 measurement",
              "submission_consistency_status": "UNVERIFIED_FOR_BATCH1; frozen candidate Submission used E08 cached Test Study probabilities and S17 batch8 image probabilities",
              "t4_status": "T4_UNVERIFIED", "candidate_sha256": cfg["candidate_submission_sha256"]}
    (output / "paired_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
