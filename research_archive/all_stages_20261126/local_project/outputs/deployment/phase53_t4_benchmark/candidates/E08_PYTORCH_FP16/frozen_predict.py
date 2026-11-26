"""Phase30 candidate deployment. Frozen E08/H17 weights; no training or test adaptation.

Both this module and cost.py call load_models() and predict_image(). The input
transform is the exact black_pad_v1 transform verified in Phase20.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from cxr.data.dataset import preprocess_image
from cxr.utils.checkpoint import load_model_state
from cxr.models.factory import create_model

E08_MANIFEST = HERE / "frozen_e08_manifest.json"
ASSET_ROOT = HERE / "model"
H17_WEIGHT = (ASSET_ROOT / "H17_02.pth") if ASSET_ROOT else (
    ROOT / "outputs/analysis/phase29_deployment/weights/H17_DISTILL_02_best_model.pth")
H17_SHA = "35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
NAMES = ("B01", "B02", "B03", "M02", "N01_EMA")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_models(candidate: str, device: str = "cuda") -> dict:
    if candidate == "H17_02":
        if not H17_WEIGHT.is_file() or sha256(H17_WEIGHT) != H17_SHA:
            raise FileNotFoundError("H17-02 inference weight missing or SHA mismatch")
        # Friend best_model.pth is a pure state_dict, unlike the wrapped local .pt files.
        model = create_model("efficientnet_v2_s", num_classes=10, pretrained=False)
        model.load_state_dict(torch.load(H17_WEIGHT, map_location="cpu", weights_only=True), strict=True)
        return {"H17_02": model.to(device).eval()}
    if candidate != "E08_EMA":
        raise ValueError(candidate)
    manifest = json.loads(E08_MANIFEST.read_text(encoding="utf-8"))
    models = {}
    for name in NAMES:
        meta = manifest["models"][name]
        if meta["preprocessing"] != "black_pad_v1" or meta["input_size"] != 384:
            raise ValueError(f"Unexpected frozen preprocessing: {name}")
        path = (ASSET_ROOT / f"{name}.pt") if ASSET_ROOT else (
            ROOT / "outputs/experiments" / meta["run_id"] / meta["checkpoint"])
        if path.stat().st_size != meta["checkpoint_bytes"] or sha256(path) != meta["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint mismatch: {name}")
        models[name] = load_model_state(path, device=device)
    return models

def predict_image(image_path: str | Path | Image.Image, models: dict, *, candidate: str, device: str = "cuda") -> np.ndarray:
    """Read a single image, preprocess once, infer, return ten image probabilities."""
    # The PIL input form is only for read-only archive audits; official timing
    # always passes a Path so file opening/decoding remains inside the timer.
    if isinstance(image_path, Image.Image):
        tensor = preprocess_image(image_path, 384, preprocessing="black_pad_v1")
    else:
        with Image.open(image_path) as source:
            tensor = preprocess_image(source, 384, preprocessing="black_pad_v1")
    tensor = tensor.unsqueeze(0).to(device)
    probabilities = {}
    with torch.inference_mode():
        for name, model in models.items():
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda"), dtype=torch.float16):
                probabilities[name] = torch.sigmoid(model(tensor)).squeeze(0).float().cpu().numpy().astype(np.float64)
    if candidate == "E08_EMA":
        e01 = (probabilities["B01"] + probabilities["B02"] + probabilities["B03"]) / 3.0
        e03 = 0.5 * e01 + 0.5 * probabilities["M02"]
        result = 0.5 * e03 + 0.5 * probabilities["N01_EMA"]
    else:
        result = probabilities["H17_02"]
    if result.shape != (10,) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid image probabilities")
    return result

def predict_studies(rows: list[tuple[str, str, Path]], models: dict, *, candidate: str,
                    device: str = "cuda") -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for subject, study, path in rows:
        grouped.setdefault((subject, study), []).append(predict_image(path, models, candidate=candidate, device=device))
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}

def write_probabilities(path: Path, values: dict[tuple[str, str], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id"] + [f"Prob_{i}" for i in range(1, 11)])
        for (subject, study), prob in sorted(values.items()):
            writer.writerow([subject, study] + [format(float(x), ".17g") for x in prob])

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",choices=["E08_EMA","H17_02"],required=True)
    parser.add_argument("--image-manifest",type=Path,required=True,
                        help="CSV: Subject_id,Study_id,image_path (relative to --image-root)")
    parser.add_argument("--image-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-studies",type=int,default=1414)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing submission")
    root=args.image_root.resolve()
    with args.image_manifest.open(encoding="utf-8-sig",newline="") as f:
        records=list(csv.DictReader(f))
    rows=[]
    for rec in records:
        relative=Path(rec["image_path"])
        if relative.is_absolute():
            raise ValueError("Absolute image path prohibited")
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing image or path escapes root: {relative}")
        rows.append((rec["Subject_id"],rec["Study_id"],path))
    if len(rows)!=len({str(p) for _,_,p in rows}):
        raise ValueError("Duplicate image path")
    models=load_models(args.candidate)
    studies=predict_studies(rows,models,candidate=args.candidate)
    if len(studies)!=args.expected_studies:
        raise ValueError(f"Expected {args.expected_studies} Studies, got {len(studies)}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["Subject_id","Study_id","Predict_class","Probability"])
        for (subject,study),prob in sorted(studies.items()):
            for class_id,value in enumerate(prob,start=1):
                writer.writerow([subject,study,class_id,format(float(value),".17g")])
    print(json.dumps({"studies":len(studies),"rows":10*len(studies),"sha256":sha256(args.output)}))

if __name__=="__main__":
    main()
