"""Exercise best-model state_dict save/reload for one Phase17 backbone on GPU."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from cxr.data.patient_folds import sha256
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.checkpoint import load_model_state, save_model_state
from cxr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        parser.error("GPU required")
    name = config["model"]["name"]
    root = Path(__file__).resolve().parents[1]
    os.environ["TORCH_HOME"] = str(root / ".wheels" / "torch_cache")
    _, _, filename = imagenet_weight_info(name)
    weight = root / ".wheels" / "torch_cache" / "hub" / "checkpoints" / filename
    if sha256(weight) != config["model"]["pretrained_sha256"]:
        raise ValueError("ImageNet weight SHA-256 mismatch")
    model = create_model(name, pretrained=True).cuda().eval()
    x = torch.zeros(1, 3, 384, 384, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        original = model(x).float().cpu()
    with tempfile.TemporaryDirectory(dir=root / "outputs" / "analysis") as temporary:
        path = Path(temporary) / "best_model.pt"
        save_model_state(model, path, model_name=name)
        size, digest = path.stat().st_size, sha256(path)
        restored = load_model_state(path, device="cuda").eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            reloaded = restored(x).float().cpu()
    if not torch.equal(original, reloaded) or original.shape != (1, 10) or not torch.isfinite(original).all():
        raise ValueError("Best state_dict save/reload changed 10-class logits")
    result = {"model": name, "config_sha256": sha256(args.config),
              "pretrained_weight_sha256": sha256(weight), "checkpoint_roundtrip_sha256": digest,
              "checkpoint_roundtrip_size_bytes": size, "checkpoint_format": "state_dict",
              "logits_identical": True, "temporary_checkpoint_removed": True}
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
