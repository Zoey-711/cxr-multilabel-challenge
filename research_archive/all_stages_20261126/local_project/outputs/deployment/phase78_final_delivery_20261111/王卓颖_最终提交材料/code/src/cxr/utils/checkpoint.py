"""Portable state_dict checkpoints without dataset paths or Python object pickles."""

from pathlib import Path
from typing import Any

MAX_MODEL_BYTES = 500_000_000


def save_model_state(model: Any, path: str | Path, *, model_name: str = "densenet121") -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": model_name,
            "num_classes": 10,
            "model_state_dict": model.state_dict(),
        },
        target,
    )
    check_model_file_size(target)
    return target


def check_model_file_size(path: str | Path) -> int:
    size = Path(path).stat().st_size
    if size > MAX_MODEL_BYTES:
        raise ValueError(f"Model file exceeds 500 MB: {size} bytes")
    return size


def load_model_state(path: str | Path, *, device: Any = "cpu") -> Any:
    import torch

    from cxr.models.factory import create_model

    check_model_file_size(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("num_classes") != 10:
        raise ValueError("Checkpoint must have 10 output classes")
    model = create_model(checkpoint["model_name"], num_classes=10, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
