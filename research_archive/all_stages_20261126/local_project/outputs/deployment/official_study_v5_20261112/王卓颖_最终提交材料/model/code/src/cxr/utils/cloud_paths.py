"""Opt-in cloud storage roots; defaults preserve all existing local paths."""

import os
from pathlib import Path


def output_root() -> Path:
    value = os.environ.get("CXR_OUTPUT_ROOT")
    if not value:
        return Path("outputs")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("CXR_OUTPUT_ROOT must be an absolute path")
    return path


def pretrained_home(project_root: Path) -> Path:
    value = os.environ.get("CXR_PRETRAINED_ROOT")
    if not value:
        return project_root / ".wheels" / "torch_cache"
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("CXR_PRETRAINED_ROOT must be an absolute path")
    return path
