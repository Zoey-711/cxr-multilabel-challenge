"""Read experiment YAML without assuming any competition metadata fields."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    for section in ("experiment", "data", "model", "training", "validation", "inference"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid config section: {section}")
    if config["model"].get("num_classes") != 10:
        raise ValueError("Competition model must have 10 output classes")
    if config["data"].get("input_size") not in (384, 448, 512):
        raise ValueError("Supported input sizes are 384, 448 and 512")
    return config
