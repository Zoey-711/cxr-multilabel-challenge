"""Check DenseNet output dimensions without downloading weights."""

import argparse
from pathlib import Path

from cxr.models.factory import create_model
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"]["pretrained"] != "imagenet":
        parser.error("This check expects the baseline ImageNet source declaration")
    model = create_model(config["model"]["name"], num_classes=config["model"]["num_classes"], pretrained=False)
    head = model.head if config["model"]["name"] == "swin_t" else model.classifier
    out_features = head.out_features if hasattr(head, "out_features") else head[-1].out_features
    print(f"{type(model).__name__}: classifier output = {out_features}; weights downloaded = no")


if __name__ == "__main__":
    main()
