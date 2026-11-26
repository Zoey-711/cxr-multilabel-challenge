"""Construct competition classifiers; pretrained=True may populate TORCH_HOME."""

from pathlib import PurePosixPath


def imagenet_weight_info(name: str) -> tuple[str, str, str]:
    """Return the official torchvision weight enum name, URL, and cache filename."""
    try:
        from torchvision.models import ConvNeXt_Tiny_Weights, DenseNet121_Weights, EfficientNet_V2_S_Weights, Swin_T_Weights
    except ImportError as exc:
        raise RuntimeError("Install torchvision in the Python 3.12 environment") from exc
    weights = {"densenet121": DenseNet121_Weights.IMAGENET1K_V1,
               "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
               "efficientnet_v2_s": EfficientNet_V2_S_Weights.IMAGENET1K_V1,
               "swin_t": Swin_T_Weights.IMAGENET1K_V1}.get(name)
    if weights is None:
        raise ValueError(f"Unsupported model: {name}")
    return f"torchvision {type(weights).__name__}.{weights.name}", weights.url, PurePosixPath(weights.url).name


def create_model(name: str, *, num_classes: int = 10, pretrained: bool = False):
    if name not in ("densenet121", "convnext_tiny", "efficientnet_v2_s", "swin_t"):
        raise ValueError(f"Unsupported model: {name}")
    if num_classes != 10:
        raise ValueError("This competition requires 10 output classes")
    try:
        from torch import nn
        from torchvision.models import (ConvNeXt_Tiny_Weights, DenseNet121_Weights,
                                        EfficientNet_V2_S_Weights, Swin_T_Weights,
                                        convnext_tiny, densenet121, efficientnet_v2_s, swin_t)
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and torchvision in the Python 3.12 environment") from exc
    if name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        if not isinstance(model.classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt-Tiny classifier structure")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        if not isinstance(model.classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected torchvision EfficientNetV2-S classifier structure")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        if not isinstance(model.head, nn.Linear) or model.head.in_features != 768:
            raise RuntimeError("Unexpected torchvision Swin-Tiny classification head")
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model
