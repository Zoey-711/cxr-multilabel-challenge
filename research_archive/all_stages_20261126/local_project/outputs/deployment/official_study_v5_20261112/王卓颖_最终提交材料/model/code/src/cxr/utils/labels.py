"""Canonical model output order and original competition IDs."""

CLASS_NAMES = (
    "Enlarged Cardiomediastinum",
    "Pneumothorax",
    "Consolidation",
    "Pneumonia",
    "Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Lung Opacity",
    "Pleural Effusion",
    "No Finding",
)


def original_class_id(model_index: int) -> int:
    if not 0 <= model_index < len(CLASS_NAMES):
        raise ValueError(f"Model class index must be 0–9, got {model_index}")
    return model_index + 1
