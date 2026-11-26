"""Portable image inference using an explicit image-level manifest mapping."""

import csv
from pathlib import Path
from typing import Any, Iterable

from cxr.inference.aggregate import ImagePrediction


def read_image_rows(manifest: str | Path, *, subject_column: str, study_column: str, image_column: str) -> list[tuple[str, str, str]]:
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {subject_column, study_column, image_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Test manifest lacks configured columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, start=2):
            subject, study, image = (str(row[column] or "").strip() for column in (subject_column, study_column, image_column))
            if not subject or not study or not image:
                raise ValueError(f"Blank required value in test manifest line {index}")
            rows.append((subject, study, image))
    if not rows:
        raise ValueError("Test manifest has no image rows")
    return rows


def resolve_image(image_root: str | Path, relative_path: str) -> Path:
    root = Path(image_root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Image paths must be relative to data.image_root for portability")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes image root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def make_transform(input_size: int, *, preprocessing: str = "black_pad_v1") -> Any:
    from cxr.data.dataset import preprocess_image

    return lambda image: preprocess_image(image, input_size, preprocessing=preprocessing)


def infer_image(model: Any, image_path: str | Path, *, transform: Any, device: Any) -> tuple[float, ...]:
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            probabilities = torch.sigmoid(model(tensor)).squeeze(0).float().cpu().tolist()
    if len(probabilities) != 10:
        raise ValueError("Model must output 10 probabilities per image")
    return tuple(probabilities)


def predict_image_rows(
    model: Any,
    rows: Iterable[tuple[str, str, str]],
    *,
    image_root: str | Path,
    input_size: int,
    device: Any,
    preprocessing: str = "black_pad_v1",
) -> list[ImagePrediction]:
    transform = make_transform(input_size, preprocessing=preprocessing)
    results = []
    for subject, study, image in rows:
        path = resolve_image(image_root, image)
        probabilities = infer_image(model, path, transform=transform, device=device)
        results.append(ImagePrediction(subject, study, image, probabilities))
    return results


def predict_archive_images(model: Any, archive: str | Path, *, batch_size: int,
                           num_workers: int, device: Any,
                           preprocessing: str = "black_pad_v1", input_size: int = 384) -> list[ImagePrediction]:
    """Predict the real test split directly from its original ZIP; read no labels."""
    import torch

    from cxr.data.dataset import make_dataloader

    loader = make_dataloader(archive, "test", batch_size=batch_size, num_workers=num_workers,
                             preprocessing=preprocessing, input_size=input_size)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                probabilities = torch.sigmoid(model(images)).float().cpu().tolist()
            for index, vector in enumerate(probabilities):
                predictions.append(ImagePrediction(
                    batch["subject_id"][index], batch["study_id"][index], batch["image_id"][index],
                    tuple(vector),
                ))
    loader.dataset.close()
    return predictions
