"""Archive-backed image dataset with study IDs and label masks."""

import io
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, functional as TF

from cxr.data.archive import image_member, parse_positive_classes, read_split_rows, split_archive_path
from cxr.data.sampling import StudyBalancedSampler
from cxr.engine.validate import StudyTarget


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_MODES = ("black_pad_v1", "imagenet_mean_pad_v1")
IMAGENET_MEAN_RGB8 = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


def augment_train_image(image: Image.Image, settings: dict | None) -> Image.Image:
    """Apply small CXR-safe perturbations before the unchanged M01 preprocessing."""
    if not settings or not settings.get("enabled", False):
        return image
    if settings.get("name") != "conservative_cxr_v1":
        raise ValueError("Unsupported training augmentation")
    rgb = image.convert("RGB")
    if torch.rand(()) < float(settings["affine_probability"]):
        rotation = float(settings["rotation_degrees"])
        translation = float(settings["translation_fraction"])
        angle = (2 * torch.rand(()).item() - 1) * rotation
        shift = [round((2 * torch.rand(()).item() - 1) * translation * dimension)
                 for dimension in rgb.size]
        rgb = TF.affine(rgb, angle=angle, translate=shift, scale=1.0, shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
    if torch.rand(()) < float(settings["photometric_probability"]):
        variation = float(settings["brightness_contrast_fraction"])
        brightness = 1.0 + (2 * torch.rand(()).item() - 1) * variation
        contrast = 1.0 + (2 * torch.rand(()).item() - 1) * variation
        rgb = TF.adjust_brightness(rgb, brightness)
        rgb = TF.adjust_contrast(rgb, contrast)
    return rgb


def resize_pad_image(image: Image.Image, size: int = 384,
                     *, preprocessing: str = "black_pad_v1") -> Image.Image:
    """Fit all source pixels into a square with bilinear resize and centered padding."""
    if size not in (384, 448, 512):
        raise ValueError("Supported image sizes are 384, 448 and 512")
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")
    rgb = image.convert("RGB")
    fill = (0, 0, 0) if preprocessing == "black_pad_v1" else IMAGENET_MEAN_RGB8
    return ImageOps.pad(rgb, (size, size), method=Image.Resampling.BILINEAR, color=fill)


def preprocess_image(image: Image.Image, size: int = 384,
                     *, preprocessing: str = "black_pad_v1") -> torch.Tensor:
    """Convert to RGB, fit without crop, then apply ImageNet normalization."""
    padded = resize_pad_image(image, size, preprocessing=preprocessing)
    return TF.normalize(TF.to_tensor(padded), IMAGENET_MEAN, IMAGENET_STD)


class CXRArchiveDataset(Dataset):
    """One sample per image; all images of a study retain the same study ID."""

    def __init__(self, archive: str | Path, split: str, *, rows: list[dict[str, str]] | None = None,
                 unlisted_policy: str = "negative", augmentation: dict | None = None,
                 preprocessing: str = "black_pad_v1", input_size: int = 384,
                 study_soft_targets: dict | None = None, study_soft_weights: dict | None = None):
        if unlisted_policy not in ("negative", "unknown"):
            raise ValueError("unlisted_policy must be 'negative' or 'unknown'")
        self.archive = split_archive_path(archive, split)
        self.split = split
        self.unlisted_policy = unlisted_policy
        if augmentation and split != "train":
            raise ValueError("Augmentation is restricted to the training split")
        self.augmentation = augmentation
        if preprocessing not in PREPROCESSING_MODES:
            raise ValueError(f"Unsupported preprocessing: {preprocessing}")
        self.preprocessing = preprocessing
        if input_size not in (384, 448, 512):
            raise ValueError("Supported image sizes are 384, 448 and 512")
        self.input_size = input_size
        self.rows = read_split_rows(self.archive, split) if rows is None else rows
        if (study_soft_targets is None) != (study_soft_weights is None):
            raise ValueError("Teacher targets and Study weights must be supplied together")
        if study_soft_targets is not None:
            if split != "train":
                raise ValueError("Teacher targets are restricted to Train")
            studies = {(row["Subject_id"], row["Study_id"]) for row in self.rows}
            if studies != set(study_soft_targets) or studies != set(study_soft_weights):
                raise ValueError("Teacher targets differ from Train Study IDs")
        self.study_soft_targets = study_soft_targets
        self.study_soft_weights = study_soft_weights
        self._zip: zipfile.ZipFile | None = None
        self._members = [image_member(row, split) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive)
        row = self.rows[index]
        member = self._members[index]
        with Image.open(io.BytesIO(self._zip.read(member))) as image:
            pixels = preprocess_image(augment_train_image(image, self.augmentation),
                                      size=self.input_size, preprocessing=self.preprocessing)
        item = {
            "image": pixels,
            "subject_id": row["Subject_id"],
            "study_id": row["Study_id"],
            "image_id": row["Dicom_id"],
            "image_member": member,
        }
        if self.split != "test":
            if not row["Predict_class"].strip():
                raise ValueError(f"Blank Predict_class for {row['Study_id']}")
            positives = parse_positive_classes(row["Predict_class"])
            target = torch.zeros(10, dtype=torch.float32)
            valid = torch.full((10,), self.unlisted_policy == "negative", dtype=torch.bool)
            for class_id in positives:
                target[class_id - 1] = 1.0
                valid[class_id - 1] = True
            item["target"] = target
            item["valid_mask"] = valid
            if self.study_soft_targets is not None:
                key = row["Subject_id"], row["Study_id"]
                item["soft_target"] = torch.as_tensor(self.study_soft_targets[key], dtype=torch.float32)
                item["soft_weight"] = torch.tensor(self.study_soft_weights[key], dtype=torch.float32)
        return item

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None


def make_dataloader(archive: str | Path, split: str, *, batch_size: int = 2,
                    num_workers: int = 0, seed: int = 42, shuffle: bool = False,
                    rows: list[dict[str, str]] | None = None,
                    unlisted_policy: str = "negative", augmentation: dict | None = None,
                    preprocessing: str = "black_pad_v1", sampling: str | None = None,
                    input_size: int = 384, persistent_workers: bool | None = None,
                    study_soft_targets: dict | None = None,
                    study_soft_weights: dict | None = None) -> DataLoader:
    if shuffle and split != "train":
        raise ValueError("Only training data may be shuffled")
    if sampling and (split != "train" or shuffle):
        raise ValueError("Study-balanced sampling requires train split without shuffle")
    if sampling not in (None, "study_balanced_v1"):
        raise ValueError(f"Unsupported training sampling: {sampling}")
    dataset = CXRArchiveDataset(archive, split, rows=rows, unlisted_policy=unlisted_policy,
                                augmentation=augmentation, preprocessing=preprocessing,
                                input_size=input_size, study_soft_targets=study_soft_targets,
                                study_soft_weights=study_soft_weights)
    generator = torch.Generator().manual_seed(seed)
    sampler = StudyBalancedSampler(dataset.rows, seed=seed) if sampling else None
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, sampler=sampler, generator=generator,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=(num_workers > 0 if persistent_workers is None else
                                          bool(persistent_workers) and num_workers > 0))


def study_targets_from_rows(rows: list[dict[str, str]], *, unlisted_policy: str = "negative") -> list[StudyTarget]:
    """Return one target per study under the recorded competition label rule."""
    if unlisted_policy not in ("negative", "unknown"):
        raise ValueError("unlisted_policy must be 'negative' or 'unknown'")
    targets: dict[tuple[str, str], tuple[float | None, ...]] = {}
    for row in rows:
        if "Predict_class" not in row:
            raise ValueError("Study targets require labeled train/validation rows")
        if not row["Predict_class"].strip():
            raise ValueError(f"Blank Predict_class for {row['Study_id']}")
        key = (row["Subject_id"], row["Study_id"])
        positives = parse_positive_classes(row["Predict_class"])
        labels = tuple(1.0 if class_id in positives else (0.0 if unlisted_policy == "negative" else None)
                       for class_id in range(1, 11))
        if key in targets and targets[key] != labels:
            raise ValueError(f"Inconsistent study labels: {key}")
        targets[key] = labels
    return [StudyTarget(subject, study, targets[(subject, study)]) for subject, study in sorted(targets)]
