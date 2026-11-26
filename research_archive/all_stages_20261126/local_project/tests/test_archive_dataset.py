import csv
import io
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TF

from cxr.data.archive import image_member, parse_positive_classes, read_split_rows
from cxr.data.dataset import (CXRArchiveDataset, IMAGENET_MEAN, IMAGENET_MEAN_RGB8,
                              IMAGENET_STD, augment_train_image, preprocess_image,
                              resize_pad_image, study_targets_from_rows)
from cxr.inference.predict import predict_archive_images


def _jpeg() -> bytes:
    image = Image.new("L", (80, 160), 128)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _archive(path):
    columns = ["Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation", "Rows", "Columns"]
    with zipfile.ZipFile(path, "w") as archive:
        for split, label in (("train", "1;9"), ("valid", "2"), ("test", None)):
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns + (["Predict_class"] if label is not None else []))
            writer.writeheader()
            row = dict(zip(columns, ["pic_1", "subject_1", "study_1", "AP", "Erect", "160", "80"]))
            if label is not None:
                row["Predict_class"] = label
            writer.writerow(row)
            archive.writestr(f"cxr_first/{split}.csv", output.getvalue())
            archive.writestr(f"cxr_first/{split}/subject_1/study_1/pic_1.jpg", _jpeg())


def test_archive_mapping_and_test_has_no_fake_labels(tmp_path):
    path = tmp_path / "tiny.zip"
    _archive(path)
    rows = read_split_rows(path, "train")
    assert image_member(rows[0], "train") == "cxr_first/train/subject_1/study_1/pic_1.jpg"
    sample = CXRArchiveDataset(path, "train")[0]
    assert tuple(sample["image"].shape) == (3, 384, 384)
    assert sample["target"].nonzero().flatten().tolist() == [0, 8]
    assert sample["valid_mask"].all()
    study_target = study_targets_from_rows(rows)[0]
    assert study_target.labels[0] == 1.0 and study_target.labels[8] == 1.0
    assert study_target.labels[1] == 0.0
    conservative = CXRArchiveDataset(path, "train", unlisted_policy="unknown")[0]
    assert torch.equal(conservative["valid_mask"], conservative["target"].bool())
    assert study_targets_from_rows(rows, unlisted_policy="unknown")[0].labels[1] is None
    test_sample = CXRArchiveDataset(path, "test")[0]
    assert "target" not in test_sample and "valid_mask" not in test_sample
    assert test_sample["study_id"] == "study_1"


def test_padding_preserves_aspect_and_class_parser_is_strict():
    image = Image.new("L", (80, 160), 255)
    tensor = preprocess_image(image)
    assert torch.equal(tensor[0, :, 0], tensor[0, :, 383])
    assert tensor[0, 100, 192] > tensor[0, 100, 0]
    assert parse_positive_classes("1;9") == {1, 9}
    for invalid in ("0", "11", "1;1", "1;-1", "1; 2"):
        with pytest.raises(ValueError):
            parse_positive_classes(invalid)


def test_512_padding_preserves_entire_image_and_label_mapping(tmp_path):
    path = tmp_path / "tiny.zip"
    _archive(path)
    sample = CXRArchiveDataset(path, "train", input_size=512)[0]
    assert tuple(sample["image"].shape) == (3, 512, 512)
    assert sample["target"].nonzero().flatten().tolist() == [0, 8]
    assert torch.isfinite(sample["image"]).all()
    for width, height in ((80, 160), (160, 80)):
        image = Image.new("L", (width, height), 255)
        fitted = resize_pad_image(image, 512)
        pixels = np.asarray(fitted)
        assert fitted.size == (512, 512)
        # Every source dimension is visible after fit; only the shorter axis is padded.
        if width < height:
            assert (pixels[:, 128:384] == 255).all()
            assert (pixels[:, :128] == 0).all()
        else:
            assert (pixels[128:384, :] == 255).all()
            assert (pixels[:128, :] == 0).all()


def test_m03_mean_pad_preserves_geometry_and_changes_only_border():
    for size in ((80, 160), (160, 80), (160, 160)):
        image = Image.new("L", size, 255)
        old = resize_pad_image(image, preprocessing="black_pad_v1")
        new = resize_pad_image(image, preprocessing="imagenet_mean_pad_v1")
        assert old.size == new.size == (384, 384)
        old_pixels = torch.from_numpy(np.asarray(old).copy()).reshape(-1, 3)
        new_pixels = torch.from_numpy(np.asarray(new).copy()).reshape(-1, 3)
        source_mask = (old_pixels == 255).all(dim=1)
        assert torch.equal(source_mask, (new_pixels == 255).all(dim=1))
        assert source_mask.any()
        assert torch.equal(old_pixels[source_mask], new_pixels[source_mask])
        if size[0] != size[1]:
            assert tuple(new_pixels[~source_mask][0].tolist()) == IMAGENET_MEAN_RGB8
            assert tuple(old_pixels[~source_mask][0].tolist()) == (0, 0, 0)
        tensor = preprocess_image(image, preprocessing="imagenet_mean_pad_v1")
        assert torch.isfinite(tensor).all()
        if size[0] != size[1]:
            border = tensor[:, 0, 0]
            expected = torch.tensor([(c / 255 - m) / s for c, m, s in
                                     zip(IMAGENET_MEAN_RGB8, IMAGENET_MEAN, IMAGENET_STD)])
            assert torch.allclose(border, expected, atol=1e-6)
    with pytest.raises(ValueError, match="Unsupported preprocessing"):
        preprocess_image(Image.new("L", (80, 160)), preprocessing="crop")


def test_archive_test_inference_keeps_study_ids_and_has_no_labels(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("GPU-only inference requires CUDA")
    path = tmp_path / "tiny.zip"
    _archive(path)

    class ConstantModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((len(images), 10), device=images.device)

    predictions = predict_archive_images(ConstantModel().cuda(), path, batch_size=1,
                                         num_workers=0, device="cuda")
    assert len(predictions) == 1
    assert (predictions[0].subject_id, predictions[0].study_id, predictions[0].image_id) == (
        "subject_1", "study_1", "pic_1")
    assert predictions[0].probabilities == (0.5,) * 10


def test_conservative_augmentation_is_train_only_and_reproducible(tmp_path):
    path = tmp_path / "tiny.zip"
    _archive(path)
    settings = {"enabled": True, "name": "conservative_cxr_v1", "affine_probability": 1.0,
                "rotation_degrees": 3.0, "translation_fraction": 0.02,
                "photometric_probability": 1.0, "brightness_contrast_fraction": 0.05}
    baseline = CXRArchiveDataset(path, "train")[0]["image"]
    torch.manual_seed(42)
    first = CXRArchiveDataset(path, "train", augmentation=settings)[0]["image"]
    torch.manual_seed(42)
    second = CXRArchiveDataset(path, "train", augmentation=settings)[0]["image"]
    assert torch.equal(first, second)
    assert not torch.equal(first, baseline)
    assert torch.isfinite(first).all()
    assert torch.equal(CXRArchiveDataset(path, "val")[0]["image"], baseline)
    with pytest.raises(ValueError, match="training split"):
        CXRArchiveDataset(path, "val", augmentation=settings)


def test_conservative_augmentation_uses_configured_bounds_and_order():
    image = Image.new("RGB", (100, 150), (120, 80, 30))
    settings = {"enabled": True, "name": "conservative_cxr_v1", "affine_probability": 1.0,
                "rotation_degrees": 3.0, "translation_fraction": 0.02,
                "photometric_probability": 1.0, "brightness_contrast_fraction": 0.05}
    torch.manual_seed(123)
    _ = torch.rand(())  # affine probability
    angle = (2 * torch.rand(()).item() - 1) * 3.0
    dx = round((2 * torch.rand(()).item() - 1) * 0.02 * image.width)
    dy = round((2 * torch.rand(()).item() - 1) * 0.02 * image.height)
    _ = torch.rand(())  # photometric probability
    brightness = 1 + (2 * torch.rand(()).item() - 1) * 0.05
    contrast = 1 + (2 * torch.rand(()).item() - 1) * 0.05
    assert abs(angle) <= 3 and abs(dx) <= 2 and abs(dy) <= 3
    assert 0.95 <= brightness <= 1.05 and 0.95 <= contrast <= 1.05
    expected = TF.affine(image, angle=angle, translate=[dx, dy], scale=1.0,
                         shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
                         fill=[0, 0, 0])
    expected = TF.adjust_contrast(TF.adjust_brightness(expected, brightness), contrast)
    torch.manual_seed(123)
    actual = augment_train_image(image, settings)
    assert actual.tobytes() == expected.tobytes()
