import csv

import pytest
from PIL import Image

from cxr.inference.predict import make_transform, read_image_rows, resolve_image


def test_test_manifest_uses_explicit_columns_and_relative_paths(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "a.jpg"
    image.write_bytes(b"synthetic")
    manifest = tmp_path / "test.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient", "exam", "file"))
        writer.writeheader()
        writer.writerow({"patient": "p1", "exam": "s1", "file": "a.jpg"})
    assert read_image_rows(manifest, subject_column="patient", study_column="exam", image_column="file") == [("p1", "s1", "a.jpg")]
    assert resolve_image(image_root, "a.jpg") == image
    with pytest.raises(ValueError, match="escapes image root"):
        resolve_image(image_root, "../outside.jpg")


def test_inference_transform_preserves_aspect_ratio_with_padding():
    image = Image.new("L", (80, 160), 255)
    tensor = make_transform(384)(image)
    assert tuple(tensor.shape) == (3, 384, 384)
    assert tensor[0, 100, 192] > tensor[0, 100, 0]
