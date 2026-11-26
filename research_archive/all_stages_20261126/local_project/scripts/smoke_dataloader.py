"""Read a few real archive images on CUDA without loading or training a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from cxr.data.archive import archive_path, parse_positive_classes, read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/audits/dataloader_test.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for this smoke test")
    config = load_config(args.config)
    path = archive_path(config)
    unlisted_policy = config["data"]["label_encoding"]["unlisted_policy"]
    checks = {}
    for split in ("train", "val", "test"):
        all_rows = read_split_rows(path, split)
        counts = Counter((row["Subject_id"], row["Study_id"]) for row in all_rows)
        multi = next((key for key, count in counts.items() if count >= 2), None)
        if multi is None:
            raise AssertionError(f"No multi-image study in {split}")
        rows = [row for row in all_rows if (row["Subject_id"], row["Study_id"]) == multi][:2]
        loader = make_dataloader(path, split, rows=rows, batch_size=2, unlisted_policy=unlisted_policy)
        batch = next(iter(loader))
        images = batch["image"].to("cuda")
        assert tuple(images.shape) == (2, 3, 384, 384)
        assert torch.isfinite(images).all().item()
        assert list(batch["subject_id"]) == [row["Subject_id"] for row in rows]
        assert list(batch["study_id"]) == [row["Study_id"] for row in rows]
        assert list(batch["image_id"]) == [row["Dicom_id"] for row in rows]
        if split == "test":
            assert "target" not in batch and "valid_mask" not in batch
        else:
            assert tuple(batch["target"].shape) == (2, 10)
            assert tuple(batch["valid_mask"].shape) == (2, 10)
            assert torch.isfinite(batch["target"].to("cuda")).all().item()
            for index, row in enumerate(rows):
                expected = parse_positive_classes(row["Predict_class"])
                observed = {class_id for class_id in range(1, 11) if batch["target"][index, class_id - 1] == 1}
                assert observed == expected
                expected_mask = torch.ones(10, dtype=torch.bool) if unlisted_policy == "negative" else batch["target"][index].bool()
                assert torch.equal(batch["valid_mask"][index], expected_mask)
        predictions = [
            ImagePrediction(row["Subject_id"], row["Study_id"], row["Dicom_id"], (0.2 + 0.6 * index,) * 10)
            for index, row in enumerate(rows)
        ]
        fused = aggregate_mean_probabilities(predictions)
        assert len(fused) == 1 and fused[0].image_count == 2
        assert abs(fused[0].probabilities[0] - 0.5) < 1e-6
        assert (fused[0].subject_id, fused[0].study_id) == multi
        checks[split] = {
            "image_shape": list(images.shape), "target_shape": list(batch["target"].shape) if split != "test" else None,
            "mask_shape": list(batch["valid_mask"].shape) if split != "test" else None,
            "valid_label_count": int(batch["valid_mask"].sum()) if split != "test" else None,
            "subject_id": multi[0], "study_id": multi[1], "image_ids": list(batch["image_id"]),
            "study_fusion_image_count": fused[0].image_count,
            "fused_probability_first_class": fused[0].probabilities[0],
            "cuda_device": torch.cuda.get_device_name(0), "passed": True,
            "unlisted_policy": unlisted_policy if split != "test" else None,
        }
        loader.dataset.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
