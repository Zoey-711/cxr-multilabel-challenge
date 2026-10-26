"""Study grouping uses IDs rather than image adjacency or count."""

import csv

import numpy as np

from scripts.train_study_heads import Studies, collate


def test_study_grouping_and_labels(tmp_path):
    with (tmp_path / "train_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", "Subject_id", "Study_id", "Image_id"])
        writer.writerows([(0, "subject_01", "study_01", "pic_01"),
                          (1, "subject_02", "study_02", "pic_02"),
                          (2, "subject_01", "study_01", "pic_03")])
    with (tmp_path / "train_studies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject_id", "Study_id"] + [f"label_{j}" for j in range(1, 11)])
        writer.writerows([("subject_01", "study_01", 1, *([0] * 9)),
                          ("subject_02", "study_02", *([0] * 9), 1)])
    values = np.zeros((3, 768), dtype=np.float16)
    values[:, 0] = [1, 2, 3]
    np.save(tmp_path / "train_features.npy", values)
    studies = Studies(tmp_path, "train")
    assert studies.indices == [[0, 2], [1]]
    x, mask, y, index = collate([studies[0], studies[1]])
    assert x.shape == (2, 2, 768)
    assert mask.tolist() == [[True, True], [True, False]]
    assert x[0, :, 0].tolist() == [1, 3]
    assert y[0, 0] == 1 and y[1, 9] == 1
    assert index.tolist() == [0, 1]
