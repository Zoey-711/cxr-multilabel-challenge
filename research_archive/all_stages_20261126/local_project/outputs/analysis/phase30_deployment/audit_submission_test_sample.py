"""Read-only consistency check on 100 deterministic single-image Test Studies.

No Test labels, no AUC, and no optimization choice is made from these outputs.
The official fixed timing sample is not available; this is a separate local check.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "candidate_deployment"))
from cxr.data.archive import image_member, read_split_rows, split_archive_path
from predict import load_models, predict_image


def read_submission(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["Subject_id"], row["Study_id"])
            cls = int(row["Predict_class"])
            result.setdefault(key, np.full(10, np.nan))[cls - 1] = float(row["Probability"])
    if len(result) != 1414 or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Submission is not a complete 1414-Study probability matrix")
    return result


def main() -> None:
    archive = os.environ.get("CXR_DATA_ROOT")
    if not archive:
        raise ValueError("Set CXR_DATA_ROOT to the official unlabeled Test archive")
    rows = read_split_rows(archive, "test")
    counts = Counter((r["Subject_id"], r["Study_id"]) for r in rows)
    selected = sorted((r for r in rows if counts[(r["Subject_id"], r["Study_id"])] == 1),
                      key=lambda r: (r["Subject_id"], r["Study_id"]))[:100]
    if len(selected) != 100:
        raise ValueError("Need 100 complete single-image Test Studies")
    submissions = {
        "E08_EMA": ROOT / "outputs/submissions/E08_EMA_20261020/submission.csv",
        "H17_02": ROOT.parent / "cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/submissions/single_models/H17_DISTILL_02_submission.csv",
    }
    truth = {name: read_submission(path) for name, path in submissions.items()}
    observed = {name: [] for name in submissions}
    with zipfile.ZipFile(split_archive_path(archive, "test")) as source:
        for name in submissions:
            models = load_models(name)
            for row in selected:
                key = (row["Subject_id"], row["Study_id"])
                with Image.open(io.BytesIO(source.read(image_member(row, "test")))) as image:
                    calculated = predict_image(image, models, candidate=name)
                observed[name].append(np.abs(calculated - truth[name][key]))
            del models
    report = {"source": "official unlabeled Test, first 100 sorted single-image Studies",
              "official_fixed_timing_sample": False, "test_auc": None,
              "selection_rule": "Study cardinality exactly one, first 100 by Subject_id/Study_id; no predictions used to select",
              "candidates": {}}
    for name, pieces in observed.items():
        diff = np.stack(pieces)
        report["candidates"][name] = {"compared_studies": 100,
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "p99_abs_diff": float(np.percentile(diff, 99)),
            "values_above_1e-4": int(np.sum(diff > 1e-4)),
            "status_1e-4": "PASS" if float(diff.max()) <= 1e-4 else "FAIL",
            "frozen_submission": str(submissions[name])}
    output = HERE / "local_test_100_submission_consistency.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: report["candidates"][name]["max_abs_diff"] for name in submissions}))


if __name__ == "__main__":
    main()
