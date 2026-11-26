"""Verify standalone ONNX deployment reproduces the evaluated Phase49 Val CSV."""
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"
sys.path[:0] = [str(HERE / "deployment_candidate"),
                str(ROOT / "outputs/analysis/phase44_e08_acceleration/isolated_deps"),
                str(PACKAGE / "src")]
import predict as candidate
from cxr.data.archive import image_member, read_split_rows, split_archive_path
from cxr.data.dataset import preprocess_image

archive = Path("F:/数据集/cxr_first.zip")
rows = read_split_rows(archive, "val")
model = candidate.load_model()
grouped = defaultdict(list)
max_tensor = 0.0
with zipfile.ZipFile(split_archive_path(archive, "val")) as source:
    for row in rows:
        with Image.open(io.BytesIO(source.read(image_member(row, "val")))) as im:
            old_tensor = preprocess_image(im, 384, preprocessing="black_pad_v1").numpy()
            new_tensor = candidate.preprocess(im)[0]
            max_tensor = max(max_tensor, float(np.max(np.abs(old_tensor - new_tensor))))
            grouped[(row["Subject_id"], row["Study_id"])].append(candidate.predict_image(im, model))
actual = {k: np.mean(v, axis=0) for k, v in grouped.items()}
with (HERE / "val_onnx_fp32.csv").open(encoding="utf-8", newline="") as stream:
    reference = {(r["Subject_id"], r["Study_id"]): np.array([float(r[f"Prob_{j}"]) for j in range(1, 11)])
                 for r in csv.DictReader(stream)}
if set(actual) != set(reference) or len(actual) != 859:
    raise ValueError("Val Study IDs differ")
diff = np.abs(np.asarray([actual[k] - reference[k] for k in sorted(actual)]))
result = dict(studies=859, images=len(rows), input_tensor_max_abs_diff=max_tensor,
              deployment_probability_max_abs_diff=float(diff.max()),
              deployment_probability_mean_abs_diff=float(diff.mean()),
              pass_exact=bool(max_tensor == 0 and diff.max() == 0))
(HERE / "deployment_entry_consistency.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
if not result["pass_exact"]:
    raise SystemExit(2)
