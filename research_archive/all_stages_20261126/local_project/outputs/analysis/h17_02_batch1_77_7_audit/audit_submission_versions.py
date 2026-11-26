"""Read-only comparison of the two H17-02 submissions; writes only here."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD_ROOT = ROOT.parent / "cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch"
OLD = OLD_ROOT / "submissions/single_models/H17_DISTILL_02_submission.csv"
NEW = ROOT / "outputs/submissions/H17_02_BATCH1_20261024/submission.csv"
NEW_MANIFEST = NEW.parent / "manifest.json"
PACKAGE = ROOT / "outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Subject_id", "Study_id", "Predict_class", "Probability"]:
            raise ValueError(f"Bad columns: {path}")
        rows = list(reader)
    values = {}
    studies = Counter()
    for row in rows:
        key = (row["Subject_id"], row["Study_id"], int(row["Predict_class"]))
        value = float(row["Probability"])
        if key in values or not 1 <= key[2] <= 10 or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Bad or duplicate prediction: {path}")
        values[key] = value
        studies[key[:2]] += 1
    if len(rows) != 14140 or len(studies) != 1414 or set(studies.values()) != {10}:
        raise ValueError(f"Incomplete submission: {path}")
    return values


def read_wide(path: Path):
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for cls in range(1, 11):
                key = (row["Subject_id"], row["Study_id"], cls)
                if key in values:
                    raise ValueError("Duplicate wide probability")
                values[key] = float(row[f"Prob_{cls}"])
    return values


def main():
    old, new = read(OLD), read(NEW)
    if set(old) != set(new):
        raise ValueError("Study/class keys differ between old and batch1 submissions")
    ordered = sorted(old)
    old_wide = read_wide(OLD_ROOT / "experiments/H17_DISTILL_02/test_predictions.csv")
    if set(old_wide) != set(old):
        raise ValueError("Old wide Test coverage differs from old submission")
    old_wide_submission_max_diff = max(abs(old_wide[k] - old[k]) for k in ordered)
    old_provenance = json.loads((OLD_ROOT / "experiments/H17_DISTILL_02/test_predictions.provenance.json").read_text())
    diffs = np.array([abs(new[key] - old[key]) for key in ordered], dtype=np.float64)
    biggest = ordered[int(np.argmax(diffs))]
    manifest = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    package_manifest = json.loads((PACKAGE / "package_manifest.json").read_text(encoding="utf-8"))
    missing_package_files = []
    changed_package_files = []
    for relative, expected in package_manifest["files"].items():
        item = PACKAGE / relative
        if not item.is_file():
            missing_package_files.append(relative)
        elif item.stat().st_size != expected["bytes"] or sha(item) != expected["sha256"]:
            changed_package_files.append(relative)
    checked = {}
    for relative in ("model/H17_02.pth", "config/deployment_config.json", "config/resolved_config.yaml",
                     "predict.py", "cost.py", "requirements.txt"):
        item = PACKAGE / relative
        expected = package_manifest["files"][relative]
        checked[relative] = dict(bytes=item.stat().st_size, sha256=sha(item),
                                 pass_hash=sha(item) == expected["sha256"] and item.stat().st_size == expected["bytes"])
    if not all(x["pass_hash"] for x in checked.values()):
        raise ValueError("Deployment package changed")
    old_manifest = json.loads((OLD.parent / "H17_DISTILL_02_submission.manifest.json").read_text(encoding="utf-8"))
    if sha(OLD) != old_manifest["format"]["sha256"] or sha(NEW) != manifest["submission_sha256"]:
        raise ValueError("Submission hash mismatch")
    if old_manifest["checkpoint_sha256"] != [checked["model/H17_02.pth"]["sha256"]]:
        raise ValueError("Old/new checkpoint identity differs")
    result = dict(old_submission=str(OLD), new_submission=str(NEW),
                  old_submission_sha256=sha(OLD), new_submission_sha256=sha(NEW),
                  old_wide_submission_max_diff=old_wide_submission_max_diff,
                  old_test_provenance_checkpoint_sha256=old_provenance["checkpoint_sha256"],
                  model_sha256=checked["model/H17_02.pth"]["sha256"],
                  old_val_auc=old_manifest["val_macro_auc"],
                  new_val_auc=manifest["validation_macro_auc_batch1"],
                  same_study_class_keys=True, study_count=1414, row_count=14140,
                  changed_probability_rows=int(np.count_nonzero(diffs)),
                  mean_abs_probability_change=float(diffs.mean()),
                  p99_abs_probability_change=float(np.percentile(diffs, 99)),
                  max_abs_probability_change=float(diffs.max()),
                  max_difference_key=dict(subject=biggest[0], study=biggest[1], class_id=biggest[2]),
                  old_probability=float(old[biggest]), new_probability=float(new[biggest]),
                  necessary_model_and_config_bytes=(PACKAGE / "model/H17_02.pth").stat().st_size +
                                                   (PACKAGE / "config/deployment_config.json").stat().st_size +
                                                   (PACKAGE / "config/resolved_config.yaml").stat().st_size,
                  package_without_results_bytes=package_manifest["full_package_without_results_bytes"],
                  package_manifest_file_count=len(package_manifest["files"]),
                  missing_package_files=missing_package_files,
                  changed_package_files=changed_package_files,
                  package_files=checked, official_new_score_user_confirmed=77.7,
                  official_old_score_user_reported=77.50,
                  score_source="user report; no local organizer receipt available",
                  causality="old Test provenance references a different checkpoint file hash; batch-only cause not proven")
    output = HERE / "submission_version_comparison_v3.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("changed_probability_rows", "mean_abs_probability_change",
                      "p99_abs_probability_change", "max_abs_probability_change",
                      "necessary_model_and_config_bytes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
