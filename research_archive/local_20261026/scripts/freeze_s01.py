"""Validate and seal all completed S01 artifacts without touching frozen controls."""

import argparse
import gzip
import hashlib
import json
import os
import stat
from pathlib import Path

from cxr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest_path = run / "frozen_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if load_config(args.config) != load_config(run / "config.yaml"):
        raise ValueError("S01 runtime config differs")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete":
        raise ValueError("S01 training is incomplete")
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    independent = json.loads((run / "validation_analysis" / "analysis.json").read_text(encoding="utf-8"))
    comparison = json.loads((run / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records or [record["epoch"] for record in records] != list(range(1, len(records) + 1)) or \
            records[-1]["epoch"] != status["stop_epoch"] or best["epoch"] != status["best_epoch"] or \
            status["early_stopping_patience"] != 6 or independent["study_count"] != 859 or \
            abs(independent["macro_auc"] - best["val"]["macro_auc"]) > 1e-8 or \
            abs(comparison["scores"]["S01"]["macro_auc"] - independent["macro_auc"]) > 1e-8:
        raise ValueError("S01 training, independent validation, or comparison is inconsistent")
    for record in records:
        sampling = record["study_sampling"]
        detail = run / sampling["detail_file"]
        if sampling["actual_image_draws"] != 30344 or sampling["study_count"] != 18513 or \
                {int(key): value for key, value in sampling["study_exposure_histogram"].items()} != {1: 6682, 2: 11831} or \
                record["optimizer_steps"] + record["amp_skipped_steps"] != 1897 or not detail.is_file():
            raise ValueError(f"S01 epoch {record['epoch']} sampling or optimizer-group count differs")
        with gzip.open(detail, "rt", encoding="utf-8", newline="") as handle:
            if sum(1 for _ in handle) != 18514:
                raise ValueError(f"S01 epoch {record['epoch']} per-Study sampling record is incomplete")
    manifest = {}
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.name == "frozen_manifest.json":
            continue
        relative = path.relative_to(run).as_posix()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest[relative] = {"size_bytes": path.stat().st_size, "sha256": digest}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in run.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    print(json.dumps({"run_dir": str(run), "best_epoch": best["epoch"],
                      "macro_auc": independent["macro_auc"], "files_frozen": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
