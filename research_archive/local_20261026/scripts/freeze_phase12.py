"""Hash and mark complete Phase 12 feature/head outputs read-only."""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import yaml


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache = root / cfg["source"]["feature_cache"]
    result = root / "outputs/experiments/v00_v01_v02_m02_frozen_features"
    if (result / "frozen_manifest.json").exists():
        raise FileExistsError("Phase 12 already frozen")
    required = [cache / "manifest.json", result / "comparison.json", result / "deployment.json", result / "report.md"]
    for kind in ("v00", "v01", "v02"):
        required += [result / kind / "best_head.pt", result / kind / "study_predictions.csv",
                     result / kind / "metrics.jsonl", result / kind / "summary.json"]
    if any(not file.is_file() for file in required):
        raise FileNotFoundError("Incomplete Phase 12 outputs")
    source = root / cfg["source"]["m02_run"] / "best_model.pt"
    if sha(source) != cfg["source"]["checkpoint_sha256"]:
        raise ValueError("Source M02 checkpoint changed")
    files = [p for directory in (cache, result) for p in directory.rglob("*") if p.is_file()]
    manifest = {"source_checkpoint_sha256": sha(source), "config_sha256": sha(args.config),
                "files": {str(path.relative_to(root)).replace("\\", "/"): {"bytes": path.stat().st_size,
                          "sha256": sha(path)} for path in files}}
    target = result / "frozen_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [*files, target]:
        os.chmod(path, stat.S_IREAD)
    print(json.dumps({"file_count": len(files), "manifest": str(target)}))


if __name__ == "__main__":
    main()
