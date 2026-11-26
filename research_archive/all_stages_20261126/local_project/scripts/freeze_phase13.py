"""Freeze V03 and the no-training patient protocol with SHA-256 manifests."""

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
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = root / "outputs/experiments/v03_m02_multiview_residual"
    protocol = root / "outputs/analysis/phase13_internal_protocol"
    target = result / "frozen_manifest.json"
    if target.exists():
        raise FileExistsError(target)
    required = [result / name for name in ("config.yaml", "best_head.pt", "study_predictions.csv",
                                          "metrics.jsonl", "comparison.json", "deployment.json", "report.md")]
    required += [protocol / name for name in ("audit.json", "report.md", "class_support.csv")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Incomplete Phase 13 outputs")
    source = root / cfg["source"]["m02_run"] / "best_model.pt"
    if sha(source) != cfg["source"]["checkpoint_sha256"]:
        raise ValueError("M02 checkpoint changed")
    files = [p for directory in (result, protocol) for p in directory.rglob("*") if p.is_file()]
    manifest = {"m02_checkpoint_sha256": sha(source), "v03_config_sha256": sha(args.config),
                "phase12_manifest_sha256": sha(root / cfg["source"]["phase12_run"] / "frozen_manifest.json"),
                "files": {str(path.relative_to(root)).replace("\\", "/"): {"bytes": path.stat().st_size,
                          "sha256": sha(path)} for path in files}}
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [*files, target]:
        os.chmod(path, stat.S_IREAD)
    print(json.dumps({"files": len(files), "manifest": str(target)}))


if __name__ == "__main__":
    main()
