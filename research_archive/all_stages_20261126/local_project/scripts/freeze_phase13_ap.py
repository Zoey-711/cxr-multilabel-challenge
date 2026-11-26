"""Hash and freeze the tie-aware AP supplement without editing frozen experiments."""

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
    output = root / "outputs/analysis/phase13_ap_tieaware"
    target = output / "manifest.json"
    if target.exists():
        raise FileExistsError(target)
    files = [output / "metrics.json", output / "report.md"]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Incomplete AP audit")
    v03 = root / "outputs/experiments/v03_m02_multiview_residual/frozen_manifest.json"
    phase12 = root / cfg["source"]["phase12_run"] / "frozen_manifest.json"
    manifest = {"v03_manifest_sha256": sha(v03), "phase12_manifest_sha256": sha(phase12),
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files}}
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in [*files, target]:
        os.chmod(path, stat.S_IREAD)
    print(json.dumps({"manifest": str(target)}))


if __name__ == "__main__":
    main()
