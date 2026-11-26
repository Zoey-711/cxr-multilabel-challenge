"""Verify packaged assets without importing model libraries or modifying files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    for candidate in sorted((root / "candidates").iterdir()):
        if not candidate.is_dir():
            continue
        manifest = json.loads((candidate / "manifest_sha256.json").read_text(encoding="utf-8"))
        checked = 0
        for relative, record in manifest["files"].items():
            path = (candidate / relative).resolve()
            if not path.is_relative_to(candidate.resolve()) or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unsafe/missing asset: {relative}")
            if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                raise ValueError(f"Size/hash mismatch: {candidate.name}/{relative}")
            checked += 1
        actual = {str(path.relative_to(candidate)).replace("\\", "/") for path in candidate.rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                  and path.name != "manifest_sha256.json"}
        if actual != set(manifest["files"]):
            raise ValueError(f"Unlisted or missing candidate files: {candidate.name}")
        model_bytes = sum(p.stat().st_size for p in candidate.rglob("*") if p.suffix.lower() in {".pt", ".pth", ".onnx"})
        config_bytes = sum(p.stat().st_size for p in candidate.rglob("*.json") if p.name != "manifest_sha256.json")
        total = model_bytes + config_bytes
        if total != manifest["actual_model_and_config_bytes"]:
            raise ValueError(f"Deployment size changed: {candidate.name}")
        results[candidate.name] = {"files_verified": checked, "model_and_config_bytes": total,
                                   "under_500000000_bytes": total <= 500000000}
    if len(results) != 4:
        raise ValueError("Expected four independent candidates")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
