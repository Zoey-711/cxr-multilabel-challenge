"""Create a small, path-sanitized Phase17 code/index bundle; no data or weights."""

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cxr.utils.config import load_config


def bundle_entries(root: Path) -> dict[str, bytes]:
    files = [root / "pyproject.toml", root / "README.md", root / "README_cloud.md",
             root / ".gitignore", root / "requirements/cloud.txt"]
    files += [p for folder in ("src/cxr", "scripts", "tests") for p in (root / folder).rglob("*.py")]
    files += [root / "scripts/cloud_install.sh"]
    files += list((root / "configs").glob("*17*fold*.yaml"))
    if len([p for p in files if p.parent == root / "configs"]) != 15:
        raise ValueError("Expected exactly 15 frozen Phase17 configs")
    fold_dir = root / "outputs/analysis/phase17_patient_folds"
    files += [fold_dir / name for name in ("patient_folds.csv", "study_folds.csv", "image_folds.csv",
                                              "fold_report.json", "fold_sha256.json", "config_audit.json",
                                              "joint_config_audit.json")
              if (fold_dir / name).is_file()]
    result = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = path.relative_to(root).as_posix()
        content = path.read_bytes()
        if rel.startswith("configs/"):
            source = content.decode("utf-8")
            clean, count = re.subn(r"(?m)^  archive_path:.*$", "  archive_path: null", source)
            if count != 1:
                raise ValueError(f"Cannot sanitize archive path in {rel}")
            before, after = load_config(path), __import__("yaml").safe_load(clean)
            before["data"]["archive_path"] = None
            if before != after:
                raise ValueError(f"Unexpected protocol change in sanitized {rel}")
            content = clean.encode("utf-8")
        if rel.endswith("/fold_report.json"):
            report = json.loads(content.decode("utf-8"))
            report["source_archive"] = None
            content = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        if rel in result:
            raise ValueError(f"Duplicate bundle path: {rel}")
        result[rel] = content
    if not {"outputs/analysis/phase17_patient_folds/" + x for x in
            ("patient_folds.csv", "study_folds.csv", "image_folds.csv")}.issubset(result):
        raise FileNotFoundError("Frozen patient fold CSVs missing")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="A frozen Phase17 config")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve().parent != (root / "configs").resolve():
        parser.error("Use one of the local Phase17 configs")
    entries = bundle_entries(root)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_config": args.config.name,
                "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                          for name, content in sorted(entries.items())},
                "omitted": "Original data, Test, checkpoints, historical outputs, caches, secrets"}
    entries["cloud_bundle_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(name, content)
    print(json.dumps({"bundle": str(output), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "file_count": len(entries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
