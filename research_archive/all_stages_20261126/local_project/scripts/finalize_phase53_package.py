"""Hash, archive and extract-verify the immutable Phase53 T4 benchmark package."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/deployment/phase53_t4_benchmark"
ARCHIVE = PACKAGE.with_suffix(".tar.gz")

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def safe_files(folder):
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink prohibited: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path

def write_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)

def verify_root(folder):
    manifest = json.loads((folder / "manifests/package_sha256.json").read_text(encoding="utf-8"))
    actual = {str(path.relative_to(folder)).replace("\\", "/"):path for path in safe_files(folder)
              if path.name != "package_sha256.json"}
    if set(actual) != set(manifest["files"]):
        raise ValueError("Extracted package file list differs")
    for relative, record in manifest["files"].items():
        path = actual[relative]
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Extracted hash mismatch: {relative}")
    return len(actual)

def main():
    prohibited = {".jpg", ".jpeg", ".png", ".dcm", ".zip", ".tar", ".gz"}
    if any(path.suffix.lower() in prohibited for path in safe_files(PACKAGE)):
        raise ValueError("Raw image/dataset or nested archive found in package")
    for candidate in sorted((PACKAGE / "candidates").iterdir()):
        if not candidate.is_dir():
            continue
        files = {str(path.relative_to(candidate)).replace("\\", "/"):
                 {"bytes":path.stat().st_size, "sha256":sha(path)}
                 for path in safe_files(candidate) if path.name != "manifest_sha256.json"}
        model_bytes = sum(path.stat().st_size for path in candidate.rglob("*")
                          if path.suffix.lower() in {".pt", ".pth", ".onnx"})
        config_bytes = sum(path.stat().st_size for path in candidate.rglob("*.json")
                           if path.name != "manifest_sha256.json")
        write_atomic(candidate / "manifest_sha256.json",
                     {"candidate":candidate.name, "files":files,
                      "actual_model_and_config_bytes":model_bytes + config_bytes,
                      "under_500000000_bytes":model_bytes + config_bytes <= 500000000,
                      "t4_latency":"T4_UNVERIFIED"})
    package_files = {str(path.relative_to(PACKAGE)).replace("\\", "/"):
                     {"bytes":path.stat().st_size, "sha256":sha(path)}
                     for path in safe_files(PACKAGE) if path.name != "package_sha256.json"}
    write_atomic(PACKAGE / "manifests/package_sha256.json", {"files":package_files})
    verify_root(PACKAGE)
    subprocess.run([sys.executable, str(PACKAGE / "verify_packages.py")], check=True)
    if not ARCHIVE.exists():
        with tarfile.open(ARCHIVE, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tar:
            for path in safe_files(PACKAGE):
                relative = path.relative_to(PACKAGE)
                tar.add(path, arcname=str(Path(PACKAGE.name) / relative).replace("\\", "/"), recursive=False)
    with tempfile.TemporaryDirectory(prefix="phase53_extract_", dir=PACKAGE.parent) as temporary:
        target = Path(temporary).resolve()
        if not target.is_relative_to(PACKAGE.parent.resolve()):
            raise ValueError("Unsafe extraction target")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            for member in tar:
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts or not member.isfile():
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(target, filter="data")
        extracted = target / PACKAGE.name
        if sha(PACKAGE / "manifests/package_sha256.json") != sha(extracted / "manifests/package_sha256.json"):
            raise ValueError("Archive root manifest differs from current package")
        checked = verify_root(extracted)
        subprocess.run([sys.executable, str(extracted / "verify_packages.py"), "--root", str(extracted)], check=True)
    summary = {"archive":str(ARCHIVE), "archive_bytes":ARCHIVE.stat().st_size,
               "archive_sha256":sha(ARCHIVE), "extracted_files_verified":checked,
               "dataset_images_included":False, "t4_status":"T4_UNVERIFIED"}
    write_atomic(PACKAGE.parent / "phase53_t4_benchmark_archive.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
