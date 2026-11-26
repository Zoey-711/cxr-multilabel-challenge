"""Resume exactly four approved Private Release assets; never print a token."""
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
GH = WORKSPACE / "cxr-multilabel-challenge/outputs/analysis/phase29_deployment/tools/bin/gh.exe"
API = json.loads((HERE / "release_api.json").read_text(encoding="utf-8-sig"))
MANIFEST = json.loads((HERE / "binary_release_manifest.json").read_text(encoding="utf-8-sig"))
ALLOWED = {f"phase32_batch--experiments--P32_NS_{variant}--{kind}best_model.pth"
           for variant in ("POS", "512") for kind in ("", "ema_")}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    token = subprocess.check_output([str(GH), "auth", "token"], text=True).strip()
    assets = {a["name"]: a for a in API["assets"]}
    metadata = {a["asset_name"]: a for a in MANIFEST["files"]}
    folder = HERE / "fold1_weights"
    folder.mkdir(exist_ok=True)
    for name in sorted(ALLOWED):
        asset, meta = assets[name], metadata[name]
        path = folder / name
        current = path.stat().st_size if path.exists() else 0
        if current > meta["bytes"]:
            raise ValueError(f"Local partial exceeds expected size: {name}")
        if current == meta["bytes"]:
            if digest(path) != meta["sha256"]:
                raise ValueError(f"Complete file has wrong SHA: {name}")
            print(name, "ALREADY_VERIFIED")
            continue
        req = urllib.request.Request(
            asset["apiUrl"],
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/octet-stream",
                     "Range": f"bytes={current}-", "X-GitHub-Api-Version": "2022-11-28"},
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            print(name, "HTTP", response.status, "resume_at", current, flush=True)
            if response.status != 206:
                raise RuntimeError("Server did not honor Range; partial file preserved")
            header = response.headers.get("Content-Range", "")
            if not header.startswith(f"bytes {current}-"):
                raise RuntimeError("Unexpected Content-Range; partial file preserved")
            with path.open("ab") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if path.stat().st_size != meta["bytes"] or digest(path) != meta["sha256"]:
            raise ValueError(f"Downloaded weight failed SHA/size: {name}")
        print(name, "SHA256_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
