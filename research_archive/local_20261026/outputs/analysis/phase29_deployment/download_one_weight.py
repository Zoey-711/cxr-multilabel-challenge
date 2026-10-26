"""Download only H17-02's raw inference weight from the user's private Release."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
GH = HERE / "tools/bin/gh.exe"
DEST = HERE / "weights/H17_DISTILL_02_best_model.pth"
PARTIAL = DEST.with_suffix(".partial")
SHA = "35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
BYTES = 81651999

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    if DEST.is_file():
        if DEST.stat().st_size==BYTES and digest(DEST)==SHA:
            print(json.dumps({"status":"VERIFIED_LOCAL","bytes":BYTES,"sha256":SHA}))
            return
        raise RuntimeError("Existing H17 weight mismatches frozen SHA; refusing overwrite")
    if not GH.is_file():
        raise RuntimeError("GitHub CLI portable binary missing")
    if PARTIAL.exists():
        raise RuntimeError("Partial download exists; resolve or resume it before starting another download")
    DEST.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([str(GH),"release","download","phase17-results-20261024",
        "--repo","Zoey-711/cxr-multilabel-challenge",
        "--pattern","experiments--H17_DISTILL_02--best_model.pth",
        "--output",str(PARTIAL)],check=False)
    if result.returncode:
        raise RuntimeError("GitHub Release download failed; partial file retained for inspection")
    if PARTIAL.stat().st_size!=BYTES or digest(PARTIAL)!=SHA:
        raise RuntimeError("H17 weight SHA/size mismatch; partial file retained")
    PARTIAL.replace(DEST)
    print(json.dumps({"status":"DOWNLOADED_VERIFIED","bytes":BYTES,"sha256":SHA}))

if __name__=="__main__":main()
