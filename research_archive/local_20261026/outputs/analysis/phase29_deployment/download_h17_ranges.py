"""Parallel resumable fetch of one authorized H17-02 Release asset; never print token/URL."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
GH=HERE/"tools/bin/gh.exe"
OUT=HERE/"weights/H17_DISTILL_02_best_model.pth"
CHUNKS=HERE/"weights/h17_chunks"
EXPECTED_SIZE=81651999
EXPECTED_SHA="35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
PARTS=8

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def signed_url():
    token=subprocess.run([str(GH),"auth","token"],capture_output=True,text=True,check=True).stdout.strip()
    metadata=subprocess.run([str(GH),"release","view","phase17-results-20261024","--repo",
        "Zoey-711/cxr-multilabel-challenge","--json","assets"],capture_output=True,text=True,check=True)
    matching=[a for a in json.loads(metadata.stdout)["assets"] if a["name"]=="experiments--H17_DISTILL_02--best_model.pth"]
    if len(matching)!=1 or matching[0]["size"]!=EXPECTED_SIZE or matching[0].get("digest")!="sha256:"+EXPECTED_SHA:
        raise ValueError("Release asset metadata does not match frozen manifest")
    req=urllib.request.Request(matching[0]["apiUrl"],headers={"Authorization":f"Bearer {token}",
        "Accept":"application/octet-stream","X-GitHub-Api-Version":"2022-11-28","User-Agent":"phase29-audit"})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    try:
        urllib.request.build_opener(NoRedirect).open(req,timeout=30)
    except urllib.error.HTTPError as error:
        if error.code!=302:raise
        url=error.headers.get("Location")
        if not url:raise RuntimeError("Missing authorized asset redirect")
        return url
    raise RuntimeError("Expected private asset redirect")

def fetch(index,start,end,url):
    path=CHUNKS/f"part{index:02d}"
    expected=end-start+1
    for retry in range(5):
        existing=path.stat().st_size if path.exists() else 0
        if existing==expected:return (index,expected)
        if existing>expected:raise ValueError("Range chunk exceeds expected length")
        req=urllib.request.Request(url,headers={"Range":f"bytes={start+existing}-{end}"})
        try:
            with urllib.request.urlopen(req,timeout=90) as reply:
                if reply.status!=206 or not reply.headers.get("Content-Range","").startswith(f"bytes {start+existing}-{end}/"):
                    raise RuntimeError("CDN did not honor exact byte range")
                with path.open("ab") as out:
                    while True:
                        block=reply.read(1024*1024)
                        if not block:break
                        out.write(block)
        except (OSError,TimeoutError) as exc:
            if retry==4:raise RuntimeError(f"part {index} download failed after retries") from exc
            time.sleep(2)
    if path.stat().st_size!=expected:raise ValueError(f"Incomplete range part {index}")
    return (index,expected)

def main():
    if OUT.is_file():
        if OUT.stat().st_size==EXPECTED_SIZE and sha(OUT)==EXPECTED_SHA:
            print(json.dumps({"status":"VERIFIED_LOCAL","bytes":EXPECTED_SIZE,"sha256":EXPECTED_SHA}))
            return
        raise ValueError("Existing H17 weight mismatches; refusing overwrite")
    CHUNKS.mkdir(parents=True,exist_ok=True)
    url=signed_url()
    spans=[(i,i*EXPECTED_SIZE//PARTS,(i+1)*EXPECTED_SIZE//PARTS-1) for i in range(PARTS)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARTS) as pool:
        futures=[pool.submit(fetch,i,start,end,url) for i,start,end in spans]
        for future in concurrent.futures.as_completed(futures):
            index,length=future.result()
            print(json.dumps({"completed_part":index,"bytes":length}),flush=True)
    assembled=OUT.with_suffix(".assembled")
    with assembled.open("wb") as dest:
        for i in range(PARTS):
            with (CHUNKS/f"part{i:02d}").open("rb") as source:
                for block in iter(lambda:source.read(1024*1024),b""):dest.write(block)
    if assembled.stat().st_size!=EXPECTED_SIZE or sha(assembled)!=EXPECTED_SHA:
        raise ValueError("Assembled H17 weight failed frozen size/SHA-256 check")
    assembled.replace(OUT)
    print(json.dumps({"status":"DOWNLOADED_VERIFIED","bytes":EXPECTED_SIZE,"sha256":EXPECTED_SHA}),flush=True)

if __name__=="__main__":main()
