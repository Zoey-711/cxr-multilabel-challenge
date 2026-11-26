"""Inspect range support for one private Release asset without printing auth URLs."""
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlparse

HERE=Path(__file__).resolve().parent
GH=HERE/"tools/bin/gh.exe"
token=subprocess.run([str(GH),"auth","token"],capture_output=True,text=True,check=True).stdout.strip()
meta=subprocess.run([str(GH),"release","view","phase17-results-20261024","--repo",
    "Zoey-711/cxr-multilabel-challenge","--json","assets"],capture_output=True,text=True,check=True)
assets=json.loads(meta.stdout)["assets"]
asset=next(a for a in assets if a["name"]=="experiments--H17_DISTILL_02--best_model.pth")
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
opener=urllib.request.build_opener(NoRedirect)
req=urllib.request.Request(asset["apiUrl"],headers={"Authorization":f"Bearer {token}",
    "Accept":"application/octet-stream","X-GitHub-Api-Version":"2022-11-28","User-Agent":"phase29-audit"})
try:
    response=opener.open(req,timeout=20)
except urllib.error.HTTPError as error:
    response=error
redirect=response.headers.get("Location")
print(json.dumps({"api_status":response.status,"redirect_host":urlparse(redirect).hostname if redirect else None,
                  "asset_bytes":asset["size"],"api_digest":asset.get("digest")}))
if redirect:
    req2=urllib.request.Request(redirect,method="HEAD")
    with urllib.request.urlopen(req2,timeout=20) as head:
        print(json.dumps({"cdn_status":head.status,"content_length":head.headers.get("Content-Length"),
                          "accept_ranges":head.headers.get("Accept-Ranges")}))
