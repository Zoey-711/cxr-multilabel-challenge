"""Read-only staged-file inspection; never pushes or uploads."""
import argparse,json,re,subprocess
from pathlib import Path
from cxr.batch.plan import ROOT,write_json

def audit():
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    issues=[];sizes={}
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[A-Z0-9]{16})')
    for name in filter(None,files):
        p=ROOT/name;sizes[name]=p.stat().st_size
        if p.stat().st_size>2_000_000:issues.append({'path':name,'reason':'file exceeds 2MB code-only limit'})
        if name.startswith(('batch_assets/','outputs/','data/','.venv/','.wheels/','friend_bundles/','private_transfer/')) or p.suffix.lower() in ('.pt','.pth','.ckpt','.zip','.jpg','.png','.tar','.gz','.key','.pem'):
            issues.append({'path':name,'reason':'data/model/archive/secret path staged'})
        text=p.read_text(encoding='utf-8',errors='replace')
        if secret.search(text):issues.append({'path':name,'reason':'credential-like content; value not printed'})
        if re.search(r'[EF]:[/\\]',text):issues.append({'path':name,'reason':'personal E/F absolute path'})
    return {'passed':not issues,'issues':issues,'file_count':len(sizes),'total_bytes':sum(sizes.values()),'largest_files':sorted(sizes.items(),key=lambda x:x[1],reverse=True)[:10]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=audit();write_json(a.output,result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
