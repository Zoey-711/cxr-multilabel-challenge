"""Separately transfer fixed private protocol metadata, never through the code repository."""
import argparse,hashlib,tarfile
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,digest

def expected_assets(plan):
    assets={k:v for k,v in plan['artifacts'].items() if k.startswith('batch_assets/')}
    assets.update({plan['fold_directory']+'/'+name:h for name,h in plan['fold_hashes'].items()})
    return assets

def unpack(path,plan):
    assets=expected_assets(plan);payload={}
    with tarfile.open(path,'r:gz') as tar:
        for item in tar:
            if item.name not in assets or not item.isfile() or item.name in payload:
                raise ValueError('Unapproved/duplicate protocol member')
            data=tar.extractfile(item).read()
            if hashlib.sha256(data).hexdigest()!=assets[item.name]:raise ValueError('Protocol metadata hash mismatch')
            payload[item.name]=data
    if set(payload)!=set(assets):raise ValueError('Protocol attachment incomplete')
    for name,data in payload.items():
        target=(ROOT/name).resolve()
        if ROOT not in target.parents:raise ValueError('Unsafe protocol path')
        if target.exists() and digest(target)!=assets[name]:raise FileExistsError(target)
    for name,data in payload.items():
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--action',required=True,choices=['pack','import']);p.add_argument('--file',required=True,type=Path)
    a=p.parse_args();plan=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    if a.action=='import':unpack(a.file,plan)
    else:
        if a.file.exists():raise FileExistsError(a.file)
        assets=expected_assets(plan)
        for name,h in assets.items():
            if digest(ROOT/name)!=h:raise ValueError('Changed protocol metadata')
        a.file.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(a.file,'w:gz') as tar:
            for name in assets:tar.add(ROOT/name,arcname=name,recursive=False)
        print('Private metadata archive SHA256:',digest(a.file))
    print('Protocol assets verified; no new patient split generated')
if __name__=='__main__':main()
