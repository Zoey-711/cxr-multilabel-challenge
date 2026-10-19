"""Maintainer-only: seal reviewed code and configs before committing; never called by training."""
import argparse
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    plan=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    files=set((ROOT/'src').rglob('*.py'))
    files.update((ROOT/'scripts').glob('*.py'))
    files.update((ROOT/'scripts').glob('*.sh'))
    files.update((ROOT/'tests').glob('*.py'))
    files.update((ROOT/'configs/batch17').glob('*.yaml'))
    files.update(x for x in (ROOT/'batch_assets').iterdir() if x.is_file())
    files.update(ROOT/x for x in ('README.md','README_friend.md','README_results.md','requirements/cloud.txt','pyproject.toml','docs/batch17_costs.json'))
    plan['artifacts']={f.relative_to(ROOT).as_posix():digest(f) for f in sorted(files)}
    for t in plan['tasks']:t['config_sha256']=digest(ROOT/t['config'])
    a.config.write_text(yaml.safe_dump(plan,sort_keys=False,allow_unicode=True),encoding='utf-8',newline='\n')
    print('Reviewed plan SHA256:',digest(a.config))
if __name__=='__main__':main()
