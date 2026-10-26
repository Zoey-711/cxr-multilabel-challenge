"""Whitelist code, fixed metadata and optional legal ImageNet weights; never raw images."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from cxr.batch.plan import ROOT,load_plan,digest,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-pretrained',action='store_true');a=p.parse_args();plan=load_plan(a.config)
    if a.output.exists():raise FileExistsError(a.output)
    files=set()
    for base,pattern in [('src','*.py'),('configs/batch17','*.yaml'),('batch_assets','*')]:
        files.update(x for x in (ROOT/base).rglob(pattern) if x.is_file())
    files.update(ROOT/x for x in plan['artifacts'])
    files.add(a.config.resolve())
    for name in ('README_friend.md','README_results.md','README.md','pyproject.toml','requirements/cloud.txt','scripts/cloud_install.sh',
                 'scripts/analyze_validation.py','scripts/friend_start.sh','scripts/friend_slurm.sh','scripts/prepare_friend_bundle.py',
                 'docs/batch17_costs.json','docs/batch17_config_differences.json'):
        files.add(ROOT/name)
    files.update((ROOT/'scripts').glob('batch_*.py'))
    files.update((ROOT/'tests').glob('test_batch*.py'))
    files.add(ROOT/'tests/test_training_resume.py')
    files.update(ROOT/plan['fold_directory']/name for name in plan['fold_hashes'])
    if a.include_pretrained:
        from cxr.models.factory import imagenet_weight_info
        from cxr.utils.cloud_paths import pretrained_home
        for task in plan['tasks']:
            _,_,name=imagenet_weight_info(task['backbone'])
            file=pretrained_home(ROOT)/'hub/checkpoints'/name
            if digest(file)!=task['pretrained_sha256']:raise ValueError('Pretrained hash mismatch')
            files.add(file)
    entries={}
    for file in files:
        relative=file.relative_to(ROOT).as_posix()
        if file.is_symlink() or file.suffix.lower() in ('.jpg','.png','.zip','.key','.pem') or '.tmp/' in relative:
            raise ValueError(f'Unapproved delivery file: {relative}')
        entries[relative]={'sha256':digest(file),'bytes':file.stat().st_size}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'w:gz') as tar:
        for relative in sorted(entries):tar.add(ROOT/relative,arcname=relative,recursive=False)
    with tarfile.open(a.output,'r:gz') as tar:
        for item in tar:
            if hashlib.file_digest(tar.extractfile(item),'sha256').hexdigest()!=entries[item.name]['sha256']:
                raise ValueError('Code bundle verification failed')
    write_json(a.output.with_suffix('.manifest.json'),{'sha256':digest(a.output),'bytes':a.output.stat().st_size,'files':entries,
               'private_only':True,'contains_raw_images':False,'contains_fixed_training_label_metadata':True})
    print(json.dumps({'path':str(a.output),'bytes':a.output.stat().st_size,'sha256':digest(a.output)},indent=2))
if __name__=='__main__':main()
