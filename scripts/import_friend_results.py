"""Verify a separately received delivery against approved code/config/fold identity."""
import argparse,json
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,digest
from cxr.batch.delivery import verify_and_extract

def import_result(bundle,destination,plan,expected_commit):
    manifest=json.loads((bundle/'delivery_manifest.json').read_text(encoding='utf-8'))
    if manifest['source'].get('git_commit')!=expected_commit:
        raise ValueError('Result code commit differs from the explicitly approved commit')
    if manifest['fold_hashes']!=plan['fold_hashes'] or manifest['data_sha256']!=plan['data_sha256']:
        raise ValueError('Data/fold version mismatch')
    if manifest['source']['code_sha256']!=plan['artifacts']:
        raise ValueError('Result source file hashes differ from approved plan')
    expected={t['id']:t for t in plan['tasks']}
    if not set(manifest['completed_tasks'])<=set(expected):raise ValueError('Unknown model ID')
    target=verify_and_extract(bundle,destination)
    for tid in manifest['completed_tasks']:
        run=target/'phase17_batch/experiments'/tid
        if digest(run/'resolved_config.yaml')!=expected[tid]['config_sha256']:
            raise ValueError(f'Unexpected actual training config: {tid}')
        info=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        if info['model_id']!=tid or info['fold_hashes']!=plan['fold_hashes']:
            raise ValueError('Model/fold provenance differs')
    (target/'IMPORT_VERIFIED.json').write_text(json.dumps({'commit':expected_commit,'manifest_sha256':digest(bundle/'delivery_manifest.json'),'completed_tasks':manifest['completed_tasks']},indent=2),encoding='utf-8')
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--version',required=True);p.add_argument('--expected-commit',required=True)
    a=p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.version):p.error('Safe unique version name required')
    plan=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    target=ROOT/'outputs/phase17_friend_import'/a.version
    print(import_result(a.bundle,target,plan,a.expected_commit))
if __name__=='__main__':main()
