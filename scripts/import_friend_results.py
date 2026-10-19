"""Verify a directly transferred result folder, then copy to an independent import path."""
import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

from cxr.batch.direct_delivery import verify_direct_folder
from cxr.batch.plan import ROOT, digest


def import_result(source, destination, plan, expected_commit):
    source,destination=Path(source),Path(destination)
    manifest=verify_direct_folder(source,expected_commit)
    if manifest['fold_hashes']!=plan['fold_hashes'] or manifest['data_sha256']!=plan['data_sha256']:
        raise ValueError('Data/fold version mismatch')
    if manifest['code_sha256']!=plan['artifacts']:
        raise ValueError('Source code artifact hash mismatch')
    expected={task['id']:task for task in plan['tasks']}
    if not set(manifest['completed_tasks'])<=set(expected):
        raise ValueError('Unknown completed model ID')
    for task_id in manifest['completed_tasks']:
        run=source/'experiments'/task_id
        if digest(run/'resolved_config.yaml')!=expected[task_id]['config_sha256']:
            raise ValueError(f'Unexpected training config: {task_id}')
        record=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        if record['model_id']!=task_id or record['fold_hashes']!=plan['fold_hashes'] or record['git_commit']!=expected_commit:
            raise ValueError('Model provenance differs')
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source,destination)
    verify_direct_folder(destination,expected_commit)
    (destination/'IMPORT_VERIFIED.json').write_text(json.dumps({'commit':expected_commit,
        'source_manifest_sha256':digest(source/'manifests/delivery_manifest.json'),
        'completed_tasks':manifest['completed_tasks']},indent=2),encoding='utf-8')
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--version',required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.version):
        parser.error('Safe unique version name required')
    plan=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    print(import_result(args.source,ROOT/'outputs/phase17_friend_import'/args.version,plan,args.expected_commit))


if __name__=='__main__':
    main()
