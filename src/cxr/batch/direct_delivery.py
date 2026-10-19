"""Direct-folder completion index; no archive creation or deletion."""
import json
from pathlib import Path

from cxr.batch.plan import digest, write_json


def write_delivery_manifest(root):
    root=Path(root)
    environment=json.loads((root/'manifests/environment.json').read_text(encoding='utf-8'))
    plan=json.loads((root/'reports/experiment_summary.json').read_text(encoding='utf-8'))
    tasks=plan['tasks']
    complete=[r['model_id'] for r in tasks if r['state']=='complete']
    files={}
    for relative in ('reports/experiment_summary.json','reports/phase17_final_report.md',
                     'submissions/submission_candidates.csv','submissions/submission_validation_report.csv'):
        path=root/relative
        if path.is_file():files[relative]={'bytes':path.stat().st_size,'sha256':digest(path)}
    manifest={'format':'phase17_direct_folder_v1','git_commit':environment['git_commit'],
              'data_sha256':environment['data_sha256'],'fold_hashes':environment['fold_hashes'],
              'code_sha256':environment['code_sha256'],'completed_tasks':complete,
              'missing_tasks':[r['model_id'] for r in tasks if r['state']!='complete'],
              'model_count':len(complete),'important_files':files,
              'no_archives':True,'source_root':'relative paths only'}
    write_json(root/'manifests/delivery_manifest.json',manifest)
    (root/'reports/COMPLETION.txt').write_text(
        f'Complete models: {len(complete)}/{len(tasks)}\nDirect folder: {root}\n'
        'Verify manifests/delivery_manifest.json and each experiments/<ID>/batch_frozen_manifest.json.\n',
        encoding='utf-8')
    return manifest


def verify_direct_folder(root, expected_commit=None):
    root=Path(root)
    manifest=json.loads((root/'manifests/delivery_manifest.json').read_text(encoding='utf-8'))
    if expected_commit is not None and manifest['git_commit']!=expected_commit:
        raise ValueError('Source commit differs')
    for name,info in manifest['important_files'].items():
        path=root/name
        if path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:
            raise ValueError(f'Changed direct delivery artifact: {name}')
    from cxr.batch.results import verify_frozen
    for task in manifest['completed_tasks']:
        verify_frozen(root/'experiments'/task)
    return manifest
