"""Explicit opt-in batch queue; default command only validates the plan."""
import argparse
import json
import os
from pathlib import Path

from cxr.batch.plan import load_plan, gpu_tokens, digest, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['validate','run','status','stop','backup','restore','report'],default='validate')
    p.add_argument('--root',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--gpus',default='0')
    p.add_argument('--gpu-hours',type=float)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-extended',action='store_true')
    a=p.parse_args(); plan=load_plan(a.config)
    if a.action=='validate':
        print(json.dumps({'valid':True,'base_tasks':15,'optional_seed_tasks':2,'new_architectures':0}))
        return
    if a.root is None or not a.root.is_absolute():
        p.error('--root must be an absolute, independent output directory')
    if a.action=='run':
        if a.root.name!='phase17_batch':
            p.error('Unified output directory must be named phase17_batch')
        if a.gpu_hours is None or not a.receipt:
            p.error('Explicit friend-approved --gpu-hours and --receipt required')
        devices=gpu_tokens(a.gpus)
        backup_root=os.environ.get('CXR_BACKUP_ROOT')
        if not backup_root or not Path(backup_root).is_absolute():
            p.error('Set absolute CXR_BACKUP_ROOT to a friend-authorized backup location')
        if a.root.resolve()==Path(backup_root).resolve() or a.root.resolve() in Path(backup_root).resolve().parents:
            p.error('Backup must be outside the run directory')
        receipt=json.loads(a.receipt.read_text())
        if receipt.get('host')!=__import__('socket').gethostname():
            p.error('Preflight receipt belongs to a different host; rerun preflight')
        if receipt['plan_sha256']!=digest(a.config) or receipt['state']!='passed' or not set(devices)<=set(receipt['gpus']):
            p.error('Preflight receipt does not cover this plan/GPU pool')
        for key in ('CXR_DATA_ROOT','CXR_PRETRAINED_ROOT'):
            if receipt['environment'].get(key)!=os.environ.get(key):
                p.error(f'{key} differs from preflight')
        from cxr.batch.runner import run_queue
        run_queue(plan,a.config,a.root,devices,a.gpu_hours,a.resume,a.allow_extended,receipt.get('gpu_hours',0))
    elif a.action=='status':
        print((a.root/'queue_state.json').read_text(encoding='utf-8'))
    elif a.action=='stop':
        if not (a.root/'queue_state.json').exists():
            p.error('No existing queue at --root')
        (a.root/'STOP').touch(exist_ok=True)
        print('Drain requested: active tasks finish; no new tasks start. Remove STOP explicitly before resume.')
    elif a.action=='report':
        from cxr.batch.reporting import report
        report(a.root)
    else:
        if not a.destination:
            p.error('--destination required')
        from cxr.batch.backup import backup,restore
        print((backup if a.action=='backup' else restore)(a.root,a.destination))


if __name__=='__main__':
    main()
