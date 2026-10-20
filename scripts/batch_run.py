"""Explicit opt-in batch queue; default command only validates the plan."""
import argparse
import json
import os
from pathlib import Path

from cxr.batch.plan import load_plan, gpu_tokens, digest, ROOT


def validate_remote_receipt(receipt, plan_sha256, devices, max_concurrent, environment, host):
    if receipt.get('host') != host:
        raise ValueError('REMOTE_PREFLIGHT belongs to a different host')
    if (receipt.get('kind') != 'REMOTE_PREFLIGHT' or receipt.get('plan_sha256') != plan_sha256
            or receipt.get('state') != 'passed' or len(receipt.get('cuda_tests', [])) != 3
            or receipt.get('real_data_smoke') != 'passed'
            or 'val' not in receipt.get('real_inference', {})
            or not set(devices) <= set(receipt.get('gpus', []))):
        raise ValueError('REMOTE_PREFLIGHT does not cover the plan, data, CUDA tests and GPU pool')
    stages = {stage.get('concurrent_gpus') for stage in receipt.get('stages', [])
              if stage.get('state') == 'passed'}
    if not set((1, min(2, max_concurrent), min(4, max_concurrent), max_concurrent)) <= stages:
        raise ValueError('Staged GPU concurrency checks are incomplete')
    if max_concurrent > receipt.get('max_concurrent', 0):
        raise ValueError('Concurrency exceeds measured host-resource preflight')
    for key in ('CXR_DATA_ROOT', 'CXR_PRETRAINED_ROOT', 'CXR_BACKUP_ROOT', 'CXR_OUTPUT_ROOT'):
        if receipt.get('environment', {}).get(key) != environment.get(key):
            raise ValueError(f'{key} differs from REMOTE_PREFLIGHT')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['validate','run','run-extension','status','stop','clear-stop','backup','restore','report'],default='validate')
    p.add_argument('--root',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--gpus',default='0')
    p.add_argument('--max-concurrent',type=int,default=1)
    p.add_argument('--gpu-hours',type=float)
    p.add_argument('--gh-gpu-hours',type=float)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-extended',action='store_true')
    a=p.parse_args(); plan=load_plan(a.config)
    if a.action=='validate':
        print(json.dumps({'valid':True,'base_tasks':len([t for t in plan['tasks'] if t['enabled']]),
                          'conditional_slots':plan.get('conditional_extension',{}).get('max_additional_tasks',0),
                          'conditional_enabled':False,'formal_run_requires_preflight':True}))
        return
    if a.root is None or not a.root.is_absolute():
        p.error('--root must be an absolute, independent output directory')
    if a.action in ('run','run-extension'):
        if a.root.name!='phase17_batch':
            p.error('Unified output directory must be named phase17_batch')
        if a.gpu_hours is None or not a.receipt:
            p.error('Explicit friend-approved --gpu-hours and --receipt required')
        devices=gpu_tokens(a.gpus)
        if not 1 <= a.max_concurrent <= min(len(devices),8):
            p.error('max-concurrent must be 1..min(authorized GPU count,8)')
        backup_root=os.environ.get('CXR_BACKUP_ROOT')
        if not backup_root or not Path(backup_root).is_absolute():
            p.error('Set absolute CXR_BACKUP_ROOT to a friend-authorized backup location')
        if a.root.resolve()==Path(backup_root).resolve() or a.root.resolve() in Path(backup_root).resolve().parents:
            p.error('Backup must be outside the run directory')
        receipt=json.loads(a.receipt.read_text())
        try:
            validate_remote_receipt(receipt, digest(a.config), devices, a.max_concurrent,
                                    os.environ, __import__('socket').gethostname())
        except ValueError as error:
            p.error(str(error))
        if a.action=='run':
            if a.allow_extended:
                p.error('Use the separate run-extension action after all 22 base tasks')
            from cxr.batch.runner import run_queue
            run_queue(plan,a.config,a.root,devices,a.gpu_hours,a.resume,False,receipt.get('gpu_hours',0),a.max_concurrent)
        else:
            if not a.allow_extended or a.gh_gpu_hours is None or a.gh_gpu_hours<=0:
                p.error('G/H requires --allow-extended and a separate positive --gh-gpu-hours cap')
            from cxr.batch.extension_queue import run_extension_queue
            run_extension_queue(a.root,plan,devices,total_budget=a.gpu_hours,
                                gh_budget=a.gh_gpu_hours,max_concurrent=a.max_concurrent,
                                resume=a.resume)
    elif a.action=='status':
        print((a.root/'queue_state.json').read_text(encoding='utf-8'))
    elif a.action=='stop':
        if not (a.root/'queue_state.json').exists():
            p.error('No existing queue at --root')
        (a.root/'STOP').touch(exist_ok=True)
        print('Drain requested: active tasks finish; no new tasks start. Remove STOP explicitly before resume.')
    elif a.action=='clear-stop':
        if (a.root/'queue.lock').exists():
            p.error('Queue lock exists; confirm the former queue process has exited before clearing STOP')
        if not (a.root/'queue_state.json').exists():
            p.error('No queue state to resume')
        (a.root/'STOP').unlink(missing_ok=True)
        print('STOP cleared; rerun the same friend_start.sh command to resume')
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
