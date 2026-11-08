"""Reuse host lock, failure isolation and GPU-hour ledger from Phase42."""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from cxr.phase39_support import write_json,digest
from cxr.regularization_confirmation.support import (ROOT,read_config,task_specs,fingerprint,code_identity,
    git_commit,ensure_no_orphan,require_confirmation_gate)


def prepare(cfg):
    from cxr.regularization_confirmation.core import audit,seal
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    require_confirmation_gate(cfg)
    protocol=json.loads((ROOT/'configs/confirmation_protocol.json').read_text(encoding='utf-8'))
    identity=fingerprint(dict(server=cfg,protocol=protocol,source=code_identity(),commit=git_commit()))
    tasks=task_specs(cfg)
    for task in tasks:
        out=root/'experiments'/task['id'];out.mkdir(parents=True,exist_ok=True)
        resolved=dict(protocol,output=str(out),archive=cfg['data_root'],pretrained_root=cfg['pretrained_root'],
            protocol_root=cfg['protocol_root'],holdout_index=cfg['holdout_index'],fold=task['fold'],
            arms=['BASE',task['variant']],identity=identity,code_commit=git_commit(),source_hashes=code_identity(),max_vram_gib=cfg['max_vram_gib'])
        cp=root/'configs'/(task['id']+'.json');seal(cp,resolved);audit(resolved,cp,out)
    write_json(root/'plan.json',dict(identity=identity,stage=cfg['stage'],tasks=tasks,code_commit=git_commit(),training_runs=2*len(tasks)))
    return identity,tasks


def authorize(cfg,tasks):
    if sys.platform!='linux':raise RuntimeError('Formal remote queue requires Linux; local CPU checks only')
    for key in ('pair_gpu_hours','gpu_hours'):
        if not isinstance(cfg[key],(int,float)) or cfg[key]<=0:raise ValueError('Friend must fill authorized budget: '+key)
    if cfg['gpu_hours']<cfg['pair_gpu_hours']*len(tasks):raise ValueError('Budget must reserve every pair; reduce selected pairs or authorize enough hours')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Commit code before training')
    from cxr.batch.host_resources import available_memory_bytes
    workers=min(cfg['max_concurrent'],len(tasks))
    if (os.cpu_count() or 0)<2*workers:raise RuntimeError('Insufficient CPU budget')
    if available_memory_bytes()<cfg['minimum_free_memory_gib_per_worker']*workers*2**30:raise RuntimeError('Insufficient host RAM')
    if shutil.disk_usage(cfg['output_root']).free<cfg['minimum_free_disk_gib']*2**30:raise RuntimeError('Insufficient free disk')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(v[0]):v[1:] for v in [line.split(', ') for line in raw.strip().splitlines()]}
    for gpu in cfg['gpu_ids']:
        row=devices.get(gpu)
        if not row or '4090' not in row[0] or float(row[1])<22000:raise RuntimeError('Expected authorized RTX4090 24GB')
        if float(row[2])>500 or float(row[3])>5:raise RuntimeError(f'GPU {gpu} occupied; wait for existing jobs')
    write_json(Path(cfg['output_root'])/'server_environment.json',dict(gpus=devices,python=sys.version,host=socket.gethostname(),code_commit=git_commit()))


def execute(task,root,config,gpu,identity,attempt):
    from cxr.phase42.protocol import verify_receipt
    from cxr.regularization_confirmation.delivery import export_light
    config=Path(config);cfg=read_config(config);logs=root/'queue_logs';logs.mkdir(exist_ok=True)
    pidfile=logs/(task['id']+'.pid.json');cp=root/'configs'/(task['id']+'.json')
    command=[sys.executable,str(ROOT/'scripts/confirmation_worker.py'),'--config',str(cp)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    started=time.monotonic()
    with (logs/f"{task['id']}_{attempt}.log").open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(pidfile,dict(pid=process.pid,host=socket.gethostname()))
        try:
            try:code=process.wait(timeout=task['timeout_hours']*3600)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=-1
        finally:pidfile.unlink(missing_ok=True)
    hours=(time.monotonic()-started)/3600
    if code==0:
        verify_receipt(root/'experiments'/task['id']);export_light(root,task['id'])
        return dict(state='DONE',gpu_hours=hours)
    return dict(state='FAILED',gpu_hours=hours,retryable=False,reason=f'exit={code}; see queue_logs/{task["id"]}_{attempt}.log; previous complete epoch retained')


def summary(cfg):
    from cxr.phase39_support import write_csv
    root=Path(cfg['output_root']);rows=[]
    for task in task_specs(cfg):
        dest=root/'experiments'/task['id'];comparison=dest/'bootstrap_comparison.json'
        report=json.loads(comparison.read_text(encoding='utf-8')) if comparison.exists() else {}
        for arm in ['BASE',task['variant']]:
            path=dest/arm/'status.json';status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            metrics=dest/arm/'outer_metrics.json';data=json.loads(metrics.read_text(encoding='utf-8')) if metrics.exists() else {}
            rows.append(dict(task=task['id'],arm=arm,fold=task['fold'],state='DONE' if (dest/'DONE.json').exists() else status.get('state','PENDING'),
                epoch=status.get('epoch',''),best_epoch=status.get('best_epoch',''),outer_auc=data.get('macro_auc',''),
                inner_auc=status.get('inner_auc',status.get('inner_best_auc','')),scope='PARTIAL_OOF_HOLDOUT_EXCLUDED',decision=report.get('decision','PENDING')))
    write_csv(root/'reports'/'experiment_summary.csv',rows)
    print(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['prepare','run','retry-failed','status','export','upload','analyze'],default='run');args=p.parse_args()
    cfg=read_config(args.config);root=Path(cfg['output_root'])
    if args.action=='status':summary(cfg);return
    if args.action in ('export','upload'):
        from cxr.regularization_confirmation.delivery import export_light,publish
        for task in task_specs(cfg):
            if (root/'experiments'/task['id']/'DONE.json').exists():export_light(root,task['id'])
        if args.action=='upload':print(publish(root,cfg['upload']))
        return
    if args.action=='analyze':
        from cxr.regularization_confirmation.analysis import confirmation_analysis
        summary(cfg)
        if cfg['stage']!='pilot':confirmation_analysis(cfg)
        return
    if args.action=='prepare':identity,tasks=prepare(cfg);print(json.dumps(dict(state='CPU_AUDIT_PASS',identity=identity,tasks=tasks)));return
    from cxr.batch.host_gpu_lock import host_gpu_lock
    from cxr.phase42.queue import run
    with host_gpu_lock():
        ensure_no_orphan(root);identity,tasks=prepare(cfg);authorize(cfg,tasks)
        if args.action=='retry-failed':
            ledger=root/'queue_pilot.json';saved=json.loads(ledger.read_text(encoding='utf-8'))
            if saved['identity']!=identity:raise ValueError('Retry identity changed')
            for rec in saved['tasks'].values():
                if rec['state']=='FAILED':rec.update(state='PENDING',reason='Explicit friend retry; prior logs retained')
            write_json(ledger,saved)
        queued=dict(cfg,pilot_gpu_hours=cfg['gpu_hours'],upload={'enabled':False})
        run(tasks,queued,args.config.resolve(),identity,'pilot',executor=execute)
        summary(cfg)
        if cfg['stage']!='pilot':
            from cxr.regularization_confirmation.analysis import confirmation_analysis
            confirmation_analysis(cfg)
