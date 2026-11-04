"""Budgeted remote execution reuses the existing durable queue and host GPU lock."""
import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from cxr.batch.host_gpu_lock import host_gpu_lock
from cxr.phase39_support import write_json
from cxr.phase42.queue import run as queue_run
from cxr.phase42.protocol import verify_receipt
from cxr.strong_kd.protocol import ROOT, RECIPES, TRAINING, PRETRAINED_SHA, config, digest, fingerprint, source_lock, tasks


def authorize(cfg):
    if cfg.get('authorize_training') is not True:
        raise ValueError('Set authorize_training: true only after approving the resource budget')
    for key in ('gpu_hours','preflight_gpu_hours','minimum_free_disk_gib','max_vram_gib'):
        v = cfg.get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Explicit positive budget required: '+key)
    ids = cfg.get('gpu_ids', [])
    if not ids or len(ids)>2 or len(set(ids))!=len(ids) or any(type(i)!=int or i<0 for i in ids):
        raise ValueError('Authorize one or two distinct physical GPUs for the paired experiment')
    if type(cfg.get('max_concurrent'))!=int or not 1<=cfg['max_concurrent']<=len(ids):
        raise ValueError('Invalid concurrency')
    if cfg['gpu_hours']<=cfg['preflight_gpu_hours'] or cfg['max_vram_gib']>22:
        raise ValueError('Insufficient training budget or unsafe VRAM limit')


def identity(cfg):
    root = Path(cfg['output_root'])
    code = list((ROOT/'src/cxr/strong_kd').glob('*.py')) + [ROOT/'scripts/strong_kd_run.py', ROOT/'configs/strong_kd_sources.json']
    code += [ROOT/p for p in ('src/cxr/phase39_run.py','src/cxr/phase39_support.py',
        'src/cxr/phase42/queue.py','src/cxr/data/dataset.py','src/cxr/data/archive.py',
        'src/cxr/models/factory.py','src/cxr/utils/training_resume.py',
        'src/cxr/engine/metrics.py','src/cxr/engine/validate.py','src/cxr/inference/aggregate.py')]
    teachers = {str(p.relative_to(root)):digest(p) for p in sorted((root/'teachers').glob('*')) if p.is_file()}
    if len(teachers)!=4: raise ValueError('Both frozen teacher CSVs and provenance files are required')
    expected=json.loads((root/'teacher_files_sha256.json').read_text())
    if {Path(k).as_posix():v for k,v in teachers.items()}!=expected:
        raise ValueError('Prepared teacher target hash changed')
    return dict(config=cfg, training=TRAINING, recipes=RECIPES,
        sources_sha256=fingerprint(source_lock()), audit_sha256=digest(root/'teacher_audit.json'),
        teachers_sha256=teachers, code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code})


def execute(task, root, config_path, gpu, identity_hash, attempt):
    folder = root/'queue_logs'; folder.mkdir(parents=True,exist_ok=True)
    pidfile = folder/f"{task['id']}.pid.json"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable,str(ROOT/'scripts/strong_kd_run.py'),'--config',str(config_path),
               '--worker','train','--task',task['id'],'--identity',identity_hash]
    start = time.monotonic()
    with (folder/f"{task['id']}_{attempt}.log").open('a',encoding='utf-8') as f:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        write_json(pidfile,dict(pid=p.pid,host=platform.node(),task=task['id']))
        try:
            try: code=p.wait(timeout=task['timeout_hours']*3600)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                code=-1
        finally:pidfile.unlink(missing_ok=True)
    if code==0:
        verify_receipt(root/'experiments'/task['id'])
        return dict(state='DONE',gpu_hours=(time.monotonic()-start)/3600)
    return dict(state='FAILED',reason=f'exit={code}; see queue_logs/{task["id"]}_{attempt}.log',
                gpu_hours=(time.monotonic()-start)/3600,retryable=False)


def preflight(cfg, path):
    authorize(cfg)
    if platform.system()!='Linux': raise RuntimeError('Formal execution requires the authorized Linux server')
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < cfg['minimum_free_disk_gib']*2**30:
        raise RuntimeError('Insufficient free disk')
    if (os.cpu_count() or 0)<2*cfg['max_concurrent']: raise RuntimeError('Insufficient CPU cores')
    memory={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()}
    if memory['MemAvailable']<6*cfg['max_concurrent']*2**30: raise RuntimeError('Insufficient RAM')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(x[0]):x[1:] for line in raw.splitlines() if (x:=[s.strip() for s in line.split(',')])}
    for gpu in cfg['gpu_ids']:
        d=devices.get(gpu)
        if not d or '4090' not in d[0] or float(d[1])<22000 or float(d[2])>500 or float(d[3])>5:
            raise RuntimeError('Authorized 4090 missing/occupied; no takeover')
    weight=Path(cfg['pretrained_root'])/'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth'
    if digest(weight)!=PRETRAINED_SHA: raise ValueError('ImageNet checkpoint hash differs')
    ident=identity(cfg);ih=fingerprint(ident)
    folder=root/'preflight';folder.mkdir(exist_ok=True)
    marker=folder/'REMOTE_PREFLIGHT.json'
    if marker.exists():
        receipt=json.loads(marker.read_text())
        if receipt['state']!='PASS' or receipt['identity_sha256']!=ih: raise ValueError('Preflight identity changed')
        for rel,sha in receipt['sha256'].items():
            if digest(root/rel)!=sha:raise ValueError('CUDA receipt changed')
        return ih
    write_json(folder/'identity.json',ident)
    ledger=folder/'ledger.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(identity=ih,used_gpu_hours=0.,completed=[])
    if state['identity']!=ih or state.get('running'):
        raise ValueError('Interrupted/changed preflight; inspect PID and budget before continuing')
    for task in RECIPES:
        if task in state['completed']:continue
        if state['used_gpu_hours']+.25>cfg['preflight_gpu_hours']:raise RuntimeError('Preflight budget exhausted')
        state['running']=task;write_json(ledger,state)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(cfg['gpu_ids'][0]),PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2')
        start=time.monotonic();code=-1
        with (folder/f'{task}.log').open('a',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/strong_kd_run.py'),'--config',str(path),
                '--worker','smoke','--task',task],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            state['pid']=p.pid;write_json(ledger,state)
            try:
                code=p.wait(timeout=900)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        state['used_gpu_hours']+=(time.monotonic()-start)/3600
        state.pop('running');state.pop('pid');write_json(ledger,state)
        if code:raise RuntimeError(f'CUDA short-test failed: preflight/{task}.log')
        rec=json.loads((folder/task/'receipt.json').read_text())
        if rec['state']!='PASS' or rec['peak_reserved_mib']>cfg['max_vram_gib']*1024:
            raise RuntimeError('CUDA/VRAM check failed')
        state['completed'].append(task);write_json(ledger,state)
    write_json(marker,dict(state='PASS',identity_sha256=ih,sha256={f'preflight/{t}/receipt.json':digest(folder/t/'receipt.json') for t in RECIPES}))
    return ih


def student_decision(primary):
    delta=primary['bootstrap']['delta']
    lower=primary['bootstrap']['ci95'][0]
    per_class=[b-a for a,b in zip(primary['control']['per_class_auc'],
                                  primary['complement']['per_class_auc'])]
    if len(per_class)!=10: raise ValueError('Ten paired class AUC values required')
    evidence=dict(macro_delta=delta,ci_lower=lower,worst_per_class_delta=min(per_class))
    state='CONSIDER_NESTED_TEACHER_PILOT' if delta>=.002 and lower>0 and min(per_class)>=-.01 else 'STOP_TEACHER_REBUILD'
    return state,evidence


def analyze(cfg):
    from cxr.batch.artifacts import read_probabilities
    from cxr.phase41.analysis import auc_summary
    from cxr.phase42.analysis import paired_bootstrap
    from cxr.strong_kd.teachers import rows
    root=Path(cfg['output_root']);labels=Path(cfg['teacher_root'])/'manifests/val_labels.csv'
    truth={(r['Subject_id'],r['Study_id']):[int(r[f'Target_{j}']) for j in range(1,11)] for r in rows(labels)}
    keys=sorted(truth);report=dict(state='INCOMPLETE',student_oof='N/A',evaluation='EXPLORATORY_OFFICIAL_VAL',comparisons={})
    if all((root/'experiments'/t/'DONE.json').is_file() for t in RECIPES):
        for task in RECIPES: verify_receipt(root/'experiments'/task)
        for version in ('raw','ema'):
            preds=[]
            for task in RECIPES:
                p=read_probabilities(root/'experiments'/task/f'{version}_val_predictions.csv',labels,set(keys))
                preds.append(dict(zip(p['keys'],p['p'])))
            report['comparisons'][version]=dict(control=auc_summary(truth,preds[0],keys),
                complement=auc_summary(truth,preds[1],keys),bootstrap=paired_bootstrap(truth,*preds,keys,draws=2000,seed=5417))
        report['state']='COMPLETE_DEVELOPMENT_ONLY'
        primary=report['comparisons']['raw']
        decision,evidence=student_decision(primary)
        report['selection_rule']=dict(primary='raw_final_epoch',minimum_macro_delta=.002,
            minimum_ci_lower=0.,minimum_per_class_delta=-.01,
            note='Exploratory gate only; official Val has been reused and cannot confirm generalization')
        report['selection_evidence']=evidence
        report['decision']=decision
    else:
        report['decision']='WAIT_FOR_PAIRED_STUDENTS'
    report['scope']='No independent student OOF; no automatic expansion or Submission. Raw final epoch is primary; EMA is predeclared secondary.'
    write_json(root/'reports/student_comparison.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=('audit','preflight','run','resume','status','stop','analyze'),default='audit')
    p.add_argument('--worker',choices=('smoke','train'),help=argparse.SUPPRESS)
    p.add_argument('--task',choices=tuple(RECIPES),help=argparse.SUPPRESS)
    p.add_argument('--identity',help=argparse.SUPPRESS)
    args=p.parse_args();cfg=config(args.config);root=Path(cfg['output_root'])
    if args.worker:
        authorize(cfg)
        if platform.system()!='Linux':raise RuntimeError('Remote worker requires Linux')
        import torch
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg['max_vram_gib']*2**30/torch.cuda.get_device_properties(0).total_memory),0)
        from cxr.strong_kd.train import smoke,run_task
        if args.worker=='smoke':
            folder=root/'preflight'/args.task
            write_json(folder/'receipt.json',smoke(cfg,args.task,folder,torch.device('cuda')))
        else:
            receipt=json.loads((root/'preflight/REMOTE_PREFLIGHT.json').read_text())
            if receipt['state']!='PASS' or receipt['identity_sha256']!=args.identity or fingerprint(identity(cfg))!=args.identity:
                raise ValueError('Valid unchanged remote preflight required')
            run_task(cfg,args.task,args.identity)
        return
    if args.action=='status':
        path=root/'reports/pilot_status.csv'
        print(path.read_text(encoding='utf-8') if path.exists() else 'NOT_STARTED');return
    if args.action=='stop':root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();return
    if args.action=='analyze':print(json.dumps(analyze(cfg)));return
    from cxr.strong_kd.teachers import prepare
    if args.action=='audit':
        record=prepare(cfg['teacher_root'],root)
        print(json.dumps(dict(state=record['state'],metrics=record['metrics'],bootstrap=record['bootstrap'])));return
    with host_gpu_lock():
        if args.action=='resume':(root/'STOP').unlink(missing_ok=True)
        elif (root/'STOP').exists():raise RuntimeError('Drain stop set; use --action resume')
        authorize(cfg)
        # Recompute the frozen recipe before any GPU use, even with cached reports.
        prepare(cfg['teacher_root'],root)
        try:ih=preflight(cfg,args.config.resolve())
        except Exception as exc:
            write_json(root/'reports/PREFLIGHT_FAILED.json',dict(reason=str(exc)));raise
        if args.action=='preflight':print('REMOTE_PREFLIGHT PASS');return
        queued=dict(cfg,pilot_gpu_hours=cfg['gpu_hours']-cfg['preflight_gpu_hours'],upload={'enabled':False})
        queue_run(tasks(),queued,args.config.resolve(),ih,'pilot',executor=execute)
        print(json.dumps(analyze(cfg)))


if __name__=='__main__':main()
