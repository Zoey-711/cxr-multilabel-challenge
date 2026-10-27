"""Independent task workers; bounded runtime, durable budget, no duplicate DONE."""
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from cxr.phase35.queue import lock
from cxr.phase39_support import write_json,write_csv
from cxr.phase42.protocol import ROOT,fingerprint,verify_receipt


def report(root,state,items,stage):
    rows=[]
    for task in items:
        rec=state['tasks'][task['id']];folder=root/'experiments'/task['id']
        info=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        rows.append(dict(task=task['id'],fold=task['fold'],variant=task['variant'],state=rec['state'],
            gpu=rec.get('gpu',''),attempts=rec['attempts'],epoch=info.get('epoch',''),best_epoch=info.get('best_epoch',''),
            auc=info.get('oof_macro_auc',info.get('best_auc','')),reason=rec.get('reason','')))
    if rows:write_csv(root/'reports'/f'{stage}_status.csv',rows)
    write_json(root/'reports'/f'{stage}_budget.json',dict(used_gpu_hours=state['gpu_hours_used'],authorized_gpu_hours=state['budget_hours']))


def execute(task,root,config,gpu,identity,attempt):
    logs=root/'queue_logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/f"{task['id']}_{attempt}.log";pidfile=logs/f"{task['id']}.pid.json"
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONPATH']=str(ROOT/'src')
    env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    command=[sys.executable,str(ROOT/'scripts/phase42_worker.py'),'--config',str(config),'--mode','train',
             '--candidate',task['variant'],'--fold',str(task['fold']),'--identity',identity,'--gpu',str(gpu)]
    start=time.monotonic()
    with path.open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(pidfile,dict(pid=process.pid,host=socket.gethostname(),task=task['id']))
        try:
            try:code=process.wait(timeout=task['timeout_hours']*3600)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=-1
        finally:pidfile.unlink(missing_ok=True)
    hours=(time.monotonic()-start)/3600
    if code==0:
        verify_receipt(root/'experiments'/task['id'])
        return dict(state='DONE',gpu_hours=hours)
    tail=path.read_text(encoding='utf-8',errors='replace')[-2500:]
    retryable=any(x in tail for x in ('Temporary failure','Resource temporarily unavailable','Connection reset'))
    return dict(state='FAILED',reason=f'exit={code}: {tail}',retryable=retryable,gpu_hours=hours)


def run(items,cfg,config,identity,stage,executor=execute,poll=3):
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    guard=lock(root)
    budget=cfg['pilot_gpu_hours'] if stage=='pilot' else cfg['confirmation']['gpu_hours']
    path=root/f'queue_{stage}.json'
    try:
        state=json.loads(path.read_text()) if path.exists() else dict(identity=identity,plan=fingerprint(items),
            budget_hours=budget,gpu_hours_used=0.,tasks={t['id']:dict(state='PENDING',attempts=0) for t in items})
        if (state['identity'],state['plan'],state['budget_hours'])!=(identity,fingerprint(items),budget):
            raise ValueError('Queue identity/budget changed; no silent restart')
        for t in items:
            rec=state['tasks'][t['id']];folder=root/'experiments'/t['id']
            if rec['state']=='RUNNING':
                # Parent/child PIDs checked by queue lock. Charge the full reserved
                # amount conservatively when exact termination time is unavailable.
                state['gpu_hours_used']+=t['timeout_hours']
                rec.update(state='PENDING' if rec['attempts']<=t['max_retries'] else 'FAILED',reason='Interrupted; reservation charged once')
            if (folder/'DONE.json').exists():
                receipt=verify_receipt(folder)
                if receipt['identity_sha256']!=identity:raise ValueError('DONE identity changed')
                rec['state']='DONE'
            elif rec['state']=='DONE':raise ValueError('Missing DONE receipt')
        write_json(path,state)
        active={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['max_concurrent']) as pool:
            while True:
                for future,(task,gpu) in list(active.items()):
                    if not future.done():continue
                    try:result=future.result()
                    except Exception as exc:result=dict(state='FAILED',reason=str(exc),gpu_hours=task['timeout_hours'],retryable=False)
                    state['gpu_hours_used']+=result.pop('gpu_hours')
                    rec=state['tasks'][task['id']];rec.update(result)
                    if rec['state']=='FAILED':
                        folder=root/'experiments'/task['id'];folder.mkdir(parents=True,exist_ok=True)
                        write_json(folder/f"FAILED_attempt_{rec['attempts']}.json",rec)
                        if rec.get('retryable') and rec['attempts']<=task['max_retries']:rec['state']='PENDING'
                        else:write_json(folder/'FAILED.json',rec)
                    del active[future];write_json(path,state);report(root,state,items,stage)
                    if cfg['upload']['enabled']:
                        try:
                            from cxr.phase42.upload import upload
                            upload(root,cfg['upload'])
                        except Exception as exc:write_json(root/'reports/UPLOAD_FAILED.json',dict(reason=str(exc)))
                busy={g for _,g in active.values()}
                for gpu in cfg['gpu_ids']:
                    if gpu in busy or len(active)>=cfg['max_concurrent'] or (root/'STOP').exists():continue
                    for task in items:
                        rec=state['tasks'][task['id']]
                        if rec['state']!='PENDING':continue
                        reserved=sum(t['timeout_hours'] for t,_ in active.values())
                        if state['gpu_hours_used']+reserved+task['timeout_hours']>budget:
                            rec['reason']='BUDGET_EXHAUSTED';continue
                        rec.update(state='RUNNING',attempts=rec['attempts']+1,gpu=gpu,reason='')
                        write_json(path,state)
                        active[pool.submit(executor,task,root,config,gpu,identity,rec['attempts'])]=(task,gpu)
                        break
                report(root,state,items,stage)
                if not active:break
                time.sleep(poll)
        write_json(path,state)
        return state
    finally:guard.unlink(missing_ok=True)
