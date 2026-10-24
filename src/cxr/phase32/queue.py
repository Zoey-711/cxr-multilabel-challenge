"""One GPU per independent task; bounded retries and durable budget ledger."""
import concurrent.futures
import json
import os
import socket
import time
from pathlib import Path
from cxr.batch.plan import write_json
from cxr.phase32.protocol import fingerprint
from cxr.phase32.results import report,verify_done

def alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def lock(root):
    path=Path(root)/'queue.lock'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['host']!=socket.gethostname() or alive(prior['pid']):raise RuntimeError('Queue owner may still be active')
        for sidecar in Path(root).rglob('*.pid.json'):
            info=json.loads(sidecar.read_text())
            if info['host']!=socket.gethostname() or alive(info['pid']):raise RuntimeError('Training child still active; do not duplicate')
        path.unlink()
    with path.open('x') as f:json.dump({'pid':os.getpid(),'host':socket.gethostname()},f)
    return path

def run(plan,cfg,executor_fn,initial_hours=0,poll_seconds=3):
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    guard=lock(root)
    try:
        path=root/'queue_state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state['plan_fingerprint']!=fingerprint(plan):raise ValueError('Queue plan changed')
        else:
            state={'plan_fingerprint':fingerprint(plan),'gpu_hours':initial_hours,'tasks':{t['id']:{'state':'PENDING','attempts':0} for t in plan['tasks']}}
        for tid,r in state['tasks'].items():
            if r['state']=='RUNNING':
                # Dead parent and children verified above. Charge reservation once, never lose usage.
                state['gpu_hours']+=r['reserved_hours'];r['state']='PENDING'
            if r['state']=='DONE':verify_done(root/'experiments'/tid)
        write_json(path,state)
        active={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['max_concurrent']) as pool:
            while True:
                for future,(task,gpu) in list(active.items()):
                    if not future.done():continue
                    try:res=future.result()
                    except Exception as e:res={'state':'FAILED','reason':str(e),'gpu_hours':task['timeout_hours'],'retryable':False}
                    state['gpu_hours']+=res.pop('gpu_hours')
                    rec=state['tasks'][task['id']];rec.update(res)
                    if res['state']=='FAILED':
                        run_dir=root/'experiments'/task['id'];run_dir.mkdir(parents=True,exist_ok=True)
                        write_json(run_dir/f"FAILED_attempt_{rec['attempts']}.json",rec)
                        write_json(run_dir/'FAILED.json',rec)
                        if res.get('retryable') and rec['attempts']<=task['max_retries']:rec['state']='PENDING'
                    del active[future]
                    write_json(path,state);report(root,state,plan)
                    if cfg.get('upload',{}).get('enabled'):
                        try:
                            from cxr.phase32.upload import upload
                            upload(root,cfg['upload'])
                        except Exception as e:write_json(root/'reports/UPLOAD_FAILED.json',{'reason':str(e)})
                busy={gpu for _,gpu in active.values()}
                for gpu in cfg['gpu_ids']:
                    if gpu in busy or len(active)>=cfg['max_concurrent'] or (root/'STOP').exists():continue
                    for task in plan['tasks']:
                        r=state['tasks'][task['id']]
                        if r['state']!='PENDING':continue
                        reserved=sum(t['timeout_hours'] for t,_ in active.values())
                        if state['gpu_hours']+reserved+task['timeout_hours']>cfg['gpu_hours']:continue
                        r.update(state='RUNNING',attempts=r['attempts']+1,gpu=gpu,reserved_hours=task['timeout_hours'])
                        write_json(path,state) # persist BEFORE launching
                        active[pool.submit(executor_fn,task,root,str(gpu),r['attempts'])]=(task,gpu)
                        break
                write_json(path,state);report(root,state,plan)
                if not active:break
                time.sleep(poll_seconds)
        return state
    finally:guard.unlink(missing_ok=True)
