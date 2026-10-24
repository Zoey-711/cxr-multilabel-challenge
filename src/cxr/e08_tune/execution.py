"""Phase32 task execution reuses frozen train/evaluation paths, without Phase17 queue mutation."""
import json
import os
import time
from pathlib import Path
from cxr.batch.runner import run_command
from cxr.batch.plan import ROOT,digest,write_json,retryable
from cxr.batch.artifacts import utcnow
from cxr.e08_tune.protocol import load_plan
from cxr.e08_tune.results import finalize,verify_done

def execute(task, batch_root, gpu, attempt):
    start = time.monotonic()
    deadline = start + task['timeout_hours']*3600
    batch_root = Path(batch_root)
    run = batch_root/'experiments'/task['id']
    logs = batch_root/'queue_logs'/task['id']/f'attempt_{attempt}'
    logs.mkdir(parents=True, exist_ok=False)
    write_json(logs/'timing.json',{'start_utc':utcnow()})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, CXR_OUTPUT_ROOT=str(batch_root), PYTHONUNBUFFERED='1')
    config = str(ROOT/task['config'])
    stage = 'train'
    try:
        plan=load_plan()
        if (run/'DONE.json').exists():
            verify_done(run)
            return {'state':'DONE','gpu_hours':0,'reused':True}
        if digest(config) != task['config_sha256']:
            raise ValueError('Config hash mismatch')
        status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        if status.get('state') != 'complete':
            resume = (run/'last_training.pt').exists()
            if run.exists() and not resume:
                raise ValueError('Partial run without an epoch checkpoint; manual inspection required, no overwrite')
            run.mkdir(parents=True, exist_ok=resume)
            command=['scripts/e08_tune_train.py','--config',config,'--run-id',task['id']]
            if resume:
                command += ['--resume']
            rc=run_command(command,env,run/'train_stdout.log',run/'train_stderr.log',deadline-time.monotonic())
            if rc:
                error=(run/'train_stderr.log').read_text(encoding='utf-8',errors='replace')[-12000:]
                return {'state':'FAILED','stage':stage,'reason':error,'retryable':retryable(error,rc),
                        'gpu_hours':(time.monotonic()-start)/3600}
        if not (run/'batch_timing.json').exists():
            first=logs.parent/'attempt_1'/'timing.json'
            __import__('shutil').copy2(first,run/'batch_timing.json')
        stages=[('validation','scripts/phase32_validation.py','validation_analysis',[])]
        if task['fold'] is not None:
            stages.append(('oof','scripts/phase32_oof.py','oof_analysis',[]))
        if task['ema']:
            stages.append(('ema_validation','scripts/phase32_validation.py','ema_validation_analysis',['--checkpoint-kind','ema']))
            if task['fold'] is not None:
                stages.append(('ema_oof','scripts/phase32_oof.py','ema_oof_analysis',['--checkpoint-kind','ema']))
        for stage, script, folder, extra in stages:
            marker=run/(folder+'.done.json')
            if marker.exists():
                recorded=json.loads(marker.read_text())
                if any(digest(run/rel)!=h for rel,h in recorded.items()):
                    raise ValueError('Completed evaluation hash mismatch')
                continue
            output=run/folder
            if output.exists():
                # Preserve partial evaluation for diagnosis instead of overwriting it.
                output.rename(run/(folder+f'.partial_attempt_{attempt}'))
            rc=run_command([script,'--config',config,'--run-dir',str(run),*extra],env,
                           logs/(stage+'.out.log'),logs/(stage+'.err.log'),deadline-time.monotonic())
            if rc:
                message=(logs/(stage+'.err.log')).read_text(encoding='utf-8',errors='replace')[-12000:]
                return {'state':'FAILED','stage':stage,'reason':message,'retryable':retryable(message,rc),
                        'gpu_hours':(time.monotonic()-start)/3600}
            write_json(marker,{p.relative_to(run).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})
        for kind in (['raw','ema'] if task['ema'] else ['raw']):
            name='latency.json' if kind=='raw' else 'ema_latency.json'
            if not (run/name).exists():
                stage=kind+'_research_latency'
                rc=run_command(['scripts/phase32_latency.py','--config',config,'--run-dir',str(run),'--checkpoint-kind',kind],
                    env,logs/(stage+'.out.log'),logs/(stage+'.err.log'),deadline-time.monotonic())
                if rc:raise RuntimeError('Research latency probe failed; preserve trained weights')
        stage='export'
        finalize(run,task,plan,os.environ['CXR_E08_BASELINES'])
        return {'state':'DONE','gpu_hours':(time.monotonic()-start)/3600}
    except Exception as exc:
        return {'state':'FAILED','stage':stage,'reason':f'{type(exc).__name__}: {exc}',
                'retryable':False,'gpu_hours':(time.monotonic()-start)/3600}
