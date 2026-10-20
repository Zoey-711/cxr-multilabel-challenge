"""Bounded single-GPU tasks; one process tree per allocated GPU, fail closed."""
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cxr.batch.plan import ROOT, digest, write_json, retryable, selected_tasks, load_plan
from cxr.batch.results import finalize_task, verify_frozen, summarize
from cxr.batch.artifacts import initialize_manifests, record_failure, utcnow


def run_command(command, env, stdout, stderr, timeout):
    with Path(stdout).open('ab') as out, Path(stderr).open('ab') as err:
        p = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env=env, stdout=out, stderr=err,
                             start_new_session=os.name != 'nt')
        pidfile = Path(str(stdout) + '.pid.json')
        write_json(pidfile, {'pid': p.pid, 'host': __import__('socket').gethostname(),
                             'command': command})
        try:
            return p.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'], capture_output=True)
            else:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise TimeoutError('Task timeout: stopped only the task process tree; resume from last epoch')
        finally:
            if p.poll() is not None:
                pidfile.unlink(missing_ok=True)


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
        load_plan(batch_root/'manifests/batch_plan.yaml')
        from cxr.batch.artifacts import commit
        frozen_code=json.loads((batch_root/'manifests/environment.json').read_text(encoding='utf-8'))
        if commit()!=frozen_code['git_commit']:
            raise ValueError('Code commit changed while batch queue was running')
        if (run/'batch_frozen_manifest.json').exists():
            verify_frozen(run)
            from cxr.batch.submission_reporting import emit_single_submission
            emit_single_submission(run)
            if os.environ.get('CXR_BACKUP_ROOT'):
                target=Path(os.environ['CXR_BACKUP_ROOT'])/batch_root.name/task['id']
                if target.exists():
                    verify_frozen(target)
                else:
                    from cxr.batch.backup import backup
                    backup(run,target)
            return {'state':'complete','gpu_hours':0,'reused':True}
        if digest(config) != task['config_sha256']:
            raise ValueError('Config hash mismatch')
        status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        if status.get('state') != 'complete':
            resume = (run/'last_training.pt').exists()
            if run.exists() and not resume:
                raise ValueError('Partial run without an epoch checkpoint; manual inspection required, no overwrite')
            run.mkdir(parents=True, exist_ok=resume)
            command=['scripts/batch_train.py','--config',config,'--run-id',task['id']]
            if resume:
                command += ['--resume']
            rc=run_command(command,env,run/'train_stdout.log',run/'train_stderr.log',deadline-time.monotonic())
            if rc:
                error=(run/'train_stderr.log').read_text(encoding='utf-8',errors='replace')[-12000:]
                return {'state':'failed','stage':stage,'reason':error,'retryable':retryable(error,rc),
                        'gpu_hours':(time.monotonic()-start)/3600}
        if not (run/'batch_timing.json').exists():
            first=logs.parent/'attempt_1'/'timing.json'
            __import__('shutil').copy2(first,run/'batch_timing.json')
        stages=[('validation','scripts/analyze_validation.py','validation_analysis',[])]
        if task['fold'] is not None:
            stages.append(('oof','scripts/batch_oof.py','oof_analysis',[]))
        if task['ema']:
            stages.append(('ema_validation','scripts/analyze_validation.py','ema_validation_analysis',['--checkpoint-kind','ema']))
            if task['fold'] is not None:
                stages.append(('ema_oof','scripts/batch_oof.py','ema_oof_analysis',['--checkpoint-kind','ema']))
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
                return {'state':'failed','stage':stage,'reason':message,'retryable':retryable(message,rc),
                        'gpu_hours':(time.monotonic()-start)/3600}
            write_json(marker,{p.relative_to(run).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})
        stage='latency'
        if not (run/'latency.json').exists():
            rc=run_command(['scripts/batch_latency.py','--config',config,'--run-dir',str(run)],env,
                           logs/'latency.out.log',logs/'latency.err.log',deadline-time.monotonic())
            if rc:raise RuntimeError('Latency measurement failed; see queue_logs')
        if task['ema'] and not (run/'ema_latency.json').exists():
            rc=run_command(['scripts/batch_latency.py','--config',config,'--run-dir',str(run),
                            '--checkpoint-kind','ema'],env,
                           logs/'ema_latency.out.log',logs/'ema_latency.err.log',deadline-time.monotonic())
            if rc:raise RuntimeError('EMA latency measurement failed; see queue_logs')
        stage='test'
        from cxr.data.archive import archive_path, read_split_rows
        from cxr.utils.config import load_config
        try:
            test_rows=read_split_rows(archive_path(load_config(config)),'test')
        except (FileNotFoundError,KeyError):
            write_json(run/'TEST_PENDING.json',{'reason':'Official unlabeled Test data unavailable; training/Val/OOF retained'})
            return {'state':'test_pending','stage':'test','reason':'Official Test unavailable',
                    'gpu_hours':(time.monotonic()-start)/3600}
        if len(test_rows)!=2218:
            raise ValueError('Official Test image count differs from 2218')
        if (run/'TEST_PENDING.json').exists():
            (run/'TEST_PENDING.json').unlink()
        for kind in (['raw','ema'] if task['ema'] else ['raw']):
            name='test_predictions.csv' if kind=='raw' else 'ema_test_predictions.csv'
            target=run/name
            if target.exists():
                from cxr.batch.test_results import read_test_probabilities
                read_test_probabilities(target)
                continue
            rc=run_command(['scripts/batch_test_inference.py','--config',config,'--run-dir',str(run),
                            '--checkpoint-kind',kind],env,logs/(f'test_{kind}.out.log'),
                           logs/(f'test_{kind}.err.log'),deadline-time.monotonic())
            if rc:
                message=(logs/(f'test_{kind}.err.log')).read_text(encoding='utf-8',errors='replace')[-12000:]
                return {'state':'test_failed','stage':'test','reason':message,
                        'gpu_hours':(time.monotonic()-start)/3600}
        stage='freeze'
        finalize_task(run,task)
        from cxr.batch.submission_reporting import emit_single_submission
        emit_single_submission(run)
        if os.environ.get('CXR_BACKUP_ROOT'):
            from cxr.batch.backup import backup
            target=Path(os.environ['CXR_BACKUP_ROOT'])/batch_root.name/task['id']
            target.parent.mkdir(parents=True,exist_ok=True)
            backup(run,target)
        return {'state':'complete','gpu_hours':(time.monotonic()-start)/3600}
    except Exception as exc:
        return {'state':'failed','stage':stage,'reason':f'{type(exc).__name__}: {exc}',
                'retryable':False,'gpu_hours':(time.monotonic()-start)/3600}


def budget_allows(consumed, reserved, next_hours, budget):
    return consumed + reserved + next_hours <= budget


def run_queue(plan, plan_path, batch_root, gpus, budget, resume=False, extended=False, initial_gpu_hours=0, max_concurrent=None):
    root=Path(batch_root).resolve()
    tasks=selected_tasks(plan,extended)
    if budget <= 0:
        raise ValueError('Friend-approved positive GPU-hour budget is required')
    if resume:
        state=json.loads((root/'queue_state.json').read_text())
        if state['plan_sha256']!=digest(plan_path) or state['extended']!=extended:
            raise ValueError('Resume plan/tier changed')
    else:
        root.mkdir(parents=True,exist_ok=False)
        state={'plan_sha256':digest(plan_path),'extended':extended,'gpu_hours':initial_gpu_hours,
               'tasks':{t['id']:{'state':'pending','attempts':0} for t in tasks}}
        initialize_manifests(root,plan,plan_path)
    lock=root/'queue.lock'
    # A stale lock is intentionally not auto-removed. Owner confirms job death first.
    with lock.open('x') as f:
        json.dump({'pid':os.getpid(),'host':__import__('socket').gethostname()},f)
    try:
        for record in state['tasks'].values():
            if record['state']=='running':
                # Conservatively account interrupted reservations; never lose budget usage.
                from datetime import datetime, timezone
                started = record.get('reserved_started_utc')
                elapsed = ((datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()/3600
                           if started else record.get('reserved_hours', 0))
                state['gpu_hours'] += min(record.get('reserved_hours', 0), max(0, elapsed))
                record['state']='pending'
            elif record['state']=='failed' and record.get('stage')=='freeze':
                # Backup failures may be retried explicitly without retraining frozen weights.
                record['state']='pending'
        for record in state['tasks'].values():
            if resume and record['state'] in ('test_pending','test_failed'):
                record['state']='pending'
        active={}
        maximum=min(len(gpus), max_concurrent or len(gpus))
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=maximum)
        try:
            while True:
                for future,(task,gpu) in list(active.items()):
                    if future.done():
                        result=future.result()
                        record=state['tasks'][task['id']]
                        elapsed=result.pop('gpu_hours')
                        state['gpu_hours']+=elapsed
                        record['gpu_hours']=record.get('gpu_hours',0)+elapsed
                        record.update(result)
                        if result['state'] in ('failed','test_failed'):
                            record_failure(root,task,record)
                        if result['state']=='failed' and result.get('retryable') and record['attempts']<=task['max_retries']:
                            record['state']='pending'
                        del active[future]
                        from cxr.batch.reporting import csv_rows
                        status_rows=[{'Model_ID':tid,'State':item['state'],'Attempts':item['attempts'],
                                      'GPU_Hours':item.get('gpu_hours',0),'Stage':item.get('stage',''),
                                      'Reason':item.get('reason','')} for tid,item in state['tasks'].items()]
                        temporary=root/'reports/experiment_status.csv.tmp'
                        csv_rows(temporary,status_rows)
                        temporary.replace(root/'reports/experiment_status.csv')
                busy={gpu for _,gpu in active.values()}
                drain=(root/'STOP').exists()
                for gpu in gpus:
                    if gpu in busy or drain or len(active)>=maximum:
                        continue
                    reserved=sum(t['timeout_hours'] for t,_ in active.values())
                    for task in tasks:
                        rec=state['tasks'][task['id']]
                        if rec['state']!='pending' or not all(state['tasks'][d]['state']=='complete' for d in task['depends_on']):
                            continue
                        if not budget_allows(state['gpu_hours'],reserved,task['timeout_hours'],budget):
                            continue
                        rec.update(state='running',attempts=rec['attempts']+1,gpu=gpu,
                                   reserved_hours=task['timeout_hours'], reserved_started_utc=utcnow())
                        active[executor.submit(execute,task,root,gpu,rec['attempts'])]=(task,gpu)
                        break
                state.update(updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                             budget_gpu_hours=budget,stop_requested=drain)
                write_json(root/'queue_state.json',state)
                (root/'status.md').write_text('# Batch status\n\n'+ '\n'.join(f'- {k}: {v["state"]}, attempts={v["attempts"]}' for k,v in state['tasks'].items()),encoding='utf-8')
                if not active:
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            (root/'STOP').touch()
            # Finish active tasks safely; subsequent --resume reuses completed stages.
            raise
        finally:
            executor.shutdown(wait=True)
        from cxr.batch.reporting import report
        try:
            report(root)
            # A bounded, predeclared full-ensemble research benchmark. It never
            # establishes official latency compliance or changes model selection.
            reserve=0.25
            if not (root/'STOP').exists() and budget_allows(state['gpu_hours'],0,reserve,budget):
                bench_env=os.environ.copy();bench_env['CUDA_VISIBLE_DEVICES']=gpus[0]
                logs=root/'queue_logs';logs.mkdir(exist_ok=True)
                started=time.monotonic()
                try:
                    rc=run_command(['scripts/batch_ensemble_latency.py','--config',str(plan_path),
                                    '--root',str(root)],bench_env,
                                   logs/'ensemble_latency.out.log',logs/'ensemble_latency.err.log',reserve*3600)
                except Exception as exc:
                    rc=-1
                    write_json(root/'reports/ENSEMBLE_BENCHMARK_FAILED.json',
                               {'error':f'{type(exc).__name__}: {exc}',
                                'scope':'research full-candidate timing; model metrics remain frozen'})
                finally:
                    state['gpu_hours']+=(time.monotonic()-started)/3600
                    write_json(root/'queue_state.json',state)
                if rc and not (root/'reports/ENSEMBLE_BENCHMARK_FAILED.json').exists():
                    write_json(root/'reports/ENSEMBLE_BENCHMARK_FAILED.json',
                               {'exit_code':rc,'scope':'research full-candidate timing; model metrics remain frozen'})
                report(root)
            else:
                write_json(root/'reports/ENSEMBLE_BENCHMARK_SKIPPED.json',
                           {'reason':'STOP requested or remaining GPU-hour budget <0.25'})
        except Exception as exc:
            write_json(root/'reports/ANALYSIS_FAILED.json',{'state':'failed','reason':f'{type(exc).__name__}: {exc}'})
            write_json(root/'reports/experiment_summary.json',{'state':'analysis_failed','queue':state,'metrics':None})
            (root/'reports/phase17_final_report.md').write_text('# Analysis failed\n\n'+str(exc)+'\nNo missing metrics were fabricated. See queue_state.json and per-task artifacts.',encoding='utf-8')
        return state
    finally:
        lock.unlink()
        # Results stay in direct folders; no archive or automatic deletion.
