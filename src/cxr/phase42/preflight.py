"""Fail-closed, real-data remote probes, staged concurrency and budget accounting."""
import concurrent.futures
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from cxr.phase39_support import write_json
from cxr.phase42.protocol import ROOT,digest,fingerprint,resolved_config,tasks,verify_baseline


def preserve_baseline(plan,cfg,fold):
    folder,receipt=verify_baseline(plan,cfg,fold)
    source=Path(cfg['phase41_results_root'])
    target=Path(cfg['output_root'])/'reference_baselines'
    files=[p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.yaml','.log','.sha256')]
    if fold!=1:files.append(source/'preflight/REMOTE_PREFLIGHT.json')
    for path in files:
        destination=target/path.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=digest(path):raise ValueError('Frozen BASE reference changed')
        if not destination.exists():shutil.copy2(path,destination)
    write_json(target/f'fold{fold}_provenance.json',dict(receipt_sha256=fingerprint(receipt),
        declared_source_commit=cfg['baseline_source_commit'],
        limitation='Historical Phase41 task manifest stores an opaque identity hash, not its full Git identity; source commit is operator-declared. Config, folds, preflight binding and artifact hashes verified.'))


def prepare(plan,cfg):
    if platform.system()!='Linux':raise RuntimeError('Formal Phase42 requires authorized Linux 4090 server')
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<cfg['minimum_free_disk_gib']*2**30:raise RuntimeError('Insufficient disk')
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<cfg['minimum_free_memory_gib_per_worker']*cfg['max_concurrent']*2**30:raise RuntimeError('Insufficient RAM')
    if (os.cpu_count() or 0)<2*cfg['max_concurrent']:raise RuntimeError('Insufficient CPU cores')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(p[0]):p[1:] for line in raw.splitlines() if (p:=[x.strip() for x in line.split(',')])}
    for gpu in cfg['gpu_ids']:
        d=devices.get(gpu)
        if not d or '4090' not in d[1] or float(d[2])<22000:raise RuntimeError('Authorized RTX4090 not found')
        if float(d[3])>500 or float(d[4])>5:raise RuntimeError(f'GPU {gpu} occupied; no takeover')
    from cxr.data.archive import read_split_rows,split_archive_path
    from cxr.phase42.train import fold_rows
    archive=Path(cfg['data_root'])
    train,val=read_split_rows(archive,'train'),read_split_rows(archive,'val')
    if len(train)!=30344 or len(val)!=1422:raise ValueError('Train/Val image count differs')
    if {r['Subject_id'] for r in train}&{r['Subject_id'] for r in val}:raise ValueError('Train/Val patient overlap')
    for name,sha in plan['fold_sha256'].items():
        if digest(Path(cfg['protocol_root'])/name)!=sha:raise ValueError('Frozen Fold hash changed')
    for fold in range(1,6):
        item=dict(tasks(plan)[0],fold=fold)
        fold_rows(plan,resolved_config(plan,item),archive,Path(cfg['protocol_root']))
    weight=Path(cfg['pretrained_root'])/'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth'
    if digest(weight)!=plan['pretrained_sha256']:raise ValueError('Approved ImageNet weight differs')
    baseline,receipt=verify_baseline(plan,cfg,2)
    preserve_baseline(plan,cfg,2)
    if cfg['upload']['enabled']:
        up=cfg['upload'];meta=json.loads(subprocess.check_output(['gh','repo','view',up['repository'],'--json','isPrivate,viewerPermission'],text=True))
        if not meta['isPrivate'] or meta['viewerPermission'] not in ('ADMIN','WRITE','MAINTAIN'):raise ValueError('Private upload permission missing')
    probe_file=split_archive_path(archive,'train');start=time.perf_counter()
    with probe_file.open('rb') as f:read_bytes=len(f.read(32*2**20))
    write_json(root/'reports/host_resources.json',dict(cpu=os.cpu_count(),available_ram_bytes=mem['MemAvailable'],
        free_disk_bytes=shutil.disk_usage(root).free,read_probe_bytes=read_bytes,read_probe_seconds=time.perf_counter()-start,
        io_scope='32MiB read; cache may apply; concurrent loaders additionally probed'))
    source=[p for folder in ('src/cxr/phase42','scripts','configs/phase42') for p in (ROOT/folder).glob('*.py' if folder!='configs/phase42' else '*.yaml')
            if folder!='scripts' or p.name.startswith('phase42')]
    source.extend([ROOT/'configs/phase42_plan.yaml',ROOT/'docs/phase42/model_costs.json'])
    from cxr.batch.artifacts import commit
    identity=dict(code_commit=commit(),code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in source},
        host=platform.node(),devices={str(g):devices[g][:3] for g in cfg['gpu_ids']},
        plan_sha256=fingerprint(plan),data_index_sha256=fingerprint(dict(train=train,val=val)),
        baseline_receipt_sha256=fingerprint(receipt),baseline_source_commit=cfg['baseline_source_commit'],
        resource_authorization={k:cfg[k] for k in ('gpu_ids','max_concurrent','gpu_hours','pilot_gpu_hours','preflight_gpu_hours','confirmation','max_vram_gib')})
    target=root/'preflight/identity.json'
    if target.exists() and json.loads(target.read_text())!=identity:raise ValueError('Sealed preflight identity changed')
    write_json(target,identity)
    return identity


def run_probe(item,gpu,cfg,config,folder):
    folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[sys.executable,str(ROOT/'scripts/phase42_worker.py'),'--config',str(config),'--mode',item.get('mode','smoke'),
             '--candidate',item['variant'],'--fold','2','--gpu',str(gpu),'--probe',str(folder)]
    start=time.monotonic()
    with (folder/'probe.log').open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(folder/'worker.pid.json',dict(pid=process.pid,host=platform.node()))
        try:
            try:code=process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();raise RuntimeError('Remote preflight timeout')
        finally:
            (folder/'worker.pid.json').unlink(missing_ok=True)
    if code:raise RuntimeError('Remote probe failed: '+str(folder/'probe.log'))
    result=json.loads((folder/'receipt.json').read_text())
    if result['state']!='PASS' or result['peak_reserved_mib']>cfg['max_vram_gib']*1024:raise RuntimeError('CUDA/VRAM probe failed')
    return dict(task=item['id'],gpu=gpu,hours=(time.monotonic()-start)/3600,
                path=(folder/'receipt.json').relative_to(Path(cfg['output_root'])).as_posix(),sha256=digest(folder/'receipt.json'))


def preflight(plan,cfg,config,identity):
    root=Path(cfg['output_root']);folder=root/'preflight';folder.mkdir(exist_ok=True)
    marker=folder/'REMOTE_PREFLIGHT.json';ih=fingerprint(identity)
    if marker.exists():
        result=json.loads(marker.read_text())
        if result['identity_sha256']!=ih or {p['task'] for p in result['probes']}!={t['id'] for t in tasks(plan)}:
            raise ValueError('Incomplete preflight receipt')
        for p in result['probes']:
            if digest(root/p['path'])!=p['sha256']:raise ValueError('Probe hash changed')
        baseline=result['baseline_benchmark']
        if digest(root/baseline['path'])!=baseline['sha256']:raise ValueError('Baseline timing receipt changed')
        benchmark=json.loads((root/baseline['path']).read_text())
        for name,sha in benchmark['sha256'].items():
            if digest((root/baseline['path']).parent/name)!=sha:raise ValueError('Baseline timing changed')
        return result
    ledger=folder/'ledger.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(identity_sha256=ih,used_gpu_hours=0.,probes=[])
    if state['identity_sha256']!=ih or state.get('running'):
        raise RuntimeError('Interrupted preflight; inspect recorded child PIDs; do not silently reset budget')
    remaining=[t for t in tasks(plan) if t['id'] not in {p['task'] for p in state['probes']}]
    # 1 -> 2 -> 4 -> authorized maximum, never jump straight to eight cards.
    level=1
    while remaining:
        count=min(level,cfg['max_concurrent'],len(remaining));batch=remaining[:count]
        if state['used_gpu_hours']+.25*count>cfg['preflight_gpu_hours']:raise RuntimeError('Preflight GPU-hour budget exhausted')
        state['running']=[t['id'] for t in batch];write_json(ledger,state)
        results=[];failure=None
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            futures=[pool.submit(run_probe,t,cfg['gpu_ids'][i],cfg,config,folder/t['id']) for i,t in enumerate(batch)]
            for future in futures:
                try:results.append(future.result())
                except Exception as exc:failure=exc
        state['used_gpu_hours']+=sum(r['hours'] for r in results)+.25*(count-len(results))
        state['probes'].extend(results);state.pop('running');write_json(ledger,state)
        if failure:raise failure
        remaining=remaining[count:];level=min(level*2,cfg['max_concurrent'])
    if 'baseline_benchmark' not in state:
        if state['used_gpu_hours']+.25>cfg['preflight_gpu_hours']:raise RuntimeError('Baseline timing exceeds authorized preflight budget')
        item=dict(tasks(plan)[0],id='BASE_REFERENCE',mode='baseline')
        state['running']=['BASE_REFERENCE'];write_json(ledger,state)
        try:
            result=run_probe(item,cfg['gpu_ids'][0],cfg,config,root/'reports/baseline_reference')
        except Exception:
            state['used_gpu_hours']+=.25;state.pop('running');write_json(ledger,state);raise
        state['used_gpu_hours']+=result['hours'];state['baseline_benchmark']=result
        state.pop('running');write_json(ledger,state)
    state['state']='PASS';write_json(marker,state)
    return state
