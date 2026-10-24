"""Remote-only real-data probes. No success inferred from local CPU tests."""
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from cxr.batch.plan import ROOT,digest,write_json
from cxr.batch.runner import run_command
from cxr.phase32.protocol import fingerprint

def prepare(cfg,plan):
    if platform.system()!='Linux':raise RuntimeError('Formal Phase32 execution requires the authorized Linux server')
    os.environ.update(CXR_DATA_ROOT=cfg['data_root'],CXR_PRETRAINED_ROOT=cfg['pretrained_root'],
                      CXR_OUTPUT_ROOT=cfg['output_root'],CXR_PHASE32_BASELINES=cfg['baseline_results_root'])
    out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out).free<cfg['minimum_free_disk_gib']*2**30:raise RuntimeError('Insufficient output disk capacity')
    if (os.cpu_count() or 0)<4*cfg['max_concurrent']:raise RuntimeError('Insufficient CPUs for 4 DataLoader workers per task; reduce authorized concurrency')
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<cfg['minimum_free_memory_gib_per_worker']*cfg['max_concurrent']*2**30:raise RuntimeError('Insufficient available host RAM')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(r.split(',')[0]):[v.strip() for v in r.split(',')[1:]] for r in raw.strip().splitlines()}
    for gpu in cfg['gpu_ids']:
        d=devices.get(gpu)
        if not d or '4090' not in d[1] or float(d[2])<22000:raise RuntimeError(f'Authorized GPU {gpu} is not a 24GB RTX4090')
        if float(d[3])>500 or float(d[4])>5:raise RuntimeError(f'GPU {gpu} occupied; no interference allowed')
    directory=ROOT/'outputs/analysis/phase17_patient_folds';directory.mkdir(parents=True,exist_ok=True)
    for name,h in plan['fold_hashes'].items():
        source=Path(cfg['protocol_root'])/name;target=directory/name
        if digest(source)!=h:raise ValueError('Frozen fold source mismatch')
        if target.exists() and digest(target)!=h:raise ValueError('Existing fold assets differ; no overwrite')
        if not target.exists():shutil.copy2(source,target)
    from cxr.data.archive import archive_path,read_split_rows
    from cxr.data.patient_folds import select_fold_rows
    from cxr.models.factory import imagenet_weight_info
    from cxr.phase32.config import load_config
    config=load_config(ROOT/plan['tasks'][0]['config']);archive=archive_path(config)
    # Bounded read-throughput probe; not a claim about eight concurrent loaders.
    probe_file=archive if archive.is_file() else archive/'train.zip'
    started=time.perf_counter()
    with probe_file.open('rb') as f:read_bytes=len(f.read(32*2**20))
    io_report={'bytes':read_bytes,'seconds':time.perf_counter()-started,'scope':'32MiB sequential read, cache may apply'}
    write_json(out/'reports/host_resources.json',{'available_memory_bytes':mem['MemAvailable'],
               'cpu_count':os.cpu_count(),'free_disk_bytes':shutil.disk_usage(out).free,'io_probe':io_report})
    rows=read_split_rows(archive,'train');val=read_split_rows(archive,'val')
    if len(rows)!=30344 or len(val)!=1422:raise ValueError('Official image counts differ')
    if {r['Subject_id'] for r in rows}&{r['Subject_id'] for r in val}:raise ValueError('Train/Val patient leakage')
    data_identity=fingerprint({'train':rows,'val':val})
    for t in plan['tasks']:
        c=load_config(ROOT/t['config'])
        select_fold_rows(rows,c['data']['train_subset'],role='train')
        select_fold_rows(rows,c['data']['train_subset'],role='oof')
        _,_,name=imagenet_weight_info(t['backbone'])
        if digest(Path(cfg['pretrained_root'])/'hub/checkpoints'/name)!=c['model']['pretrained_sha256']:
            raise ValueError('Pretrained checksum mismatch')
        baseline=Path(cfg['baseline_results_root'])/'experiments'/t['baseline_id']
        history=json.loads((ROOT/'docs/phase32/history_inventory.json').read_text())
        records=history['tasks'] if isinstance(history,dict) else history
        record=next(r for r in records if r.get('id',r.get('model_id'))==t['baseline_id'])
        if digest(baseline/'resolved_config.yaml')!=record['config_sha256']:raise ValueError('Historical baseline config mismatch')
        if digest(baseline/'train_history.csv')!=record['history_sha256']:raise ValueError('Historical baseline history mismatch')
        for prefix in (['','ema_'] if t['ema'] else ['']):
            for folder in ('validation_analysis','oof_analysis'):
                if not (baseline/(prefix+folder)/'study_predictions.csv').is_file():
                    raise FileNotFoundError('Matching raw/EMA baseline predictions required for paired comparison')
    return {'host':platform.node(),'devices':{str(i):devices[i][:3] for i in cfg['gpu_ids']},'data_identity':data_identity,'plan':fingerprint(plan),
            'environment':subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)}

def preflight(cfg,plan,identity):
    from cxr.phase32.probe import verify_gh_preflight
    root=Path(cfg['output_root']);folder=root/'preflight';folder.mkdir(exist_ok=True)
    receipt=folder/'REMOTE_PREFLIGHT.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=identity:raise ValueError('Preflight identity changed; choose a new output root')
        for item in saved['probes']:verify_gh_preflight(root/item['path'],item['config_sha256'])
        return saved['gpu_hours']
    ledger=folder/'ledger.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else {'identity':identity,'gpu_hours':0,'probes':[]}
    if state['identity']!=identity:raise ValueError('Preflight assets changed')
    if state.get('running'):raise RuntimeError('Interrupted preflight: inspect child processes before clearing running ledger')
    for index,t in enumerate(plan['tasks']):
        if any(p['task']==t['id'] for p in state['probes']):continue
        reserve=plan['preflight_timeout_hours_per_task']
        if state['gpu_hours']+reserve>cfg['gpu_hours']:raise RuntimeError('GPU budget insufficient for preflight')
        probe_root=folder/t['id'];probe_root.mkdir(exist_ok=False)
        run_id=t['id']+'_PREFLIGHT';gpu=cfg['gpu_ids'][index%len(cfg['gpu_ids'])]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CXR_OUTPUT_ROOT=str(probe_root))
        state['running']=t['id'];write_json(ledger,state);started=time.monotonic()
        try:
            rc=run_command(['scripts/phase32_train.py','--config',str(ROOT/t['config']),'--run-id',run_id,'--preflight-steps','1'],
                env,probe_root/'stdout.log',probe_root/'stderr.log',reserve*3600)
            if rc:raise RuntimeError(f"Real probe failed: {t['id']}; see {probe_root/'stderr.log'}")
            run=probe_root/'experiments'/run_id
            verify_gh_preflight(run,t['config_sha256'])
            status=json.loads((run/'status.json').read_text())
            if status['gpu_peak_reserved_mib']>t['max_allowed_vram_gib']*1024:raise RuntimeError('Insufficient VRAM headroom')
            state['probes'].append({'task':t['id'],'gpu':gpu,'path':run.relative_to(root).as_posix(),'config_sha256':t['config_sha256']})
        finally:
            state['gpu_hours']+=(time.monotonic()-started)/3600;state.pop('running',None);write_json(ledger,state)
    write_json(receipt,state)
    return state['gpu_hours']
