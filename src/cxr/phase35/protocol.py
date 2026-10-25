"""Exactly eight new tasks, byte-pinned Fold1 reuse, explicit resources."""
import json,math,shutil
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,digest,write_json
from cxr.phase32.protocol import differences,fingerprint

def load_plan(path=None):
    plan=yaml.safe_load(Path(path or ROOT/'configs/phase35_plan.yaml').read_text(encoding='utf-8'))
    expected={f'P35_{v}_F{f:02d}' for v in ('POS','RES512') for f in range(2,6)}
    if plan['schema']!='phase35_v1' or len(plan['tasks'])!=8 or {t['id'] for t in plan['tasks']}!=expected:
        raise ValueError('Exactly eight Fold2-5 training tasks required; Fold1 must never retrain')
    for variant,reuse in plan['reused_fold1'].items():
        if digest(ROOT/reuse['config'])!=reuse['config_sha256']:raise ValueError('Frozen Fold1 config changed')
    for t in plan['tasks']:
        cfg=yaml.safe_load((ROOT/t['config']).read_text());reuse=plan['reused_fold1'][t['variant']]
        base=yaml.safe_load((ROOT/reuse['config']).read_text())
        allowed={'experiment.name','data.train_subset.fold','phase32.baseline_id'}
        diff=differences(base,cfg)
        if any(k not in allowed and not k.startswith('data.train_subset.expected_counts.') for k in diff):
            raise ValueError('Training intervention drifted from frozen Fold1')
        if digest(ROOT/t['config'])!=t['config_sha256']:raise ValueError('Task config hash mismatch')
        s=cfg['data']['train_subset']
        if s['fold']!=t['fold'] or s['sha256']!=plan['fold_hashes'] or s['expected_counts']!=t['expected_counts']:raise ValueError('Fold mismatch')
        if t['depends_on']!=[reuse['id']] or t['baseline_id']!=f'N17_F{t["fold"]:02d}':raise ValueError('Dependency mismatch')
        if cfg['training']['batch_size']*cfg['training']['gradient_accumulation_steps']!=16:raise ValueError('Effective batch changed')
    for rel,h in plan['code_hashes'].items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Code seal mismatch: {rel}')
    return plan

def server_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    paths=('data_root','pretrained_root','protocol_root','baseline_results_root','phase32_results_root','output_root')
    for k in paths:
        p=Path(cfg[k]).expanduser()
        if not p.is_absolute():raise ValueError('Absolute authorized server paths required')
        cfg[k]=str(p.resolve())
    output=Path(cfg['output_root'])
    if output.name!='phase35_batch':raise ValueError('Independent phase35_batch output root required')
    for k in paths[:-1]:
        source=Path(cfg[k])
        if output==source or output in source.parents or source in output.parents:raise ValueError('Output overlaps input/source tree')
    ids=cfg['gpu_ids']
    if not ids or len(ids)>8 or len(set(ids))!=len(ids) or any(type(i)!=int or i<0 for i in ids):raise ValueError('Explicit unique GPU IDs required, at most eight')
    if type(cfg['max_concurrent'])!=int or not 1<=cfg['max_concurrent']<=min(8,len(ids)):raise ValueError('Concurrency outside authorization')
    if type(cfg.get('gpu_hours')) not in (int,float) or not math.isfinite(cfg['gpu_hours']) or not 0<cfg['gpu_hours']<10000:
        raise ValueError('Positive finite authorized GPU hours required')
    for key in ('minimum_free_disk_gib','minimum_free_memory_gib_per_worker'):
        if type(cfg.get(key)) not in (int,float) or not math.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('Resource headroom required')
    return cfg

def reuse_fold1(cfg,plan):
    """Copy only pinned inference and result files. Never write the original Pilot directory."""
    for variant,info in plan['reused_fold1'].items():
        source=Path(cfg['phase32_results_root'])/'experiments'/info['id']
        target=Path(cfg['output_root'])/'reused_fold1'/info['id']
        for rel,h in info['sha256'].items():
            p=source/rel
            if not p.is_file() or digest(p)!=h:raise ValueError(f'Missing/changed frozen Fold1 artifact: {p}')
            q=target/rel
            if q.exists() and digest(q)!=h:raise ValueError('Reused Fold1 copy changed; no overwrite')
            if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            if digest(q)!=h:raise ValueError('Copy checksum mismatch')
        # Test is generated only in this independent copy, from pure inference weights.
        status=target/'status.json'
        if not status.exists():write_json(status,{'state':'complete','training':'REUSED_FROZEN_FOLD1_NO_TRAINING'})
        write_json(target/'REUSED.json',{'state':'REUSED','variant':variant,'source_commit':plan['pilot_results_commit'],
                   'source_id':info['id'],'sha256':info['sha256']})

def verify_reused(run,info):
    for rel,h in info['sha256'].items():
        if digest(Path(run)/rel)!=h:raise ValueError('Frozen Fold1 reused file changed')
