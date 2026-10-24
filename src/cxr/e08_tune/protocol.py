"""Exact five interventions; no implicit baseline retraining or parameter search."""
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,digest
from cxr.phase32.protocol import differences,fingerprint

FACTORS={'DECAY_LOW':{'training.ema.decay'},'DECAY_HIGH':{'training.ema.decay'},
         'LR_LOW':{'training.backbone_lr','training.classifier_lr'},
         'LR_HIGH':{'training.backbone_lr','training.classifier_lr'},'LONG':{'training.epochs'}}
VALUES={'DECAY_LOW':.998,'DECAY_HIGH':.9995,'LR_LOW':.7,'LR_HIGH':1.3,'LONG':40}

def load_plan(path=None):
    plan=yaml.safe_load(Path(path or ROOT/'configs/e08_tune_plan.yaml').read_text(encoding='utf-8'))
    expected={f'ET_{v}_F{f:02d}' for v in FACTORS for f in range(1,6)}
    if plan['schema']!='e08_tune_v1' or len(plan['tasks'])!=25 or {t['id'] for t in plan['tasks']}!=expected:
        raise ValueError('Frozen five pilots and their conditional four-fold templates required')
    for task in plan['tasks']:
        path=ROOT/task['config'];cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        base=yaml.safe_load((ROOT/task['baseline_config']).read_text(encoding='utf-8'))
        variant=task['variant'];training=cfg['training']
        if digest(path)!=task['config_sha256']:raise ValueError('Config checksum changed')
        if differences(base,cfg)-{'experiment.name','e08_tune'} != FACTORS[variant]:raise ValueError('More than declared factor changed')
        if variant.startswith('DECAY') and training['ema']['decay']!=VALUES[variant]:raise ValueError('EMA changed')
        if variant.startswith('LR'):
            for k in ('backbone_lr','classifier_lr'):
                if abs(training[k]/base['training'][k]-VALUES[variant])>1e-10:raise ValueError('LR factor changed')
        if variant=='LONG' and training['epochs']!=40:raise ValueError('Schedule horizon changed')
        subset=cfg['data']['train_subset']
        if subset['fold']!=task['fold'] or subset['sha256']!=plan['fold_hashes']:raise ValueError('Fold protocol changed')
        task['expected_counts']=subset['expected_counts']
    for rel,h in plan['code_hashes'].items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Code seal mismatch: {rel}')
    return plan

def server_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('data_root','pretrained_root','protocol_root','baseline_results_root','output_root'):
        p=Path(cfg[key]).expanduser()
        if not p.is_absolute():raise ValueError('Use absolute server paths')
        cfg[key]=str(p.resolve())
    if Path(cfg['output_root']).name!='e08_tuning':raise ValueError('Dedicated e08_tuning output root required')
    ids=cfg['gpu_ids']
    if not ids or len(set(ids))!=len(ids) or any(type(i)!=int or i<0 for i in ids):raise ValueError('Explicit GPU IDs required')
    if not 1<=cfg['max_concurrent']<=min(8,len(ids)):raise ValueError('Invalid concurrency')
    for k in ('gpu_hours','extension_gpu_hours'):
        if type(cfg.get(k)) not in (float,int) or not 0<=cfg[k]<10000:raise ValueError('Explicit GPU budgets required')
    if cfg['gpu_hours']<=0 or cfg['extension_gpu_hours']>cfg['gpu_hours']:raise ValueError('Invalid total/sub-budget')
    return cfg
