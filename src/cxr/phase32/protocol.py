"""Frozen Phase32 plan validation; no CUDA or training side effects."""
import hashlib
import json
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT, digest

CHANGES = {
 'P32_CS_LR': {'training.backbone_lr'}, 'P32_NM_LR': {'training.backbone_lr'},
 'P32_NS_448': {'data.input_size'}, 'P32_NS_512': {'data.input_size'},
 'P32_NS_ASL': {'training.loss','training.asl'},
 'P32_NS_POS': {'training.loss','training.positive_weight'},
 'P32_NS_PHOTO': {'data.train_augmentation.photometric_probability'},
 'P32_NS_MIXUP': {'training.mixup'},
}

def differences(a,b,prefix=''):
    out=set()
    for k in set(a)|set(b):
        name=prefix+k
        if k not in a or k not in b: out.add(name)
        elif isinstance(a[k],dict) and isinstance(b[k],dict): out |= differences(a[k],b[k],name+'.')
        elif a[k]!=b[k]: out.add(name)
    return out

def load_plan(path=None):
    plan=yaml.safe_load(Path(path or ROOT/'configs/phase32_plan.yaml').read_text(encoding='utf-8'))
    if plan['schema']!='cxr_phase32_v1' or {t['id'] for t in plan['tasks']}!=set(CHANGES) or len(plan['tasks'])!=8:
        raise ValueError('Exactly the eight approved pilots are required')
    for t in plan['tasks']:
        cfg=yaml.safe_load((ROOT/t['config']).read_text(encoding='utf-8'))
        baseline=yaml.safe_load((ROOT/t['baseline_config']).read_text(encoding='utf-8'))
        if digest(ROOT/t['config'])!=t['config_sha256']: raise ValueError('Config hash mismatch')
        diff=differences(baseline,cfg)-{'experiment.name','phase32'}
        if diff!=CHANGES[t['id']]: raise ValueError(f"Unapproved differences {t['id']}: {diff}")
        if cfg['data']['train_subset']['sha256']!=plan['fold_hashes'] or t['fold']!=1:
            raise ValueError('Frozen patient folds changed')
        if cfg['training']['batch_size']*cfg['training']['gradient_accumulation_steps']!=16:
            raise ValueError('Effective batch must remain 16')
    for rel,h in plan['code_hashes'].items():
        if digest(ROOT/rel)!=h: raise ValueError(f'Code seal mismatch: {rel}')
    return plan

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def server_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('data_root','pretrained_root','protocol_root','baseline_results_root','output_root'):
        p=Path(cfg[key]).expanduser()
        if not p.is_absolute(): raise ValueError(f'{key} must be absolute')
        cfg[key]=str(p.resolve())
    gpus=cfg['gpu_ids']
    if not isinstance(gpus,list) or not gpus or len(set(gpus))!=len(gpus) or any(type(x)!=int or x<0 for x in gpus):
        raise ValueError('Explicit unique physical GPU IDs required')
    if not 1<=cfg['max_concurrent']<=min(8,len(gpus)): raise ValueError('Concurrency outside authorization')
    if not isinstance(cfg.get('gpu_hours'),(int,float)) or not 0<cfg['gpu_hours']<10000:
        raise ValueError('Explicit positive GPU-hour authorization required')
    if Path(cfg['output_root']).name!='phase32_batch': raise ValueError('Use a dedicated phase32_batch output root')
    return cfg
