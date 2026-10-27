"""Frozen candidates, budgets and explicit confirmation authorization."""
import json
import math
from pathlib import Path
import yaml
from cxr.phase41.protocol import ROOT, digest, fingerprint, SOURCE_COUNTS


def load_plan():
    from cxr.phase42.models import IDS
    plan = yaml.safe_load((ROOT / 'configs/phase42_plan.yaml').read_text(encoding='utf-8'))
    if plan['schema'] != 'phase42_v1' or plan['pilot_fold'] != 2 or tuple(c['id'] for c in plan['candidates']) != IDS:
        raise ValueError('Frozen candidate roster or exploration Fold changed')
    if digest(ROOT / plan['protocol_config']) != plan['protocol_sha256']:
        raise ValueError('Phase39 source config changed')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / name) != sha: raise ValueError('Historical training implementation changed: ' + name)
    for item in plan['candidates']:
        cfg = yaml.safe_load((ROOT / 'configs/phase42' / (item['id'] + '.yaml')).read_text(encoding='utf-8'))
        if cfg != dict(candidate=item, fold=2, protocol='phase42_plan.yaml', seed=42, micro_batch=4,
                       accumulation=4, epochs=30, patience=6, head_seed=4201):
            raise ValueError('Candidate config differs from sealed plan')
    return plan


def task(plan, candidate, fold):
    if candidate not in {c['id'] for c in plan['candidates']} or fold not in range(1, 6):
        raise ValueError('Unknown candidate/Fold')
    return dict(id=f'P42_{candidate}_F{fold:02d}', variant=candidate, fold=fold,
                seed=42, timeout_hours=plan['timeout_hours'], max_retries=plan['max_retries'])


def tasks(plan, selected=None):
    if selected is None: return [task(plan, c['id'], plan['pilot_fold']) for c in plan['candidates']]
    if len(selected) != len(set(selected)) or len(selected) > 4: raise ValueError('At most four candidates')
    return [task(plan, c, f) for c in selected for f in range(1, 6) if f != plan['pilot_fold']]


def resolved_config(plan, task):
    cfg = yaml.safe_load((ROOT / plan['protocol_config']).read_text(encoding='utf-8'))
    cfg.update(experiment_id=task['id'], architecture='efficientnet_v2_s_' + task['variant'],
               variant=task['variant'], fold=task['fold'], selection_metric='fold_study_macro_auc',
               head_seed=4201, label_smoothing=.05 if task['variant'] == 'E03' else 0.,
               candidate_spec=next(c for c in plan['candidates'] if c['id'] == task['variant']))
    cfg['fold_expected_counts'] = dict(oof=plan['fold_counts'][task['fold']],
        train={k: SOURCE_COUNTS[k]-plan['fold_counts'][task['fold']][k] for k in SOURCE_COUNTS})
    return cfg


def server_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    paths = ('data_root','pretrained_root','protocol_root','phase41_results_root','output_root')
    for name in paths:
        if not isinstance(cfg.get(name), str) or not Path(cfg[name]).is_absolute():
            raise ValueError('Absolute authorized path required: ' + name)
    output=Path(cfg['output_root']).resolve()
    for source in (ROOT, Path(cfg['phase41_results_root'])):
        source=source.resolve()
        if output==source or source in output.parents or output in source.parents:
            raise ValueError('Phase42 output must be independent of source/Phase41')
    ids=cfg['gpu_ids']
    if not ids or len(ids)>8 or len(set(ids))!=len(ids) or any(type(i)!=int or i<0 for i in ids):
        raise ValueError('Authorize distinct physical GPU IDs')
    if type(cfg['max_concurrent'])!=int or not 1<=cfg['max_concurrent']<=len(ids):
        raise ValueError('Invalid maximum concurrency')
    for name in ('gpu_hours','pilot_gpu_hours','preflight_gpu_hours','max_vram_gib','minimum_free_disk_gib','minimum_free_memory_gib_per_worker'):
        v=cfg.get(name)
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:
            raise ValueError('Explicit positive finite resource budget required: '+name)
    if cfg['max_vram_gib']>22: raise ValueError('Keep GPU memory safety margin below 22 GiB')
    confirm=cfg['confirmation']
    if type(confirm['enabled']) is not bool: raise ValueError('Confirmation needs explicit boolean authorization')
    hours=confirm['gpu_hours']
    if type(hours) not in (int,float) or not math.isfinite(hours) or hours<0 or (confirm['enabled'] and hours<=0):
        raise ValueError('Authorized confirmation GPU-hour budget required')
    if cfg['pilot_gpu_hours']+cfg['preflight_gpu_hours']+hours>cfg['gpu_hours']:
        raise ValueError('Stage reservations exceed total GPU budget')
    if cfg['baseline_source_commit']!='76456056b2c7a8ebc426ad473f7556b025f7ce68':
        raise ValueError('Phase41 baseline provenance declaration differs')
    up=cfg['upload']
    if up['enabled'] and (up['repository']!='Zoey-711/cxr-multilabel-challenge' or
        up['branch']!='results/phase42-pilots' or up['release_tag']!='phase42-pilots-v1' or not Path(up['checkout']).is_absolute()):
        raise ValueError('Dedicated private Phase42 result destination required')
    return cfg


def verify_receipt(folder, name='DONE.json', require_weights=True):
    folder=Path(folder)
    record=json.loads((folder/name).read_text(encoding='utf-8'))
    if not record.get('sha256'): raise ValueError('Empty integrity receipt')
    for rel,sha in record['sha256'].items():
        path=(folder/rel).resolve()
        if folder.resolve() not in path.parents: raise ValueError('Unsafe manifest path')
        if not path.exists() and not require_weights and path.suffix in ('.pth','.pt'): continue
        if digest(path)!=sha: raise ValueError('Artifact hash mismatch: '+str(path))
    return record


def baseline_folder(cfg, fold):
    root=Path(cfg['phase41_results_root'])
    return root/'reused_fold1/P39_BASE_F1' if fold==1 else root/'experiments'/f'P41_BASE_F{fold:02d}'


def verify_baseline(plan, cfg, fold, require_weights=True):
    folder=baseline_folder(cfg,fold)
    receipt=verify_receipt(folder,'REUSED.json' if fold==1 else 'DONE.json',require_weights)
    filename='resolved_config.json' if fold==1 else 'resolved_config.yaml'
    actual=yaml.safe_load((folder/filename).read_text(encoding='utf-8'))
    reference=yaml.safe_load((ROOT/plan['protocol_config']).read_text(encoding='utf-8'))
    for key,value in reference.items():
        if key in ('experiment_id','fold','selection_metric'): continue
        if actual.get(key)!=value: raise ValueError(f'BASE protocol differs: fold={fold}, {key}')
    if actual['fold']!=fold: raise ValueError('BASE fold mismatch')
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    if fold==1:
        for rel,sha in plan['baseline_fold1']['assets'].items():
            if not require_weights and rel.endswith('.pth') and not (folder/rel).exists(): continue
            if digest(folder/rel)!=sha: raise ValueError('Frozen Phase39 BASE changed')
    else:
        preflight=Path(cfg['phase41_results_root'])/'preflight/REMOTE_PREFLIGHT.json'
        identity=json.loads(preflight.read_text(encoding='utf-8'))
        if identity.get('state')!='PASS' or identity['identity_sha256']!=receipt['identity_sha256']:
            raise ValueError('BASE receipt is not bound to the completed Phase41 preflight')
        if manifest['task']!=f'P41_BASE_F{fold:02d}' or manifest['variant']!='BASE' or manifest['fold_sha256']!=plan['fold_sha256']:
            raise ValueError('BASE training provenance differs')
        if manifest['official_val_used_for_training'] or manifest['test_labels_used']:
            raise ValueError('BASE selected with prohibited source')
    return folder, receipt
