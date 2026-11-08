"""Portable source identities and pre-registered task definitions."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PAIRS={'MIXUP_F02':('MIXUP',2), 'NO_PHOTO_F03':('NO_PHOTO',3)}


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_config(path):
    import yaml
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required=('data_root','pretrained_root','protocol_root','output_root','gpu_ids','max_concurrent','pair_gpu_hours','gpu_hours','stage')
    if any(k not in cfg for k in required):raise ValueError('Missing server configuration fields')
    if cfg['stage'] not in ('pilot','mixup_confirmation','no_photo_confirmation'):raise ValueError('Unknown stage')
    if not isinstance(cfg['max_concurrent'],int) or not 1<=cfg['max_concurrent']<=8:raise ValueError('Concurrency must be 1..8')
    ids=cfg['gpu_ids']
    if not ids or len(ids)!=len(set(ids)) or any(type(i)!=int or i<0 for i in ids):raise ValueError('Invalid GPU IDs')
    if cfg['max_concurrent']>len(ids):raise ValueError('More workers than GPUs')
    for key in ('data_root','pretrained_root','protocol_root','output_root'):
        p=Path(cfg[key]).expanduser()
        if not p.is_absolute() or 'AUTHORIZED' in str(p):raise ValueError('Fill absolute path: '+key)
        cfg[key]=str(p.resolve())
    cfg['holdout_index']=str(Path(cfg.get('holdout_index') or ROOT/'protocol_assets/confirmation_patient_split.csv').resolve())
    output=Path(cfg['output_root'])
    for key in ('data_root','pretrained_root','protocol_root'):
        p=Path(cfg[key]);
        if output==p or output in p.parents or p in output.parents:raise ValueError('Output overlaps source assets')
    return cfg


def task_specs(cfg):
    stage=cfg['stage']
    pairs=PAIRS if stage=='pilot' else {f'MIXUP_F{k:02d}':('MIXUP',k) for k in (3,4,5)} if stage=='mixup_confirmation' else {f'NO_PHOTO_F{k:02d}':('NO_PHOTO',k) for k in (4,5)}
    selected=cfg.get('pilot_pairs',list(PAIRS))
    if stage=='pilot':
        if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(PAIRS):raise ValueError('Invalid pilot subset')
        pairs={k:v for k,v in pairs.items() if k in selected}
    return [dict(id=k,variant=v[0],fold=v[1],timeout_hours=cfg['pair_gpu_hours'],max_retries=1) for k,v in pairs.items()]


def code_identity():
    from cxr.phase39_support import digest
    paths=list((ROOT/'src/cxr/regularization_confirmation').glob('*.py'))
    paths += [ROOT/p for p in ('scripts/confirmation_run.py','scripts/confirmation_worker.py','configs/confirmation_protocol.json',
        'src/cxr/data/dataset.py','src/cxr/data/archive.py','src/cxr/models/factory.py','src/cxr/phase39_run.py',
        'src/cxr/phase39_support.py','src/cxr/utils/training_resume.py','src/cxr/engine/metrics.py')]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}


def git_commit():
    return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()


def ensure_no_orphan(root):
    import socket
    from cxr.phase35.queue import alive
    for path in Path(root).rglob('*.pid.json'):
        item=json.loads(path.read_text(encoding='utf-8'))
        if item['host']!=socket.gethostname() or alive(item['pid']):raise RuntimeError('Existing training child: '+str(path))
        path.unlink()
    for path in (Path(root)/'experiments').glob('*/queue.lock'):
        pid=int(path.read_text())
        if alive(pid):raise RuntimeError('Pair still running: '+str(path))
        path.unlink()


def require_confirmation_gate(cfg):
    if cfg['stage']=='pilot':return
    from cxr.phase42.protocol import verify_receipt
    prior=Path(cfg.get('pilot_results_root',''))
    name='MIXUP_F02' if cfg['stage']=='mixup_confirmation' else 'NO_PHOTO_F03'
    folder=prior/'experiments'/name
    verify_receipt(folder,require_weights=False)
    report=json.loads((folder/'bootstrap_comparison.json').read_text(encoding='utf-8'))
    if report['decision']!='PROMISING_PILOT':raise ValueError('Frozen pilot gate did not pass')
    if report['delta']<.001:raise ValueError('Pilot delta below gate')
    per=__import__('csv')
    with (folder/'per_class_comparison.csv').open(encoding='utf-8',newline='') as f:
        rows=list(per.DictReader(f))
    if len(rows)!=10 or min(float(r['delta']) for r in rows)<-.01:raise ValueError('Pilot class guard failed')
