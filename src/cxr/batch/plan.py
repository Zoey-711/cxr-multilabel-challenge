"""Fail-closed plan validation. No CUDA initialization or training here."""
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema') != 'cxr_batch_v1':
        raise ValueError('Unknown batch plan schema')
    host=plan.get('server_architecture',{})
    if (host.get('type')!='single_linux_host' or host.get('authorized_gpu_count_max')!=8 or
            host.get('gpu_memory_gib')!=24 or '4090' not in host.get('gpu_model','')):
        raise ValueError('Single-host RTX4090 eight-GPU architecture changed')
    if (plan.get('expected_test_studies'),plan.get('expected_test_images'))!=(1414,2218):
        raise ValueError('Official unlabeled Test coverage changed')
    tasks = plan['tasks']
    for tid, expected in plan.get('original_A_config_hashes', {}).items():
        task = next(t for t in tasks if t['id'] == tid)
        if task['config_sha256'] != expected:
            raise ValueError('Original A configuration changed')
    ids = [t['id'] for t in tasks]
    groups = {g:sum(t['group']==g for t in tasks) for g in ('A','N','P')}
    if groups!={'A':15,'N':5,'P':2} or len(tasks)!=22:
        raise ValueError('Reviewed 22-task inventory changed')
    extension = plan.get('conditional_extension', {})
    slots = extension.get('reserved_slots', [])
    if (extension.get('enabled') is not False or extension.get('max_additional_tasks') != 4 or
            [slot.get('id') for slot in slots] != ['G17_FULL_01', 'G17_FULL_02', 'H17_DISTILL_01', 'H17_DISTILL_02'] or
            any(slot.get('enabled') is not False or slot.get('implementation_status') != 'CPU_VERIFIED_GPU_PENDING'
                for slot in slots)):
        raise ValueError('G/H extension may run only by explicit opt-in after its reviewed configuration')
    required = {f'{prefix}_F{fold:02d}' for prefix in ('D17', 'C17', 'S17', 'N17') for fold in range(1, 6)}
    if any(set(slot.get('depends_on', [])) != required for slot in slots):
        raise ValueError('G/H tasks must await all 20 five-fold models')
    if any(slot.get('fold') is not None for slot in slots):
        raise ValueError('G/H full-Train tasks cannot claim an OOF fold')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs')
    import re
    for t in tasks:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', t['id']):
            raise ValueError('Unsafe task ID')
        if t['max_retries'] not in (0, 1) or t['timeout_hours'] <= 0:
            raise ValueError('Retry/timeout bound missing')
        if t['gpu'] != 'allocated_pool' or t['group'] not in ('A', 'C', 'N', 'P'):
            raise ValueError('Unapproved GPU or experiment group')
        if t['output']!='experiments/'+t['id']:
            raise ValueError('Task ID and unique output directory differ')
        if any(d not in ids or d == t['id'] for d in t['depends_on']):
            raise ValueError('Unknown/self dependency')
        cfg_path = ROOT / t['config']
        if digest(cfg_path) != t['config_sha256']:
            raise ValueError(f'Changed config: {t["id"]}')
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        if cfg['experiment']['seed'] != t['seed'] or cfg['model']['name'] != t['backbone']:
            raise ValueError('Model/seed mismatch')
        if cfg['model']['pretrained']!='imagenet' or cfg['model']['pretrained_sha256']!=t['pretrained_sha256']:
            raise ValueError('Unapproved pretrained source/hash')
        if cfg['data']['input_size']!=384 or cfg['training']['loss']!='bce_with_logits' or cfg['validation']['study_fusion']!='mean_probability':
            raise ValueError('Resolution/loss/Study protocol changed')
        if bool(cfg['training'].get('ema',{}).get('enabled'))!=t['ema']:
            raise ValueError('EMA policy and task declaration differ')
        if cfg['training']['early_stopping_patience']!=(8 if t['backbone']=='densenet121' else 6):
            raise ValueError('Early stopping patience differs from frozen protocol')
        if cfg['training']['batch_size'] * cfg['training'].get('gradient_accumulation_steps', 1) != 16:
            raise ValueError('Effective batch must remain 16')
        subset = cfg['data'].get('train_subset')
        if t['group'] in ('A','N','P') and (not subset or subset['fold'] != t['fold']):
            raise ValueError('Fold mismatch')
        if t['group'] == 'C' and (t['enabled'] or subset is not None):
            raise ValueError('Optional full-Train seeds must be disabled by default')
        for name, expected in plan['fold_hashes'].items():
            if digest(ROOT / plan['fold_directory'] / name) != expected:
                raise ValueError(f'Frozen index changed: {name}')
            if subset and subset['sha256'][name] != expected:
                raise ValueError('Backbones do not share frozen folds')
    done = set()
    while len(done) < len(ids):
        ready = {t['id'] for t in tasks if t['id'] not in done and set(t['depends_on']) <= done}
        if not ready:
            raise ValueError('Dependency cycle')
        done |= ready
    for path, expected in plan['artifacts'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Changed frozen batch artifact: {path}')
    return plan


def selected_tasks(plan, extended=False):
    if extended and 'conditional_extension' in plan:
        raise ValueError('Use run-extension with separate G/H GPU-hour authorization after the 22-task queue')
    return [t for t in plan['tasks'] if t['enabled'] or (extended and t['group'] == 'C')]


def gpu_tokens(requested):
    tokens = requested.split(',')
    if not tokens or any(not x or '/' in x or '\\' in x for x in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError('GPU pool must have unique device IDs/UUIDs')
    allocation = os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocation is not None and not set(tokens) <= set(allocation.split(',')):
        raise ValueError('Requested GPUs exceed CUDA_VISIBLE_DEVICES allocation')
    return tokens


def deployment_bytes(paths, limit=500_000_000):
    unique = {Path(p).resolve() for p in paths}
    total = sum(p.stat().st_size for p in unique)
    return {'bytes': total, 'limit_bytes': limit, 'fits': total <= limit, 'models': len(unique)}


def retryable(message, returncode):
    low = message.lower()
    if any(x in low for x in ('out of memory', 'non-finite', 'nan', 'hash mismatch', 'leakage', 'timeout')):
        return False
    return returncode == 75 or any(x in low for x in ('input/output error', 'stale file handle', 'temporarily unavailable'))
