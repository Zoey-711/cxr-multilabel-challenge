"""Read-only prerequisites for the proposed G/H full-Train extension.

This module does not select a backbone, create teacher targets, or start training.
"""
import csv
import json
import statistics
import math
from pathlib import Path

import numpy as np
import yaml

from cxr.batch.artifacts import label_keys, read_probabilities, write_probabilities
from cxr.batch.plan import digest, write_json
from cxr.batch.results import combine, scores


def _rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def verify_teacher_sources(root, branches=('D17', 'C17', 'S17', 'N17'), *, expected_studies=18513):
    """Verify every teacher OOF row came from the patient's unseen fold.

    Returns file hashes for an eventual reviewed distillation recipe. No labels or
    probabilities from Val/Test are used as training targets.
    """
    root = Path(root)
    train_labels = root/'manifests/train_labels.csv'
    patient_rows = _rows(root/'manifests/patient_folds.csv')
    patient_fold = {row['Subject_id']: int(row['fold']) for row in patient_rows}
    if len(patient_fold) != len(patient_rows):
        raise ValueError('Duplicate patient in frozen folds')
    expected = label_keys(train_labels)
    if len(expected) != expected_studies:
        raise ValueError('Frozen Train Study count changed')
    if {subject for subject, _ in expected} != set(patient_fold):
        raise ValueError('Train patients differ from frozen folds')
    manifest = {}
    for branch in branches:
        probability_file = root/'oof'/f'{branch}_oof_all.csv'
        source_file = root/'oof'/f'{branch}_source_index.csv'
        table = read_probabilities(probability_file, train_labels, expected)
        source_rows = _rows(source_file)
        sources = {}
        for row in source_rows:
            key = row['Subject_id'], row['Study_id']
            if key in sources:
                raise ValueError(f'Duplicate teacher source: {branch} {key}')
            fold = int(row['fold'])
            if fold != patient_fold.get(key[0]) or row['model_id'] != f'{branch}_F{fold:02d}':
                raise ValueError(f'Teacher trained on or mismatched patient fold: {branch} {key}')
            sources[key] = row['model_id']
        if len(table['keys']) != expected_studies or set(sources) != expected:
            raise ValueError(f'Incomplete teacher OOF or source index: {branch}')
        for fold in range(1, 6):
            run = root/'experiments'/f'{branch}_F{fold:02d}'
            if not (run/'DONE.json').is_file():
                raise ValueError(f'Teacher fold incomplete: {branch}_F{fold:02d}')
            trained = {row['Subject_id'] for row in _rows(run/'training_image_index.csv')}
            held_out = {subject for subject, assigned in patient_fold.items() if assigned == fold}
            if trained & held_out or trained != set(patient_fold)-held_out:
                raise ValueError(f'Teacher training patients violate fold isolation: {branch}_F{fold:02d}')
        manifest[branch] = {'probabilities_sha256': digest(probability_file),
                            'source_index_sha256': digest(source_file),
                            'studies': len(table['keys']),
                            'patients': len({subject for subject, _ in table['keys']})}
    return manifest


BRANCHES = ('D17', 'C17', 'S17', 'N17')
TEACHER_BRANCHES = ('D17', 'C17', 'S17')
TEACHER_WEIGHTS = (0.25, 0.25, 0.50)


def median_best_epoch(root, branch):
    epochs = []
    for fold in range(1, 6):
        run = Path(root)/'experiments'/f'{branch}_F{fold:02d}'
        if not (run/'DONE.json').is_file():
            raise ValueError(f'Incomplete fold: {run.name}')
        epochs.append(int(json.loads((run/'metrics.json').read_text(encoding='utf-8'))['best_epoch']))
    value = int(statistics.median(epochs))
    if not 1 <= value <= 30:
        raise ValueError('Median best Epoch outside frozen 1..30 training budget')
    return {'epochs': value, 'fold_best_epochs': epochs, 'rule': 'median of five raw best epochs; five values imply exact integer'}


def rank_full_train_backbones(root, *, branches=BRANCHES, max_bytes=500_000_000,
                              research_p95_limit_ms=100.0):
    """Predeclared OOF ranking; 4090 latency is only a research screen, never T4 proof."""
    root = Path(root)
    labels = root/'manifests/train_labels.csv'
    expected = label_keys(labels)
    ranked = []
    for priority, branch in enumerate(branches):
        oof = root/'oof'/f'{branch}_oof_all.csv'
        source = root/'oof'/f'{branch}_source_index.csv'
        if not oof.is_file() or not source.is_file():
            raise ValueError(f'Incomplete five-fold OOF: {branch}')
        # Verify all five models and the patient's original held-out fold.
        verify_teacher_sources(root, (branch,), expected_studies=len(expected))
        table = read_probabilities(oof, labels, expected)
        fold_sizes, fold_latency = [], []
        for fold in range(1, 6):
            run = root/'experiments'/f'{branch}_F{fold:02d}'
            size = (run/'best_model.pth').stat().st_size
            measure = json.loads((run/'latency.json').read_text(encoding='utf-8'))
            if '4090' not in str(measure.get('device', '')) or measure.get('batch') != 1:
                raise ValueError(f'Missing single-4090 research timing: {run.name}')
            fold_sizes.append(size)
            fold_latency.append(float(measure['p95_ms']))
        auc = scores(table)['macro_auc']
        if auc is None:
            raise ValueError(f'Cannot rank OOF with missing class AUC: {branch}')
        ranked.append({'branch': branch, 'oof_macro_auc': auc,
                       'maximum_fold_weight_bytes': max(fold_sizes),
                       'maximum_4090_p95_ms': max(fold_latency),
                       'eligible_research_screen': max(fold_sizes) <= max_bytes and
                           max(fold_latency) < research_p95_limit_ms,
                       'official_target_latency_status': 'LATENCY_UNVERIFIED',
                       'priority': priority, 'oof_sha256': digest(oof),
                       'source_index_sha256': digest(source)})
    eligible = sorted((row for row in ranked if row['eligible_research_screen']),
                      key=lambda row: (-row['oof_macro_auc'], row['priority']))
    if len(eligible) < 2:
        raise ValueError('Fewer than two research-screen eligible full-Train backbones')
    return {'selected': [row['branch'] for row in eligible[:2]], 'ranking': ranked,
            'rule': 'size <=500000000 bytes and single-4090 p95 <100ms research screen; '
                    'rank by complete five-fold Train OOF Macro-AUC descending; exact ties D,C,S,N',
            'deployment_note': '4090 timing is never official target-hardware compliance'}


def build_teacher_oof(root, *, expected_studies=18513):
    root = Path(root)
    verified = verify_teacher_sources(root, TEACHER_BRANCHES, expected_studies=expected_studies)
    labels = root/'manifests/train_labels.csv'
    expected = label_keys(labels)
    tables = [read_probabilities(root/'oof'/f'{branch}_oof_all.csv', labels, expected)
              for branch in TEACHER_BRANCHES]
    teacher = combine(tables, TEACHER_WEIGHTS)
    path = root/'analysis/gh_teacher_oof.csv'
    provenance = {'formula': {'D17': .25, 'C17': .25, 'S17': .5},
                  'source': verified, 'role': 'training soft targets; unseen patient folds only'}
    if path.exists():
        old = read_probabilities(path, labels, expected)
        if not np.array_equal(old['p'], teacher['p']):
            raise ValueError('Existing teacher OOF differs from frozen formula')
        recorded = json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        if recorded['csv_sha256'] != digest(path) or recorded['source'] != verified:
            raise ValueError('Existing teacher OOF provenance changed')
    else:
        write_probabilities(path, teacher, provenance)
    return {'path': str(path.resolve()), 'sha256': digest(path),
            'source': verified, 'formula': provenance['formula'],
            'studies': len(teacher['keys'])}


def load_soft_studies(path, expected_hash, train_rows, labels_path):
    if digest(path) != expected_hash:
        raise ValueError('Teacher OOF SHA-256 changed')
    studies = {(r['Subject_id'], r['Study_id']) for r in train_rows}
    table = read_probabilities(path, labels_path, studies)
    counts = {}
    for row in train_rows:
        key = row['Subject_id'], row['Study_id']
        counts[key] = counts.get(key, 0) + 1
    targets = {key: vector for key, vector in zip(table['keys'], table['p'])}
    weights = {key: 1.0/count for key, count in counts.items()}
    if len(targets) != len(studies) or not np.isfinite(table['p']).all():
        raise ValueError('Teacher soft target coverage or values invalid')
    return targets, weights


def blended_bce(logits, hard, mask, soft, per_image_study_weight,
                *, hard_fraction=0.5, temperature=1.0):
    """Reference loss for tests; train loop uses the same two group denominators."""
    import torch
    import torch.nn.functional as F
    if logits.shape != hard.shape or hard.shape != soft.shape or hard.shape[-1] != 10:
        raise ValueError('Distillation requires aligned ten-class logits and targets')
    if per_image_study_weight.shape != (len(logits),) or temperature != 1.0 or not 0 <= hard_fraction <= 1:
        raise ValueError('Unapproved distillation shape or hyperparameters')
    valid = mask.float()
    weighted = valid * per_image_study_weight[:, None]
    if valid.sum() <= 0 or weighted.sum() <= 0:
        raise ValueError('Empty effective labels')
    hard_loss = (F.binary_cross_entropy_with_logits(logits, hard, reduction='none')*valid).sum()/valid.sum()
    soft_loss = (F.binary_cross_entropy_with_logits(logits/temperature, soft, reduction='none')*weighted).sum()/weighted.sum()
    return hard_fraction*hard_loss+(1-hard_fraction)*soft_loss


def prepare_extension(root, plan):
    """Freeze four runtime configs only after all 22 base DONE markers exist."""
    root = Path(root).resolve()
    for task in plan['tasks']:
        if not (root/'experiments'/task['id']/'DONE.json').is_file():
            raise ValueError(f'Base task incomplete: {task["id"]}')
        from cxr.batch.results import verify_frozen
        verify_frozen(root/'experiments'/task['id'])
    selection = rank_full_train_backbones(root)
    teacher = build_teacher_oof(root)
    branch_tasks = {name: next(t for t in plan['tasks'] if t['id'] == f'{name}_F01')
                    for name in BRANCHES}
    import cxr.batch.plan as batch_plan
    configurations = [('G17_FULL_01', selection['selected'][0], 'full_train'),
                      ('G17_FULL_02', selection['selected'][1], 'full_train'),
                      ('H17_DISTILL_01', 'D17', 'distillation'),
                      ('H17_DISTILL_02', 'N17', 'distillation')]
    directory = root/'manifests/extension_configs'
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id, branch, mode in configurations:
        source = branch_tasks[branch]
        cfg = yaml.safe_load((batch_plan.ROOT/source['config']).read_text(encoding='utf-8'))
        cfg['experiment']['name'] = task_id.lower()
        cfg['data'].pop('train_subset')
        cfg['data']['full_train_expected_counts'] = {'patients': 11055, 'studies': 18513, 'images': 30344}
        epoch_rule = median_best_epoch(root, branch)
        cfg['training']['epochs'] = epoch_rule['epochs']
        cfg['training']['epoch_selection'] = epoch_rule
        fold_hours_per_epoch=[]
        for fold in range(1, 6):
            metrics=json.loads((root/'experiments'/f'{branch}_F{fold:02d}'/'metrics.json').read_text(encoding='utf-8'))
            stop_epoch=int(metrics['stop_epoch'])
            epoch_hours=float(metrics['resources']['epoch_hours'])
            if stop_epoch < 1 or not math.isfinite(epoch_hours) or epoch_hours <= 0:
                raise ValueError(f'Invalid measured fold training time: {branch}_F{fold:02d}')
            fold_hours_per_epoch.append(epoch_hours/stop_epoch)
        train_images=int(yaml.safe_load((batch_plan.ROOT/source['config']).read_text(encoding='utf-8'))
                         ['data']['train_subset']['expected_counts']['train']['images'])
        estimated_train_hours=(statistics.median(fold_hours_per_epoch)*30344/train_images*
                               epoch_rule['epochs']*(1.15 if mode=='distillation' else 1.0))
        estimate={'gpu_hours_train_only': round(estimated_train_hours,3),
                  'basis':'median measured 4090 fold GPU-hours per Epoch x 30344/fold train images '
                          'x median-best Epoch; H includes provisional 15% loss overhead; '
                          'Val/Test/latency inference excluded'}
        if mode == 'distillation':
            cfg['data']['teacher_soft_labels'] = {'path': teacher['path'], 'sha256': teacher['sha256'],
                                                   'formula': teacher['formula'],
                                                   'label_sha256': digest(root/'manifests/train_labels.csv')}
            cfg['training']['distillation'] = {'temperature': 1.0, 'hard_fraction': 0.5,
                'soft_fraction': 0.5, 'soft_weighting': 'inverse_images_per_study',
                'normalization': 'separate hard image-label and soft Study-label denominators'}
        config_path = directory/f'{task_id}.yaml'
        encoded = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode('utf-8')
        if config_path.exists() and config_path.read_bytes() != encoded:
            raise ValueError(f'Frozen extension config differs: {task_id}')
        if not config_path.exists():
            config_path.write_bytes(encoded)
        tasks.append({'id': task_id, 'group': mode, 'enabled': True,
                      'backbone': cfg['model']['name'], 'fold': None, 'seed': cfg['experiment']['seed'],
                      'config': str(config_path), 'config_sha256': digest(config_path),
                      'source_config': source['config'], 'source_config_sha256': source['config_sha256'],
                      'pretrained_sha256': source['pretrained_sha256'],
                      'ema': bool(cfg['training'].get('ema', {}).get('enabled')),
                      'depends_on': [f'{branch}_F{i:02d}' for i in range(1, 6)] if mode == 'full_train' else
                                    [f'{b}_F{i:02d}' for b in TEACHER_BRANCHES for i in range(1, 6)],
                      'gpu': 'allocated_pool', 'minimum_vram_gib': source['minimum_vram_gib'],
                      'timeout_hours': max(source['timeout_hours'], math.ceil(estimated_train_hours*1.5)),
                      'estimated_gpu_hours': estimate, 'max_retries': 1,
                      'output': f'experiments/{task_id}', 'fold_hashes': plan['fold_hashes'],
                      'oof_status': 'N/A', 'estimated_cpu_cores': source['estimated_cpu_cores'],
                      'estimated_host_ram_gib': source['estimated_host_ram_gib']})
    frozen = {'selection': selection, 'teacher': teacher,
              'tasks': [{key: value for key, value in task.items() if key != 'depends_on'} for task in tasks],
              'source_fold_hashes': plan['fold_hashes'], 'full_train_oof': 'N/A'}
    manifest = root/'manifests/extension_selection.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Existing G/H selection manifest differs')
    if not manifest.exists():
        write_json(manifest, frozen)
    return tasks
