"""Strict Study alignment, fixed ensembles and patient-cluster uncertainty."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cxr.batch.plan import ROOT, digest, write_json, deployment_bytes
from cxr.engine.b01_analysis import read_study_predictions, binary_auc
from cxr.engine.average_precision import average_precision


def read_table(path):
    p, y, subjects, studies = read_study_predictions(Path(path))
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(set(studies)) != len(studies):
        raise ValueError('Study ID assigned to multiple subjects')
    return {'p': p, 'y': y, 'keys': list(zip(subjects, studies)),
            'counts': np.asarray([int(r['image_count']) for r in rows])}


def align(reference, candidate):
    lookup = {k: i for i, k in enumerate(candidate['keys'])}
    if set(lookup) != set(reference['keys']):
        raise ValueError('Subject/Study coverage mismatch')
    order = [lookup[k] for k in reference['keys']]
    if not np.array_equal(reference['y'], candidate['y'][order]) or not np.array_equal(reference['counts'], candidate['counts'][order]):
        raise ValueError('Labels/image counts mismatch')
    return candidate['p'][order]


def combine(tables, weights):
    if len(tables) != len(weights) or not np.isclose(sum(weights), 1) or min(weights) < 0:
        raise ValueError('Invalid fixed fusion weights')
    ref = tables[0]
    return {**ref, 'p': sum(w * align(ref, t) for t, w in zip(tables, weights))}


def concatenate_oof(tables):
    keys = [k for t in tables for k in t['keys']]
    subjects = [set(s for s, _ in t['keys']) for t in tables]
    if len(keys) != len(set(keys)) or any(a & b for i, a in enumerate(subjects) for b in subjects[i+1:]):
        raise ValueError('OOF patient overlap; never average seen-patient folds')
    return {'keys': keys, **{field: np.concatenate([t[field] for t in tables]) for field in ('p', 'y', 'counts')}}


def scores(table):
    p, y, keys = table['p'], table['y'], table['keys']
    details = []
    for j in range(10):
        yy, pp = y[:, j], p[:, j]
        pos = yy == 1
        tp = int((pos & (pp >= .5)).sum())
        denominator = int(pos.sum() + (pp >= .5).sum())
        details.append({'class_id': j+1, 'positive_studies': int(pos.sum()),
                        'negative_studies': int((~pos).sum()),
                        'positive_patients': len({s for (s, _), flag in zip(keys, pos) if flag}),
                        'negative_patients': len({s for (s, _), flag in zip(keys, pos) if not flag}),
                        'auc': binary_auc(yy, pp),
                        'ap': average_precision(yy, pp) if pos.any() else None,
                        'f1': 2*tp/denominator if denominator else 0.0})
    def mean(key):
        values = [d[key] for d in details]
        return float(np.mean(values)) if all(v is not None for v in values) else None
    return {'studies': len(keys), 'patients': len({s for s, _ in keys}), 'macro_auc': mean('auc'),
            'macro_ap': mean('ap'), 'macro_f1_at_0_5': mean('f1'), 'per_class': details}


def grouped(table):
    result = {'all': scores(table)}
    for name, mask in [('single_image', table['counts'] == 1), ('multi_image', table['counts'] > 1)]:
        result[name] = scores({'keys': [k for k, keep in zip(table['keys'], mask) if keep],
                              **{v: table[v][mask] for v in ('p', 'y', 'counts')}})
    return result


def bootstrap(tables, pairs, draws=2000, seed=42):
    names = list(tables)
    reference = tables[names[0]]
    arrays = {n: align(reference, t) for n, t in tables.items()}
    groups = defaultdict(list)
    for i, (subject, _) in enumerate(reference['keys']):
        groups[subject].append(i)
    patients = list(groups.values())
    rng = np.random.default_rng(seed)
    replicates = {n: [] for n in names}
    invalid = 0
    while len(replicates[names[0]]) < draws:
        if invalid > 20 * draws:
            raise ValueError('Too many single-class patient resamples')
        idx = np.concatenate([patients[i] for i in rng.integers(len(patients), size=len(patients))])
        y = reference['y'][idx]
        if any(np.unique(y[:, j]).size != 2 for j in range(10)):
            invalid += 1
            continue
        for name, p in arrays.items():
            replicates[name].append(np.mean([binary_auc(y[:, j], p[idx, j]) for j in range(10)]))
    ci = lambda values: np.quantile(values, [.025, .975]).tolist()
    return {'valid_draws': draws, 'invalid_draws': invalid, 'seed': seed,
            'unit': 'patient; all Studies retained, with replacement',
            'auc_ci95': {n: ci(v) for n, v in replicates.items()},
            'differences': {f'{a}_minus_{b}': {'point': scores(tables[a])['macro_auc'] - scores(tables[b])['macro_auc'],
                                             'ci95': ci(np.asarray(replicates[a])-replicates[b])} for a, b in pairs}}


def finalize_task(run, task, *, expected_val=(859,228)):
    run = Path(run)
    status = json.loads((run/'status.json').read_text())
    if status['state'] != 'complete':
        raise ValueError('Cannot freeze incomplete training')
    result = {'task': task, 'status': status, 'predictions': {}}
    kinds = ['raw', 'ema'] if task['ema'] else ['raw']
    for kind in kinds:
        folder = 'validation_analysis' if kind == 'raw' else 'ema_validation_analysis'
        table = read_table(run/folder/'study_predictions.csv')
        if (len(table['keys']),len({s for s, _ in table['keys']})) != expected_val:
            raise ValueError('Official Val coverage changed')
        result['predictions'][kind] = {'official_val': grouped(table)}
        if task['fold'] is not None:
            folder = 'oof_analysis' if kind == 'raw' else 'ema_oof_analysis'
            result['predictions'][kind]['oof'] = grouped(read_table(run/folder/'study_predictions.csv'))
    rows = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x]
    result['resources'] = {'epoch_hours': sum(x['elapsed_seconds'] for x in rows)/3600,
                           'peak_allocated_mib': max(x['gpu_peak_allocated_mib'] for x in rows),
                           'peak_reserved_mib': max(x['gpu_peak_reserved_mib'] for x in rows),
                           'optimizer_steps': sum(x['optimizer_steps'] for x in rows),
                           'amp_skips': sum(x['amp_skipped_steps'] for x in rows)}
    result['pretrained'] = json.loads((run/'pretrained_source.json').read_text())
    write_json(run/'batch_metrics.json', result)
    from cxr.batch.artifacts import export_task, utcnow
    export_task(run, task, result)
    # A previous failed attempt stays in failure history, never masquerades as current failure.
    if (run/'FAILED.json').exists():
        (run/'FAILED.json').rename(run/'recovered_failure.json')
    important = ('best_model.pth','resolved_config.yaml','metrics.json','val_predictions.csv','run_manifest.json','checkpoint_resume.pth')
    if task['fold'] is not None:
        important += ('oof_predictions.csv',)
    checks = {name:digest(run/name) for name in important}
    write_json(run/'DONE.json',{'state':'complete','model_id':task['id'],'verified_sha256':checks,'time_utc':utcnow()})
    files = {p.relative_to(run).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'batch_frozen_manifest.json'}
    write_json(run/'batch_frozen_manifest.json', {'task_id': task['id'], 'files': files})
    for p in run.rglob('*'):
        if p.is_file():
            p.chmod(0o444)


def verify_frozen(run):
    run = Path(run)
    manifest = json.loads((run/'batch_frozen_manifest.json').read_text())
    for name, meta in manifest['files'].items():
        p = run/name
        if p.stat().st_size != meta['bytes'] or digest(p) != meta['sha256']:
            raise ValueError(f'Frozen result changed: {p}')


def summarize(plan, batch_root, output):
    batch_root, output = Path(batch_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    tasks = [t for t in plan['tasks'] if t['group'] == 'A']
    tables, oofs, paths, metrics = {}, {}, {}, {}
    for t in tasks:
        run = batch_root/'experiments'/t['id']
        verify_frozen(run)
        tables[t['id']] = read_table(run/'validation_analysis/study_predictions.csv')
        oofs[t['id']] = read_table(run/'oof_analysis/study_predictions.csv')
        paths[t['id']] = [run/'best_model.pt']
        metrics[t['id']] = json.loads((run/'batch_metrics.json').read_text())
    for branch in ('D', 'C', 'S'):
        names = [f'{branch}17_F{i:02d}' for i in range(1, 6)]
        tables[branch+'5'] = combine([tables[n] for n in names], [.2]*5)
        oofs[branch+'5'] = concatenate_oof([oofs[n] for n in names])
        if len(oofs[branch+'5']['keys']) != 18513 or len({s for s, _ in oofs[branch+'5']['keys']}) != 11055:
            raise ValueError('Full OOF coverage incomplete')
        paths[branch+'5'] = [p for n in names for p in paths[n]]
    tables['E17'] = combine([tables[n] for n in ('D5','C5','S5')], [.25,.25,.5])
    oofs['E17'] = combine([oofs[n] for n in ('D5','C5','S5')], [.25,.25,.5])
    paths['E17'] = paths['D5']+paths['C5']+paths['S5']
    e07 = ROOT/plan['e07_reference']['path']
    if digest(e07) != plan['e07_reference']['sha256']:
        raise ValueError('E07 reference changed')
    tables['E07'] = read_table(e07)
    comparisons = [(n, 'E07') for n in ('D5','C5','S5','E17')]
    # One predeclared EMA alternative, no winner-based search.
    ema_names = [f'C17_F{i:02d}' for i in range(1, 6)]
    tables['C5_EMA'] = combine([read_table(batch_root/'experiments'/n/'ema_validation_analysis/study_predictions.csv') for n in ema_names], [.2]*5)
    tables['E17_EMA_C'] = combine([tables['D5'],tables['C5_EMA'],tables['S5']], [.25,.25,.5])
    paths['C5_EMA'] = [batch_root/'experiments'/n/'best_ema_model.pt' for n in ema_names]
    paths['E17_EMA_C'] = paths['D5']+paths['C5_EMA']+paths['S5']
    comparisons += [('C5_EMA','C5'),('E17_EMA_C','E17')]
    candidates = {n: tables[n] for n in ('E07','D5','C5','S5','E17','C5_EMA','E17_EMA_C')}
    result = {'tasks': metrics, 'official_val': {n: grouped(t) for n,t in candidates.items()},
              'bootstrap_val': bootstrap(tables, comparisons, plan['bootstrap_draws']),
              'oof': {n: grouped(oofs[n]) for n in ('D5','C5','S5','E17')},
              'bootstrap_oof': bootstrap({n:oofs[n] for n in ('D5','C5','S5','E17')},
                                         [('E17',n) for n in ('D5','C5','S5')], plan['bootstrap_draws']),
              'deployment': {n: deployment_bytes(p) for n,p in paths.items()},
              'e07_deployment_bytes': plan['e07_reference']['checkpoint_bytes'],
              'limitations': ['Official Val reused for epoch and candidate selection; exploratory CIs, not independent confirmation.',
                              'OOF is one unseen-fold model per backbone per patient, not a five-model ensemble per patient.',
                              'No Phase15 Holdout reused. Single/multi image class composition differs.',
                              'Multiple related candidates are not independent significance tests.']}
    result['fold_oof_bootstrap'] = {str(i):bootstrap({n:oofs[n] for n in (f'D17_F{i:02d}',f'C17_F{i:02d}',f'S17_F{i:02d}')},[],plan['bootstrap_draws']) for i in range(1,6)}
    result['error_correlations'] = {}
    ref = candidates['E07']
    for a,b in comparisons:
        ea, eb = align(ref,candidates[a])-ref['y'], align(ref,candidates[b])-ref['y']
        result['error_correlations'][a+'__'+b] = [float(np.corrcoef(ea[:,j],eb[:,j])[0,1])
            if ea[:,j].std() and eb[:,j].std() else None for j in range(10)]
    output.mkdir(parents=True)
    write_json(output/'summary.json', result)
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate','macro_auc','macro_ap','macro_f1','bytes','fits_500MB'])
        for n,t in candidates.items():
            s=scores(t); size=result['deployment'].get(n, {'bytes':plan['e07_reference']['checkpoint_bytes'],'fits':True})
            writer.writerow([n,s['macro_auc'],s['macro_ap'],s['macro_f1_at_0_5'],size['bytes'],size['fits']])
    lines=['# Phase17 批量实验报告','', '| 方案 | Val Macro-AUC | Macro-AP | F1@0.5 |','|---|---:|---:|---:|']
    for n,t in candidates.items():
        s=scores(t); lines.append(f'| {n} | {s["macro_auc"]:.6f} | {s["macro_ap"]:.6f} | {s["macro_f1_at_0_5"]:.6f} |')
    lines += ['', '完整逐类、分组支持数、患者 Bootstrap 差值和部署大小见 summary.json。', *result['limitations']]
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
