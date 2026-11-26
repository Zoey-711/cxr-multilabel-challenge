"""Read-only CPU audit of a versioned friend Phase17 result tree."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = [f'Prob_{i}' for i in range(1, 11)]
T = [f'Target_{i}' for i in range(1, 11)]
TASKS = [f'{b}_F{i:02d}' for b in ('D17', 'C17', 'S17', 'N17') for i in range(1, 6)] + ['CS17_F01', 'NM17_F01', 'G17_FULL_01', 'G17_FULL_02', 'H17_DISTILL_01', 'H17_DISTILL_02']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def labels(path, count):
    rr = rows(path)
    d = {(r['Subject_id'], r['Study_id']): np.asarray([int(r[c]) for c in T], dtype=np.int8) for r in rr}
    if len(d) != count or len(rr) != count:
        raise ValueError(f'Label coverage differs: {path}')
    keys = sorted(d)
    return keys, np.stack([d[k] for k in keys])


def probs(path, keys):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['Subject_id', 'Study_id', *P]:
            raise ValueError(f'Probability columns differ: {path}')
        rr = list(reader)
    d = {(r['Subject_id'], r['Study_id']): [float(r[c]) for c in P] for r in rr}
    if len(rr) != len(keys) or len(d) != len(keys) or set(d) != set(keys):
        raise ValueError(f'Study ID coverage differs: {path}')
    a = np.asarray([d[k] for k in keys])
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError(f'Invalid probabilities: {path}')
    return a


def auc(y, p):
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return None
    order = np.argsort(p, kind='mergesort')
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def score(y, p):
    per = [auc(y[:, c], p[:, c]) for c in range(10)]
    return (float(np.mean(per)) if all(x is not None for x in per) else None), per


def bootstrap(y, left, right, keys, draws, seed):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[0]].append(i)
    groups = [np.asarray(g) for g in groups.values()]
    rng = np.random.default_rng(seed)
    differences = []
    invalid = 0
    while len(differences) < draws and invalid < draws * 10:
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        a, _ = score(y[ix], left[ix])
        b, _ = score(y[ix], right[ix])
        if a is None or b is None:
            invalid += 1
        else:
            differences.append(a - b)
    if len(differences) != draws:
        raise ValueError('Too many bootstrap draws lacked class support')
    return {'seed': seed, 'patients': len(groups), 'valid': draws, 'invalid': invalid,
            'ci95': [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            'rule': 'Sample patients with all their Studies; discard if any class lacks positive or negative Studies'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, type=Path)
    ap.add_argument('--project', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--commit', required=True)
    a = ap.parse_args()
    root = a.version / 'results/phase17_batch'
    if not root.is_dir() or a.output.exists():
        raise ValueError('Source absent or analysis output already exists')
    a.output.mkdir(parents=True)
    man = root / 'manifests'
    train_keys, yt = labels(man / 'train_labels.csv', 18513)
    val_keys, yv = labels(man / 'val_labels.csv', 859)
    fold = {r['Subject_id']: int(r['fold']) for r in rows(man/'patient_folds.csv')}
    if len(fold) != 11055 or set(fold) != {k[0] for k in train_keys}:
        raise ValueError('Patient folds differ from Train labels')
    frozen = json.loads((a.project/'outputs/analysis/phase17_final_handoff_20261023/freeze_manifest.json').read_text(encoding='utf-8'))
    fold_hashes = {n: sha(man/n) for n in ('patient_folds.csv', 'study_folds.csv', 'image_folds.csv')}
    if fold_hashes != frozen['fold_hashes']:
        raise ValueError('Friend split differs from local frozen fivefold split')
    states = {r['Model_ID']: r['State'].upper() for r in rows(root/'reports/experiment_status.csv')}
    states.update({r['Model_ID']: r['State'].upper() for r in rows(root/'reports/extension_status.csv')})
    checks = []
    for model in TASKS:
        run = root/'experiments'/model
        done = run/'DONE.json'
        manifest = run/'run_manifest.json'
        item = {'model_id': model, 'reported_state': states.get(model, 'PENDING'), 'weight_state': 'WEIGHTS_NOT_DOWNLOADED'}
        if item['reported_state'] == 'COMPLETE' and done.is_file() and manifest.is_file():
            m = json.loads(manifest.read_text(encoding='utf-8'))
            if m.get('model_id') != model or m.get('fold_hashes') != fold_hashes:
                raise ValueError(f'Model/fold provenance differs: {model}')
            hashes = json.loads(done.read_text(encoding='utf-8')).get('verified_sha256', {})
            needed = ['resolved_config.yaml', 'metrics.json', 'val_predictions.csv', 'test_predictions.csv']
            if not model.startswith(('G17_', 'H17_')):
                needed.append('oof_predictions.csv')
            missing = [n for n in needed if not (run/n).is_file() or hashes.get(n) != sha(run/n)]
            item['audit_state'] = 'PASS' if not missing else 'INCOMPLETE'
            item['missing_or_hash_mismatch'] = missing
            if (run/'best_model.pth').exists():
                item['weight_state'] = 'DOWNLOADED'
            if 'test_predictions.csv' not in missing:
                test = rows(run/'test_predictions.csv')
                ids = {(r['Subject_id'],r['Study_id']) for r in test}
                item['test_studies'] = len(test)
                if len(test) != 1414 or len(ids) != 1414:
                    item['audit_state'] = 'INCOMPLETE'
        else:
            item['audit_state'] = 'INCOMPLETE' if item['reported_state'] == 'COMPLETE' else item['reported_state']
        checks.append(item)
    result = {'source': {'url': 'git@github.com:Zoey-711/cxr-multilabel-challenge.git',
                         'branch': 'results/phase17-20261024', 'commit': a.commit,
                         'subdirectory': 'results/phase17_batch'},
              'counts': dict(Counter(x['audit_state'] for x in checks)), 'fold_hashes': fold_hashes,
              'artifacts': checks, 'metrics': {}, 'bootstrap': {}, 'test_auc': None}
    # Each OOF row must come from the model for that patient's unseen fold.
    for name in ('D17', 'C17', 'S17', 'N17', 'C17_EMA', 'N17_EMA'):
        path = root/'oof'/f'{name}_oof_all.csv'
        provenance = json.loads((root/'oof'/f'{name}_oof_all.provenance.json').read_text(encoding='utf-8'))
        if sha(path) != provenance['csv_sha256']:
            raise ValueError(f'OOF hash differs: {name}')
        rr = rows(root/'oof'/f'{name}_source_index.csv')
        sources = {(r['Subject_id'],r['Study_id']): (r['model_id'],int(r['fold'])) for r in rr}
        if len(rr) != 18513 or len(sources) != 18513 or set(sources) != set(train_keys):
            raise ValueError(f'OOF source coverage differs: {name}')
        base = name.replace('_EMA', '')
        for k, (model, f) in sources.items():
            expected = f'{base}_F{f:02d}' + ('_EMA' if name.endswith('_EMA') else '')
            if f != fold[k[0]] or model != expected:
                raise ValueError(f'OOF patient leakage/source differs: {name} {k}')
        p = probs(path, train_keys)
        value, per = score(yt, p)
        result['metrics'][name+'_OOF'] = {'macro_auc': value, 'per_class_auc': per, 'studies': 18513, 'sha256': sha(path)}
        vpath = root/'validation'/f'{name}_val_ensemble.csv'
        vp = probs(vpath, val_keys)
        value, per = score(yv, vp)
        result['metrics'][name+'_VAL'] = {'macro_auc': value, 'per_class_auc': per, 'studies': 859, 'sha256': sha(vpath)}
    for name in ('E17', 'G17_PAIR', 'H17_PAIR'):
        path = root/'validation'/f'{name}_val_probabilities.csv'
        value, per = score(yv, probs(path, val_keys))
        result['metrics'][name+'_VAL'] = {'macro_auc': value, 'per_class_auc': per, 'sha256': sha(path)}
    for name in ('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02'):
        path = root/'experiments'/name/'val_predictions.csv'
        value, per = score(yv, probs(path,val_keys))
        result['metrics'][name+'_VAL'] = {'macro_auc': value, 'per_class_auc': per, 'sha256': sha(path), 'oof': 'N/A'}
    e08path = a.project/'outputs/submissions/E08_EMA_20261020/val_predictions.csv'
    e08manifest = json.loads((e08path.parent/'manifest.json').read_text(encoding='utf-8'))
    if sha(e08path) != e08manifest['files']['val_predictions.csv']['sha256']:
        raise ValueError('Frozen E08 hash differs')
    e08rows = rows(e08path)
    e08map = {(r['Subject_id'],r['Study_id']): [float(r[f'probability_{i}']) for i in range(1,11)] for r in e08rows}
    if len(e08rows) != 859 or set(e08map) != set(val_keys):
        raise ValueError('E08 Val IDs differ')
    e08 = np.asarray([e08map[k] for k in val_keys])
    e08auc, _ = score(yv,e08)
    if abs(e08auc-0.8116686461316818) > 1e-9:
        raise ValueError('Frozen E08 Val AUC differs')
    result['metrics']['E08_VAL'] = {'macro_auc': e08auc, 'sha256': sha(e08path)}
    local_path = a.project/'outputs/analysis/phase17_final_handoff_20261023/e17_oof_study_probabilities.csv'
    local_oof = probs(local_path, train_keys)
    local_auc, _ = score(yt, local_oof)
    if abs(local_auc-0.8030492475894413) > 1e-9:
        raise ValueError('Local E17 OOF AUC differs')
    result['metrics']['LOCAL_E17_OOF'] = {'macro_auc': local_auc, 'sha256': sha(local_path)}
    n = probs(root/'oof/N17_oof_all.csv', train_keys)
    ne = probs(root/'oof/N17_EMA_oof_all.csv', train_keys)
    result['bootstrap']['N17_EMA_minus_N17_OOF'] = bootstrap(yt, ne, n, train_keys, 1000, 42)
    result['bootstrap']['N17_EMA_minus_LOCAL_E17_OOF'] = bootstrap(yt, ne, local_oof, train_keys, 1000, 43)
    for name, path in [('N17_EMA',root/'validation/N17_EMA_val_ensemble.csv'),
                       ('E17',root/'validation/E17_val_probabilities.csv'),
                       ('H17_DISTILL_02',root/'experiments/H17_DISTILL_02/val_predictions.csv')]:
        result['bootstrap'][name+'_minus_E08_VAL'] = bootstrap(yv, probs(path,val_keys), e08, val_keys, 2000, 44)
    # Test submissions are checked for format only. Hidden labels do not exist here.
    result['submissions'] = []
    for r in rows(root/'submissions/submission_candidates.csv'):
        if r['Candidate_ID'] not in ('H17_DISTILL_02','N17_F05_EMA','E17'):
            continue
        path = root/r['File_Path']
        if path.is_file():
            content = rows(path)
            ids = {(x['Subject_id'],x['Study_id'],x['Predict_class']) for x in content}
            valid = len(content)==14140 and len(ids)==14140 and {x['Predict_class'] for x in content}==set(map(str,range(1,11)))
            valid = valid and all(np.isfinite(float(x['Probability'])) and 0<=float(x['Probability'])<=1 for x in content)
            result['submissions'].append({'candidate':r['Candidate_ID'],'file':str(path),'sha256':sha(path),
                                          'rows':len(content),'format_pass':bool(valid),
                                          'deployment_status':r['Deployment_Status']})
    result['source']['selected_file_hashes'] = {n:sha(root/'reports'/n) for n in ('experiment_status.csv','extension_status.csv','experiment_summary.csv')}
    result['source']['selected_file_hashes']['train_labels.csv'] = sha(man/'train_labels.csv')
    result['source']['selected_file_hashes']['val_labels.csv'] = sha(man/'val_labels.csv')
    (a.output/'phase27_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (a.output/'artifact_status.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['model_id','reported_state','audit_state','weight_state','missing_or_hash_mismatch','test_studies']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for item in checks:
            writer.writerow({k:json.dumps(item[k]) if isinstance(item.get(k),list) else item.get(k,'') for k in fields})
    print(json.dumps({'counts':result['counts'],
                      'key_metrics':{k:v['macro_auc'] for k,v in result['metrics'].items() if k in
                        ('N17_OOF','N17_EMA_OOF','N17_VAL','N17_EMA_VAL','E17_VAL','E08_VAL',
                         'G17_FULL_01_VAL','G17_FULL_02_VAL','H17_DISTILL_01_VAL','H17_DISTILL_02_VAL')},
                      'output':str(a.output/'phase27_audit.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
