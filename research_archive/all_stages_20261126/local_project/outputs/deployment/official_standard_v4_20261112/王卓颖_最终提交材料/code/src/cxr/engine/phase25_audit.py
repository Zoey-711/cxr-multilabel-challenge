"""CPU-only fixed image-OOF pooling and source-label audit; no image inference."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from cxr.data.archive import archive_path, read_split_rows, parse_positive_classes
from cxr.engine.metrics import binary_auc
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def aucs(y, p):
    return np.array([binary_auc(y[:, i], p[:, i]) for i in range(10)])


def pool(p, views):
    mean = p.mean(0)
    groups = {'AP': 0, 'PA': 0, 'LL': 1, 'LATERAL': 1}
    balanced = mean
    if all(v in groups for v in views):
        g = np.array([groups[v] for v in views])
        balanced = np.stack([p[g == x].mean(0) for x in sorted(set(g))]).mean(0)
    return {'Mean': mean, 'Max': p.max(0), 'HalfMeanHalfMax': 0.5 * (mean + p.max(0)),
            'ViewBalanced': balanced}


def prepared(p, y):
    result = []
    for i in range(10):
        order = np.argsort(p[:, i], kind='stable')
        values = p[order, i]
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
        result.append((order, starts, y[order, i]))
    return result


def weighted_auc(prep, weights):
    result = []
    for order, starts, target in prep:
        w = weights[order]
        pos = np.add.reduceat(w * target, starts)
        neg = np.add.reduceat(w * (1 - target), starts)
        denom = pos.sum() * neg.sum()
        if denom == 0:
            return None
        result.append(float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / denom))
    return np.mean(result)


def bootstrap(y, predictions, subjects, settings):
    _, inverse = np.unique(subjects, return_inverse=True)
    n = int(inverse.max()) + 1
    prep = {r: prepared(p, y) for r, p in predictions.items()}
    rng = np.random.default_rng(settings['seed'])
    diffs = {r: [] for r in predictions if r != 'Mean'}
    invalid = 0
    for attempt in range(settings['max_attempts']):
        w = np.bincount(rng.integers(n, size=n), minlength=n)[inverse]
        values = {r: weighted_auc(v, w) for r, v in prep.items()}
        if any(v is None for v in values.values()):
            invalid += 1
            continue
        for r in diffs:
            diffs[r].append(values[r] - values['Mean'])
        if len(next(iter(diffs.values()))) == settings['valid_draws']:
            break
    if len(next(iter(diffs.values()))) != settings['valid_draws']:
        raise ValueError('Insufficient valid patient-bootstrap draws')
    return {'seed': settings['seed'], 'valid': settings['valid_draws'], 'invalid': invalid,
            'patients': n, 'attempts': attempt + 1,
            'strategy': 'paired patient multiplicities; discard entire draw if any class lacks positive/negative; fixed fitted models',
            'comparisons': {r: {'delta': float(aucs(y, predictions[r]).mean() - aucs(y, predictions['Mean']).mean()),
                                'ci95': np.quantile(d, [0.025, 0.975]).tolist()} for r, d in diffs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    out = root / cfg['output']
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'frozen_analysis_config.yaml').write_bytes(args.config.read_bytes())
    source_cfg = load_config(root / cfg['source_config'])
    archive = archive_path(source_cfg)
    train = read_split_rows(archive, 'train')
    val = read_split_rows(archive, 'val')
    fold_dir = root / 'outputs/analysis/phase17_patient_folds'
    fold_sha = {name: sha(fold_dir / name) for name in ('patient_folds.csv', 'study_folds.csv', 'image_folds.csv')}
    assert fold_sha == source_cfg['data']['train_subset']['sha256']
    fold_rows = rows(fold_dir / 'study_folds.csv')
    folds = {(r['Subject_id'], r['Study_id']): int(r['fold']) for r in fold_rows}
    patient_fold = {r['Subject_id']: int(r['fold']) for r in rows(fold_dir / 'patient_folds.csv')}
    meta = {(r['Subject_id'], r['Study_id'], r['Dicom_id']): r for r in train}
    assert len(meta) == len(train) == 30344
    studies = defaultdict(list)
    label_report = {}
    for split, data in [('train', train), ('val', val)]:
        labels = {}
        for r in data:
            assert r['Predict_class'].strip(), 'Blank target'
            positive = parse_positive_classes(r['Predict_class'])
            key = (r['Subject_id'], r['Study_id'])
            assert key not in labels or labels[key] == positive, 'Contradictory labels within Study'
            labels[key] = positive
            if split == 'train':
                studies[key].append((key[0], key[1], r['Dicom_id']))
        report = {'fields': list(data[0]), 'images': len(data), 'studies': len(labels),
                  'blank_or_invalid_labels': 0, 'within_study_label_conflicts': 0,
                  'no_finding_with_other_label_studies': sum(10 in x and len(x) > 1 for x in labels.values()),
                  'class_support': []}
        for i in range(1, 11):
            pos = [k for k, v in labels.items() if i in v]
            report['class_support'].append({'class_id': i, 'positive_studies': len(pos),
                                          'positive_patients': len({k[0] for k in pos})})
        report['cooccurrence_3_4_8'] = {f'{a}_{b}': sum(a in x and b in x for x in labels.values())
                                     for a, b in [(3, 4), (3, 8), (4, 8)]}
        label_report[split] = report
    keys = sorted(studies)
    assert set(keys) == set(folds) and len(keys) == 18513 and len(patient_fold) == 11055
    y = np.array([[int(i in parse_positive_classes(meta[studies[k][0]]['Predict_class']))
                   for i in range(1, 11)] for k in keys])
    subjects = np.array([k[0] for k in keys])
    study_fold = np.array([folds[k] for k in keys])
    counts = np.array([len(studies[k]) for k in keys])
    assert all(patient_fold[k[0]] == folds[k] for k in keys)
    label_rows = {(r['Subject_id'], r['Study_id']): r for r in rows(root / 'batch_assets/train_labels.csv')}
    assert all(all(int(label_rows[k][f'Target_{i}']) == y[j, i-1] for i in range(1,11))
               for j, k in enumerate(keys))
    rules = cfg['rules']
    predictions, audit = {}, []
    dates = {1:'20261019', 2:'20261020', 3:'20261021', 4:'20261022', 5:'20261023'}
    for branch in cfg['backbone_weights']:
        images = {}
        reference = {}
        for f in range(1, 6):
            run = root / f'outputs/experiments/{branch.lower()}_f{f}_{dates[f]}'
            manifest = json.loads((run / 'frozen_manifest.json').read_text(encoding='utf-8'))
            assert manifest['fold'] == f and manifest['fold_index_sha256'] == fold_sha
            run_cfg = load_config(run / 'config.yaml')
            assert run_cfg['data']['train_subset']['fold'] == f
            assert run_cfg['data']['label_encoding']['unlisted_policy'] == 'negative'
            for name in ['config.yaml', 'oof_analysis/image_predictions.csv', 'oof_analysis/study_predictions.csv']:
                assert sha(run / name) == manifest['files'][name]['sha256'], str(run / name)
            image_rows = rows(run / 'oof_analysis/image_predictions.csv')
            current = set()
            for r in image_rows:
                k = (r['Subject_id'], r['Study_id'], r['Image_id'])
                assert k in meta and k not in images and patient_fold[k[0]] == f and folds[k[:2]] == f
                p = np.array([float(r[f'probability_{i}']) for i in range(1,11)])
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                images[k] = p
                current.add(k)
            assert current == {k for k in meta if folds[k[:2]] == f}
            for r in rows(run / 'oof_analysis/study_predictions.csv'):
                k = (r['Subject_id'], r['Study_id'])
                reference[k] = np.array([float(r[f'probability_{i}']) for i in range(1,11)])
            audit.append({'model': branch, 'fold': f, 'images': len(current),
                          'manifest_path': str(run.relative_to(root) / 'frozen_manifest.json'),
                          'manifest_sha256': sha(run / 'frozen_manifest.json'),
                          'image_prediction_sha256': manifest['files']['oof_analysis/image_predictions.csv']['sha256']})
        assert set(images) == set(meta)
        values = [pool(np.stack([images[k] for k in studies[s]]),
                       [meta[k]['ViewPosition'] for k in studies[s]]) for s in keys]
        predictions[branch] = {r: np.stack([v[r] for v in values]) for r in rules}
        err = float(np.max(np.abs(predictions[branch]['Mean'] - np.stack([reference[k] for k in keys]))))
        assert err <= cfg['mean_reconstruction_atol'], err
        print(branch, 'OOF images verified', len(images), 'Mean max error', err, flush=True)
        label_report[branch + '_mean_reconstruction_max_error'] = err
    predictions['E17'] = {r: sum(w * predictions[b][r] for b, w in cfg['backbone_weights'].items()) for r in rules}
    output_rows = []
    for branch, values in predictions.items():
        for group, select in [('all', np.ones(len(keys), dtype=bool)), ('single', counts == 1), ('multi', counts > 1)]:
            for rule, p in values.items():
                scores = aucs(y[select], p[select])
                output_rows.append({'model': branch, 'rule': rule, 'group': group, 'studies': int(select.sum()),
                    'patients': len(set(subjects[select])), 'macro_auc': float(scores.mean()),
                    **{f'auc_{i+1}': float(v) for i, v in enumerate(scores)}})
    write_csv(out / 'aggregation_oof_comparison.csv', output_rows)
    fold_results = [{ 'fold': f, 'rule': r, 'macro_auc': float(aucs(y[study_fold==f], p[study_fold==f]).mean())}
                    for f in range(1,6) for r, p in predictions['E17'].items()]
    write_csv(out / 'aggregation_fold_comparison.csv', fold_results)
    for rule, p in predictions['E17'].items():
        write_csv(out / f'e17_oof_{rule}.csv', [{'Subject_id': k[0], 'Study_id': k[1],
            **{f'Prob_{i+1}': float(v) for i,v in enumerate(p[j])}} for j,k in enumerate(keys)])
    result = {'all': bootstrap(y, predictions['E17'], subjects, cfg['bootstrap'])}
    print('all-Study bootstrap complete', flush=True)
    select = counts > 1
    result['multi'] = bootstrap(y[select], {r:p[select] for r,p in predictions['E17'].items()}, subjects[select], cfg['bootstrap'])
    result['single'] = 'All rules identical by construction; delta and CI exactly zero.'
    result['folds'] = fold_results
    result['view_counts'] = dict(Counter(r['ViewPosition'] or '<blank>' for r in train))
    result['fallback_studies'] = sum(any(meta[k]['ViewPosition'] not in ('AP','PA','LL','LATERAL') for k in studies[s]) for s in keys)
    (out / 'bootstrap_comparisons.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'label_audit.json').write_text(json.dumps(label_report, indent=2), encoding='utf-8')
    (out / 'source_audit.json').write_text(json.dumps({'fold_sha256':fold_sha,'models':audit,
        'config_sha256':sha(args.config), 'source_code_sha256':sha(Path(__file__)),
        'train_csv_decoded_rows':30344,'OOF_studies':18513,'OOF_patients':11055}, indent=2), encoding='utf-8')
    print(json.dumps(result['all']), flush=True)


if __name__ == '__main__':
    main()
