"""Small, prespecified CPU-only E08 validation weight assessment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cxr.engine.metrics import binary_auc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    keys = [(r['Subject_id'], r['Study_id']) for r in rows]
    if len(keys) != 859 or len(set(keys)) != 859:
        raise ValueError(f'Invalid Val coverage: {path}')
    prefix = 'probability_' if 'probability_1' in fields else 'Prob_'
    p = np.array([[float(r[f'{prefix}{j}']) for j in range(1, 11)] for r in rows])
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError(f'Invalid probabilities: {path}')
    y = (np.array([[int(r[f'target_{j}']) for j in range(1, 11)] for r in rows])
         if 'target_1' in fields else None)
    return keys, p, y


def weighted_auc_draws(y, p, multiplicity):
    # Exact AUC for patient-resampled Studies, including average-rank ties.
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    pos = np.add.reduceat(multiplicity[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(multiplicity[:, order] * (1 - y[order]), starts, axis=1)
    numerator = (pos * (np.cumsum(neg, axis=1) - 0.5 * neg)).sum(axis=1)
    return numerator / (pos.sum(axis=1) * neg.sum(axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[2]
    out = root / cfg['output_dir']
    if (out / 'comparison.csv').exists():
        raise FileExistsError('Refusing to overwrite an existing assessment')
    out.mkdir(parents=True, exist_ok=True)
    original = root / 'outputs/submissions/E08_EMA_20261020'
    onnx_dir = root / 'outputs/submissions/E08_ONNX_FP32_20261028'
    manifest = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
    onnx_manifest = json.loads((onnx_dir / 'sha256_manifest.json').read_text(encoding='utf-8'))
    sources = {}

    def checked(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'Source SHA mismatch: {path}')
        sources[str(path.relative_to(root))] = actual
        return read(path)

    keys, frozen, _ = checked(original / 'val_predictions.csv',
                              manifest['files']['val_predictions.csv']['sha256'])

    def aligned(rows):
        other_keys, p, y = rows
        if set(other_keys) != set(keys):
            raise ValueError('Patient/Study mismatch')
        lookup = {k: i for i, k in enumerate(other_keys)}
        order = [lookup[k] for k in keys]
        return p[order], y[order] if y is not None else None

    onnx, _ = aligned(checked(onnx_dir / 'final_entry_val_predictions.csv',
                             onnx_manifest['final_entry_val_predictions.csv']['sha256']))
    components, labels = {}, None
    for name, entry in manifest['models'].items():
        sub = 'ema_validation_analysis' if name == 'N01_EMA' else 'validation_analysis'
        path = root / 'outputs/experiments' / entry['run_id'] / sub / 'study_predictions.csv'
        components[name], y = aligned(checked(path, entry['val_predictions_sha256']))
        if y is None or (labels is not None and not np.array_equal(y, labels)):
            raise ValueError('Target mismatch')
        labels = y
    e03 = 0.5 * (components['B01'] + components['B02'] + components['B03']) / 3 + 0.5 * components['M02']
    reproduction_error = float(np.max(np.abs(0.5 * e03 + 0.5 * components['N01_EMA'] - frozen)))
    if reproduction_error > 1e-12:
        raise ValueError('Frozen E08 formula failed reproduction')
    arrays = {'E08_ORIGINAL': frozen, 'E08_ONNX_FP32': onnx}
    weights = {'E08_ORIGINAL': 'original=1', 'E08_ONNX_FP32': 'onnx=1'}
    for w in cfg['onnx_mix_weights']:
        name = f'VERSION_MIX_ONNX_{w:.2f}'
        arrays[name] = (1-w)*frozen + w*onnx
        weights[name] = f'original={1-w:g};onnx={w:g}'
    for w in cfg['original_n01_weights']:
        name = f'ORIGINAL_N01_{w:.2f}'
        arrays[name] = (1-w)*e03 + w*components['N01_EMA']
        weights[name] = f'B01=B02=B03={(1-w)/6:.12g};M02={(1-w)/2:g};N01_EMA={w:g}'
    subjects = np.array([key[0] for key in keys])
    patients, inverse = np.unique(subjects, return_inverse=True)
    if len(patients) != 228 or not np.isin(labels, (0, 1)).all():
        raise ValueError('Unexpected patients or labels')
    rng = np.random.default_rng(cfg['seed'])
    patient_weights = np.array([np.bincount(rng.integers(len(patients), size=len(patients)),
                                          minlength=len(patients)) for _ in range(cfg['replicates'])])
    multiplicity = patient_weights[:, inverse]
    positives = multiplicity @ labels
    valid = ((positives > 0) & (positives < multiplicity.sum(axis=1)[:, None])).all(axis=1)
    multiplicity = multiplicity[valid].astype(float)
    point, draws, per_class = {}, {}, []
    for name, p in arrays.items():
        aucs = np.array([binary_auc(labels[:, j], p[:, j]) for j in range(10)])
        point[name] = float(aucs.mean())
        result = np.stack([weighted_auc_draws(labels[:, j], p[:, j], multiplicity)
                           for j in range(10)], axis=1)
        # Check against the existing rank-based implementation on expanded draws.
        for k in range(3):
            ids = np.repeat(np.arange(len(keys)), multiplicity[k].astype(int))
            expected = np.array([binary_auc(labels[ids, j], p[ids, j]) for j in range(10)])
            if not np.allclose(result[k], expected, atol=1e-12, rtol=0):
                raise ValueError('Bootstrap weighted AUC validation failed')
        draws[name] = result.mean(axis=1)
        for j, auc in enumerate(aucs):
            per_class.append({'candidate': name, 'class_id': j+1,
                              'positive_studies': int(labels[:, j].sum()), 'auc': float(auc)})
        print(name, point[name], flush=True)
    if abs(point['E08_ORIGINAL']-0.8116686461316818)>1e-12 or abs(point['E08_ONNX_FP32']-0.8117331556496273)>1e-12:
        raise ValueError('Recorded AUC did not reproduce')
    rows = []
    for name in arrays:
        vs_orig = np.quantile(draws[name] - draws['E08_ORIGINAL'], [0.025, 0.975])
        vs_onnx = np.quantile(draws[name] - draws['E08_ONNX_FP32'], [0.025, 0.975])
        rows.append({'candidate': name, 'weights': weights[name], 'val_macro_auc': point[name],
                     'delta_vs_original': point[name]-point['E08_ORIGINAL'],
                     'ci95_vs_original_low': vs_orig[0], 'ci95_vs_original_high': vs_orig[1],
                     'delta_vs_onnx': point[name]-point['E08_ONNX_FP32'],
                     'ci95_vs_onnx_low': vs_onnx[0], 'ci95_vs_onnx_high': vs_onnx[1]})
    correlations = [{'class_id': j+1, 'pearson_original_onnx': float(np.corrcoef(frozen[:, j], onnx[:, j])[0, 1]),
                     'pearson_e03_n01': float(np.corrcoef(e03[:, j], components['N01_EMA'][:, j])[0, 1])}
                    for j in range(10)]
    write_csv(out / 'comparison.csv', rows)
    write_csv(out / 'per_class_auc.csv', per_class)
    write_csv(out / 'correlations.csv', correlations)
    audit = {'studies': len(keys), 'patients': len(patients), 'source_sha256': sources,
             'config_sha256': sha(args.config), 'analysis_code_sha256': sha(Path(__file__)),
             'original_formula_max_abs_error': reproduction_error,
             'original_onnx_max_abs_diff': float(np.max(np.abs(onnx-frozen))),
             'bootstrap': {'seed': cfg['seed'], 'requested': cfg['replicates'], 'valid': int(valid.sum()),
                           'invalid': int((~valid).sum()), 'paired': True,
                           'invalid_rule': 'Exclude a draw if any class lacks either label',
                           'method': 'Patient-cluster percentile, same resamples for all candidates; weighted AUC checked against explicit replicated Study rows'},
             'scope': 'EXPLORATORY repeated official Val; no independent OOF or hidden Test evidence; no test predictions read',
             'deployment': 'Two-engine version blending requires ten forwards and naively 555230904 bytes of current packages; not verified under 500MB/100ms. Original internal reweighting keeps five forwards; T4_UNVERIFIED.',
             'onnx_internal_reweighting': 'UNAVAILABLE: individual ONNX Val predictions not present in selected frozen artifacts'}
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# E08 fixed-weight assessment', '', audit['scope'], '',
             '| Candidate | Val AUC | Delta vs original | 95% paired patient CI |',
             '|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['candidate']} | {row['val_macro_auc']:.9f} | {row['delta_vs_original']:+.9f} | [{row['ci95_vs_original_low']:+.9f}, {row['ci95_vs_original_high']:+.9f}] |")
    lines += ['', 'Weights are in comparison.csv. This is a fixed small sensitivity check, not an optimization of all five weights.',
              'Intervals are unadjusted for multiple comparisons and repeated validation selection. No leaderboard gain can be inferred.',
              audit['deployment'], '', 'No Submission generated. No training or inference performed.']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
