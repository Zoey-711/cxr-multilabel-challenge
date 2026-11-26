"""Verify the frozen G/H teacher CSV against unseen-fold branch OOF files."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('phase27', Path(__file__).with_name('Analyze-Phase27.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = a.root
    keys, _ = module.labels(root/'manifests/train_labels.csv', 18513)
    provenance = json.loads((root/'analysis/gh_teacher_oof.provenance.json').read_text(encoding='utf-8'))
    actual = module.probs(root/'analysis/gh_teacher_oof.csv', keys)
    expected = np.zeros_like(actual)
    for name, weight in provenance['formula'].items():
        path = root/'oof'/f'{name}_oof_all.csv'
        if digest(path) != provenance['source'][name]['probabilities_sha256']:
            raise ValueError(f'Teacher source hash differs: {name}')
        expected += float(weight) * module.probs(path, keys)
    difference = float(np.max(np.abs(actual - expected)))
    if digest(root/'analysis/gh_teacher_oof.csv') != provenance['csv_sha256'] or difference > 1e-12:
        raise ValueError(f'Teacher formula/hash differs; max abs diff {difference}')
    result = {'studies': len(keys), 'formula': provenance['formula'], 'max_absolute_probability_difference': difference,
              'teacher_csv_sha256': provenance['csv_sha256'], 'source_fold_isolation_audited_by': 'Analyze-Phase27.py',
              'status': 'PASS'}
    a.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
