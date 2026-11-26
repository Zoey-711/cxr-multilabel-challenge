"""Check all 26 uploaded training histories against their run manifests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    records = []
    for run in sorted((args.root/'experiments').iterdir()):
        if not (run/'DONE.json').is_file():
            continue
        if run.name.endswith('_GH_PREFLIGHT'):
            continue
        manifest = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        needed = ['train_history.csv', 'train.log', 'resolved_config.yaml', 'metrics.json', 'val_predictions.csv', 'test_predictions.csv']
        failures = []
        for name in needed:
            path = run/name
            expected = manifest.get('files', {}).get(name, {}).get('sha256')
            if not path.is_file() or not expected or digest(path) != expected:
                failures.append(name)
        with (run/'train_history.csv').open(encoding='utf-8-sig', newline='') as f:
            history = list(csv.DictReader(f))
        metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        best = int(metrics['best_epoch'])
        if not history or best < 1 or best > len(history):
            failures.append('epoch_history_range')
        records.append({'model_id': run.name, 'history_epochs': len(history), 'best_epoch': best,
                        'status': 'PASS' if not failures else 'FAIL', 'failures': ';'.join(failures)})
    if len(records) != 26:
        raise ValueError(f'Expected 26 formal training records, got {len(records)}')
    with args.out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if any(r['status'] != 'PASS' for r in records):
        raise ValueError('A training record failed SHA or epoch coverage audit')
    print(f'TRAINING_RECORD_PASS {len(records)}')


if __name__ == '__main__':
    main()
