"""Read-only prerequisites for the proposed G/H full-Train extension.

This module does not select a backbone, create teacher targets, or start training.
"""
import csv
from pathlib import Path

from cxr.batch.artifacts import label_keys, read_probabilities
from cxr.batch.plan import digest


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
