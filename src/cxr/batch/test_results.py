"""Label-free Test probabilities and strictly validated submission candidates."""
import csv
import math
from pathlib import Path

import numpy as np

from cxr.batch.artifacts import PROB_COLUMNS, write_probabilities
from cxr.batch.plan import digest, write_json

SUBMISSION_COLUMNS = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']


def read_test_probabilities(path, expected_keys=None, expected_studies=1414):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PROB_COLUMNS:
            raise ValueError('Test probability columns/order differ from Prob_1..Prob_10')
        rows = list(reader)
    if any(set(row) != set(PROB_COLUMNS) for row in rows):
        raise ValueError('Unexpected Test probability field')
    keys = [(r['Subject_id'], r['Study_id']) for r in rows]
    if len(keys) != expected_studies or len(set(keys)) != len(keys) or len({s for _, s in keys}) != len(keys):
        raise ValueError('Test Study count, ID or uniqueness mismatch')
    if any(not subject or not study for subject, study in keys):
        raise ValueError('Blank Test ID')
    if expected_keys is not None and set(keys) != set(expected_keys):
        raise ValueError('Test Study ID coverage mismatch')
    probs = np.asarray([[float(r[f'Prob_{i}']) for i in range(1, 11)] for r in rows], dtype=np.float64)
    if probs.shape != (len(keys), 10) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Invalid Test probabilities')
    return {'keys': keys, 'p': probs}


def average_test(tables, weights, expected_studies=1414):
    if not tables or len(tables) != len(weights) or min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError('Invalid fixed Test fusion')
    keys = tables[0]['keys']
    expected = set(keys)
    if len(keys) != expected_studies:
        raise ValueError('Incomplete first Test prediction')
    total = np.zeros((len(keys), 10), dtype=np.float64)
    for table, weight in zip(tables, weights):
        lookup = {key: idx for idx, key in enumerate(table['keys'])}
        if len(lookup) != len(table['keys']) or set(lookup) != expected:
            raise ValueError('Test fusion Study IDs differ')
        total += weight * table['p'][[lookup[key] for key in keys]]
    return {'keys': keys, 'p': total}


def write_test(path, table, provenance):
    if Path(path).exists() or Path(path).with_suffix('.provenance.json').exists():
        raise FileExistsError(path)
    write_probabilities(path, table, provenance)
    read_test_probabilities(path, table['keys'], len(table['keys']))


def validate_submission(path, expected_keys=None, expected_studies=1414):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise ValueError('Submission must contain the exact four columns in order')
        rows = list(reader)
    if any(set(row) != set(SUBMISSION_COLUMNS) for row in rows):
        raise ValueError('Unexpected submission field')
    if len(rows) != expected_studies * 10:
        raise ValueError(f'Expected {expected_studies * 10} data rows, got {len(rows)}')
    pairs = set()
    studies = {}
    for row in rows:
        subject, study = row['Subject_id'], row['Study_id']
        if not subject or not study or row['Predict_class'] not in [str(i) for i in range(1, 11)]:
            raise ValueError('Blank ID or invalid original class ID')
        value = float(row['Probability'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Non-finite or out-of-range probability')
        pair = (study, row['Predict_class'])
        if pair in pairs:
            raise ValueError('Duplicate Study/class prediction')
        pairs.add(pair)
        previous = studies.setdefault(study, subject)
        if previous != subject:
            raise ValueError('Study assigned to multiple subjects')
    keys = {(subject, study) for study, subject in studies.items()}
    if len(studies) != expected_studies or (expected_keys is not None and keys != set(expected_keys)):
        raise ValueError('Submission Test Study coverage mismatch')
    return {'status': 'FORMAT_PASS', 'study_count': len(studies), 'rows': len(rows),
            'classes_per_study': 10, 'sha256': digest(path)}


def write_submission(path, table, manifest, expected_studies=1414):
    path = Path(path)
    if path.exists() or path.with_suffix('.manifest.json').exists():
        raise FileExistsError(path)
    if len(table['keys']) != expected_studies:
        raise ValueError('Incomplete Test probability table')
    probabilities=np.asarray(table['p'])
    if probabilities.shape!=(expected_studies,10) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Invalid Test probability matrix')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(SUBMISSION_COLUMNS)
        for (subject, study), vector in zip(table['keys'], table['p']):
            for class_id, value in enumerate(vector, 1):
                writer.writerow([subject, study, class_id, format(float(value), '.17g')])
    result = validate_submission(path, table['keys'], expected_studies)
    write_json(path.with_suffix('.manifest.json'), {**manifest, 'format': result,
               'test_auc': None, 'hidden_test_labels_used': False})
    return result
