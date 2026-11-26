"""Strict Study-level consistency gates; no tolerance tuning or probability edits."""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBMISSION_COLUMNS = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
INDEX_COLUMNS = ['image', 'Subject_id', 'Study_id', 'study_image_count']
OFFICIAL_TOLERANCE = 1e-3
GATE_THRESHOLD = 0.5


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Expected ten finite probabilities in [0,1]')
    return values


def read_submission(path, expected_studies=1414):
    result = {}
    keys = set()
    study_subject = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise ValueError('Submission column names/order differ')
        for row in reader:
            subject, study, label = row['Subject_id'], row['Study_id'], row['Predict_class']
            if not subject or not study or label not in {str(i) for i in range(1, 11)}:
                raise ValueError('Invalid submission identifiers/class')
            if study in study_subject and study_subject[study] != subject:
                raise ValueError('Study belongs to multiple subjects')
            study_subject[study] = subject
            key = (subject, study)
            if (key, label) in keys:
                raise ValueError('Duplicate submission key')
            keys.add((key, label))
            result.setdefault(key, np.full(10, np.nan))[int(label)-1] = float(row['Probability'])
    if len(result) != expected_studies or len(keys) != expected_studies*10:
        raise ValueError('Submission does not have the expected complete Study/class coverage')
    return {key: probabilities(value) for key, value in result.items()}


def validate_index(rows):
    """Counts must describe the complete supplied metadata, never a timed subset."""
    images, studies, subjects = {}, defaultdict(list), {}
    for row in rows:
        name = row['image']
        subject, study = row['Subject_id'], row['Study_id']
        if not name or Path(name).name != name or '/' in name or '\\' in name or name in images:
            raise ValueError('Image names must be unique plain filenames')
        if not subject or not study or (study in subjects and subjects[study] != subject):
            raise ValueError('Invalid Study/subject mapping')
        subjects[study] = subject
        total = int(row['study_image_count'])
        if total < 1:
            raise ValueError('Invalid full Study image count')
        images[name] = dict(row, study_image_count=total)
        studies[(subject, study)].append(name)
    for key, names in studies.items():
        if any(images[name]['study_image_count'] != len(names) for name in names):
            raise ValueError(f'Incomplete full metadata or incorrect image counts: {key}')
    if not images:
        raise ValueError('Empty image metadata')
    return images, dict(studies)


def read_index(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != INDEX_COLUMNS:
            raise ValueError('Unexpected metadata columns')
        rows = list(reader)
    validate_index(rows)
    return rows


def aggregate_complete(rows, image_predictions, expected_keys=None):
    """Mean of every real image in each requested Study, in metadata order."""
    images, groups = validate_index(rows)
    requested = set(groups) if expected_keys is None else set(expected_keys)
    if not requested or not requested <= set(groups):
        raise ValueError('Unknown or empty expected Study set')
    required = {name for key in requested for name in groups[key]}
    if set(image_predictions) != required:
        raise ValueError('Missing/extra/partial Study image predictions')
    return {key: probabilities(np.mean([probabilities(image_predictions[n]) for n in groups[key]], axis=0))
            for key in sorted(requested)}


def compare_studies(actual, reference):
    if not actual or set(actual) != set(reference):
        raise ValueError('Study key sets differ; partial comparisons cannot pass')
    details, all_differences = [], []
    for key in sorted(reference):
        a, b = probabilities(actual[key]), probabilities(reference[key])
        diff = np.abs(a-b)
        gate = bool(np.any((a >= GATE_THRESHOLD) != (b >= GATE_THRESHOLD)))
        # Stable class-id tie-break; no epsilon, rounding, or probability repair.
        rank = not np.array_equal(np.argsort(-a, kind='stable'), np.argsort(-b, kind='stable'))
        error = bool(diff.max() > OFFICIAL_TOLERANCE)
        details.append(dict(Subject_id=key[0], Study_id=key[1], max_abs=float(diff.max()),
                            gate_changed=gate, ranking_changed=rank, probability_failed=error,
                            failed=gate or rank or error))
        all_differences.extend(diff.tolist())
    d = np.asarray(all_differences)
    report = dict(status='FAIL' if any(r['failed'] for r in details) else 'PASS', studies=len(details),
                  failed_studies=sum(r['failed'] for r in details),
                  gate_changed_studies=sum(r['gate_changed'] for r in details),
                  ranking_changed_studies=sum(r['ranking_changed'] for r in details),
                  probability_failed_studies=sum(r['probability_failed'] for r in details),
                  max_abs=float(d.max()), mean_abs=float(d.mean()), p99_abs=float(np.quantile(d, .99)),
                  tolerance=OFFICIAL_TOLERANCE, gate_threshold=GATE_THRESHOLD,
                  gate_basis='existing 0.5 audit convention; confirm official Gate definition if different',
                  ranking='all 10 classes descending; stable class-id tie-break', unit='Study')
    return report, details


def compare_rounds(rows, rounds, reference, timed_names):
    """Every round, every timed image's complete Study; never only the last round."""
    images, groups = validate_index(rows)
    if len(rounds) != 3 or len(timed_names) != len(set(timed_names)):
        raise ValueError('Exactly three rounds and unique timed images are required')
    if not set(timed_names) <= set(images):
        raise ValueError('Unknown timed image')
    keys = {(images[n]['Subject_id'], images[n]['Study_id']) for n in timed_names}
    if not keys <= set(reference):
        raise ValueError('Referenced Studies missing from submission')
    reports = []
    first = None
    max_repeat_error = 0.
    for number, values in enumerate(rounds, 1):
        studies = aggregate_complete(rows, values, keys)
        report, details = compare_studies(studies, {k: reference[k] for k in keys})
        failed = {(d['Subject_id'], d['Study_id']) for d in details if d['failed']}
        report.update(round=number, timed_images=len(timed_names),
                      inconsistent_timed_images=sum((images[n]['Subject_id'], images[n]['Study_id']) in failed for n in timed_names))
        reports.append(report)
        if first is None:
            first = values
        else:
            for n in values:
                max_repeat_error = max(max_repeat_error, float(np.max(np.abs(probabilities(values[n])-probabilities(first[n])))))
    exact = max_repeat_error == 0.
    return dict(status='PASS' if exact and all(r['status']=='PASS' for r in reports) else 'FAIL',
                historical_submission_consistency='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',
                internal_repeatability='EXACT_MATCH' if exact else 'FAIL',
                max_repeat_probability_error=max_repeat_error, rounds=reports)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError('Need columns for empty output')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
