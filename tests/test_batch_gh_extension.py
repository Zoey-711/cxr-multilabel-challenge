"""CPU-only fail-closed checks; no GPU jobs are started."""
import csv

import numpy as np
import pytest

from cxr.batch.artifacts import write_probabilities
from cxr.batch.extension import verify_teacher_sources
from cxr.batch.plan import load_plan, selected_tasks


def _csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reviewed_extension_is_declared_but_not_executable():
    plan = load_plan('configs/batch_plan.yaml')
    assert len(selected_tasks(plan)) == 22
    slots = plan['conditional_extension']['reserved_slots']
    assert [s['group'] for s in slots] == ['G', 'G', 'H', 'H']
    assert all(s['implementation_status'] == 'CPU_VERIFIED_GPU_PENDING' and not s['enabled'] for s in slots)
    assert all(s['fold'] is None for s in slots)
    with pytest.raises(ValueError, match='run-extension'):
        selected_tasks(plan, extended=True)


def test_teacher_sources_are_patient_isolated_and_complete(tmp_path):
    root = tmp_path
    keys = [(f'p{i}', f's{i}') for i in range(1, 6)]
    _csv(root/'manifests/patient_folds.csv', ['Subject_id', 'fold'],
         [{'Subject_id': subject, 'fold': i} for i, (subject, _) in enumerate(keys, 1)])
    labels = [{'Subject_id': subject, 'Study_id': study, 'fold': i, 'image_count': 1,
               **{f'Target_{j}': int((i+j) % 2) for j in range(1, 11)}}
              for i, (subject, study) in enumerate(keys, 1)]
    _csv(root/'manifests/train_labels.csv', list(labels[0]), labels)
    write_probabilities(root/'oof/D17_oof_all.csv',
                        {'keys': keys, 'p': np.full((5, 10), .5)}, {'model_id': 'D17'})
    _csv(root/'oof/D17_source_index.csv', ['Subject_id', 'Study_id', 'model_id', 'fold'],
         [{'Subject_id': subject, 'Study_id': study, 'model_id': f'D17_F{i:02d}', 'fold': i}
          for i, (subject, study) in enumerate(keys, 1)])
    for i in range(1, 6):
        run = root/'experiments'/f'D17_F{i:02d}'
        run.mkdir(parents=True)
        (run/'DONE.json').write_text('{}', encoding='utf-8')
        _csv(run/'training_image_index.csv', ['Subject_id'],
             [{'Subject_id': subject} for j, (subject, _) in enumerate(keys, 1) if j != i])
    assert verify_teacher_sources(root, ('D17',), expected_studies=5)['D17']['studies'] == 5
    _csv(root/'experiments/D17_F01/training_image_index.csv', ['Subject_id'],
         [{'Subject_id': subject} for subject, _ in keys])
    with pytest.raises(ValueError, match='fold isolation'):
        verify_teacher_sources(root, ('D17',), expected_studies=5)
