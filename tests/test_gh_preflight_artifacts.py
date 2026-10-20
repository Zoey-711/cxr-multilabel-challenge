"""CPU-only checks that a partial G/H smoke cannot unlock formal training."""
import csv
import json

import pytest

from cxr.batch.gh_preflight import PROB_COLUMNS, verify_gh_preflight
from cxr.batch.plan import digest


def _evidence(folder, config_sha='abc'):
    folder.mkdir()
    checkpoint = folder / 'probe_resume.pth'
    checkpoint.write_bytes(b'synthetic-resume-state')
    report = {'status': 'passed', 'config_sha256': config_sha,
              'checkpoint_sha256': digest(checkpoint), 'optimizer_steps': 1,
              'amp_skipped_steps': 0, 'checkpoint_restored': True,
              'optimizer_restored': True, 'scheduler_restored': True,
              'scaler_restored': True, 'rng_restored': True}
    for split in ('val', 'test'):
        path = folder / f'{split}_probe.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(PROB_COLUMNS)
            writer.writerow(['00001', '00002', *([0.25] * 10)])
        report[split] = {'real_images': 2, 'studies': 1, 'sha256': digest(path)}
    (folder / 'probe_report.json').write_text(json.dumps(report), encoding='utf-8')
    return report


def test_gh_probe_requires_resume_and_both_real_prediction_files(tmp_path):
    folder = tmp_path / 'G17_FULL_01_GH_PREFLIGHT'
    report = _evidence(folder)
    assert verify_gh_preflight(folder, 'abc')['status'] == 'passed'
    with pytest.raises(ValueError, match='stale'):
        verify_gh_preflight(folder, 'different-config')
    (folder / 'test_probe.csv').unlink()
    with pytest.raises(FileNotFoundError):
        verify_gh_preflight(folder, 'abc')
    _evidence_again = folder / 'test_probe.csv'
    with _evidence_again.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(PROB_COLUMNS)
        writer.writerow(['00001', '00002', *([0.25] * 10)])
    report['scaler_restored'] = False
    (folder / 'probe_report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='checkpoint'):
        verify_gh_preflight(folder, 'abc')


def test_gh_probe_rejects_bad_class_order_and_probability(tmp_path):
    folder = tmp_path / 'H17_DISTILL_01_GH_PREFLIGHT'
    report = _evidence(folder)
    path = folder / 'val_probe.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Subject_id', 'Study_id', 'Prob_2', 'Prob_1', *PROB_COLUMNS[4:]])
        writer.writerow(['00001', '00002', *([0.25] * 10)])
    report['val']['sha256'] = digest(path)
    (folder / 'probe_report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='class order'):
        verify_gh_preflight(folder, 'abc')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(PROB_COLUMNS)
        writer.writerow(['00001', '00002', *([float('nan')] * 10)])
    report['val']['sha256'] = digest(path)
    (folder / 'probe_report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='probability invalid'):
        verify_gh_preflight(folder, 'abc')
