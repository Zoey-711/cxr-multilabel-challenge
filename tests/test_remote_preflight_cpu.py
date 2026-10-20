"""CPU-only checks for the remote gate; no CUDA allocation or real data access."""
import copy
import xml.etree.ElementTree as ET

import pytest

from scripts.batch_run import validate_remote_receipt
from scripts.remote_preflight import require_three_cuda_tests, stage_sizes
from cxr.batch.plan import ROOT


def receipt():
    environment = {key: f'/authorized/{key.lower()}' for key in
                   ('CXR_DATA_ROOT', 'CXR_PRETRAINED_ROOT', 'CXR_BACKUP_ROOT', 'CXR_OUTPUT_ROOT')}
    return {'kind': 'REMOTE_PREFLIGHT', 'host': 'friend-host', 'plan_sha256': 'sealed-plan',
            'state': 'passed', 'gpus': [str(i) for i in range(8)], 'max_concurrent': 8,
            'cuda_tests': [{'name': f'cuda_{i}'} for i in range(3)],
            'real_data_smoke': 'passed', 'real_inference': {'val': {}, 'test': {}},
            'environment': environment,
            'stages': [{'concurrent_gpus': i, 'state': 'passed'} for i in (1, 2, 4, 8)]}


def test_staged_allocation_plan_is_gradual():
    assert stage_sizes(1) == [1]
    assert stage_sizes(3) == [1, 2, 3]
    assert stage_sizes(8) == [1, 2, 4, 8]
    with pytest.raises(ValueError):
        stage_sizes(9)


def test_training_gate_requires_all_remote_evidence():
    item = receipt()
    args = ('sealed-plan', [str(i) for i in range(8)], 8, item['environment'], 'friend-host')
    validate_remote_receipt(item, *args)
    for field, bad in [('state', 'failed'), ('kind', 'ordinary_preflight'),
                       ('cuda_tests', []), ('real_inference', {'test': {}}),
                       ('stages', item['stages'][:-1]), ('plan_sha256', 'changed')]:
        altered = copy.deepcopy(item)
        altered[field] = bad
        with pytest.raises(ValueError):
            validate_remote_receipt(altered, *args)
    with pytest.raises(ValueError):
        validate_remote_receipt(item, *args[:-1], 'different-host')


def test_three_previously_skipped_cuda_cases_must_really_pass(tmp_path):
    report = tmp_path/'cuda.xml'
    suite = ET.Element('testsuite')
    for name in ('resume_workers_0', 'resume_workers_2', 'archive_inference'):
        ET.SubElement(suite, 'testcase', name=name)
    ET.ElementTree(suite).write(report)
    assert len(require_three_cuda_tests(report)) == 3
    ET.SubElement(suite[0], 'skipped')
    ET.ElementTree(suite).write(report)
    with pytest.raises(RuntimeError, match='skipped'):
        require_three_cuda_tests(report)


def test_documented_entry_points_exist_and_base_start_is_fail_closed():
    readme = (ROOT/'README_friend.md').read_text(encoding='utf-8')
    for name in ('friend_remote_preflight.sh', 'friend_start.sh'):
        assert (ROOT/'scripts'/name).is_file()
        assert f'bash scripts/{name} --config configs/friend_server.local.env' in readme
    start = (ROOT/'scripts/friend_start.sh').read_text(encoding='utf-8')
    assert 'REMOTE_PREFLIGHT receipt missing' in start
    assert 'scripts/batch_preflight.py' not in start
    assert '--action run-all' in start
    assert '--gh-gpu-hours "$CXR_GH_GPU_HOURS"' in start
    assert '--action clear-stop' in readme
