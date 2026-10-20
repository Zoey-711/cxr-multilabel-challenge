"""CPU-only synthetic checks of one-command gate and resume semantics."""
import json
import socket

import pytest

from cxr.batch.plan import write_json
from scripts import friend_one_launch as launch_module
from scripts.remote_gh_route_probe import prepare_probe_configs


def settings(tmp_path):
    return {**{key: str(tmp_path/key) for key in launch_module.REQUIRED},
            'CXR_GPUS': '0,1,2,3,4,5,6,7', 'CXR_MAX_CONCURRENT': '8',
            'CXR_PREFLIGHT_GPU_HOURS': '1', 'CXR_GPU_HOURS': '100',
            'CXR_GH_GPU_HOURS': '20', 'CXR_PREFLIGHT_VERSION': 'v1'}


def receipt(plan_path, env):
    return {'kind': 'REMOTE_PREFLIGHT', 'host': socket.gethostname(),
            'plan_sha256': launch_module.digest(plan_path), 'state': 'passed',
            'gpus': [str(i) for i in range(8)], 'max_concurrent': 8,
            'cuda_tests': [{'name': str(i)} for i in range(3)],
            'real_data_smoke': 'passed', 'real_inference': {'val': {'real_images': 2},
                                                          'test': {'real_images': 2}},
            'gh_route_probe': {'state': 'passed', 'runs': [{} for _ in range(6)]},
            'gpu_inventory': {'rtx_4090_count': 8},
            'stages': [{'concurrent_gpus': i, 'state': 'passed'} for i in (1, 2, 4, 8)],
            'environment': {key: env[key] for key in
                            ('CXR_DATA_ROOT', 'CXR_PRETRAINED_ROOT', 'CXR_BACKUP_ROOT', 'CXR_OUTPUT_ROOT')},
            'gpu_hours': 0.2}


def test_budget_and_resources_fail_before_any_gpu(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    env = settings(tmp_path)
    assert launch_module.authorized_settings(env)[:2] == ([str(i) for i in range(8)], 8)
    for key, value in [('CXR_GPU_HOURS', ''), ('CXR_GH_GPU_HOURS', '0'),
                       ('CXR_MAX_CONCURRENT', '9'), ('CXR_PREFLIGHT_GPU_HOURS', 'nan')]:
        bad = {**env, key: value}
        with pytest.raises(ValueError):
            launch_module.authorized_settings(bad)


def test_one_launch_preflight_then_resume_without_repeating_preflight(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    plan_path = tmp_path/'plan.yaml'
    plan_path.write_text('synthetic', encoding='utf-8')
    plan = {'tasks': [{'id': 'D17_F01', 'enabled': True}],
            'conditional_extension': {'reserved_slots': [{'id': 'G17_FULL_01'}]}}
    monkeypatch.setattr(launch_module, 'load_plan', lambda _: plan)
    monkeypatch.setattr(launch_module, 'verify_preflight_artifacts', lambda *_: None)
    env = settings(tmp_path)
    root = tmp_path/'CXR_OUTPUT_ROOT/phase17_batch'
    preflight_dir = tmp_path/'CXR_OUTPUT_ROOT/phase17_batch_preflight_v1'
    calls = []

    def runner(cmd, log):
        calls.append(cmd)
        if 'remote_preflight.py' in cmd[1]:
            write_json(preflight_dir/'receipt.json', receipt(plan_path, env))
        else:
            write_json(root/'queue_state.json', {'tasks': {'D17_F01': {'state': 'complete'}}})
            write_json(root/'extension_state.json', {'tasks': {'G17_FULL_01': {'state': 'complete'}}})
            write_json(root/'experiments/D17_F01/DONE.json', {'verified_sha256': {}})

    result = launch_module.launch(plan_path, env, runner=runner)
    assert result['complete_count'] == 2
    assert len(calls) == 3
    assert 'protocol_assets.py' in calls[0][1]
    assert 'remote_preflight.py' in calls[1][1]
    assert '--action' in calls[2] and 'run-all' in calls[2]
    launch_module.launch(plan_path, env, runner=runner)
    assert len(calls) == 5 and 'remote_preflight.py' not in calls[4][1]
    assert (root/'manifests/one_launch_delivery_manifest.json').is_file()


def test_failed_preflight_or_live_lock_never_launches(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    plan_path = tmp_path/'plan.yaml'; plan_path.write_text('synthetic', encoding='utf-8')
    monkeypatch.setattr(launch_module, 'load_plan', lambda _: {'tasks': [],
        'conditional_extension': {'reserved_slots': []}})
    monkeypatch.setattr(launch_module, 'verify_preflight_artifacts', lambda *_: None)
    env = settings(tmp_path)
    preflight_dir = tmp_path/'CXR_OUTPUT_ROOT/phase17_batch_preflight_v1'
    item = receipt(plan_path, env); item['state'] = 'failed'
    write_json(preflight_dir/'receipt.json', item)
    called = []
    with pytest.raises(ValueError):
        launch_module.launch(plan_path, env, runner=lambda *args: called.append(args))
    assert len(called) == 1 and 'protocol_assets.py' in called[0][0][1]
    item['state'] = 'passed'; write_json(preflight_dir/'receipt.json', item)
    root = tmp_path/'CXR_OUTPUT_ROOT/phase17_batch'
    write_json(root/'queue_state.json', {'tasks': {}})
    write_json(root/'queue.lock', {'pid': __import__('os').getpid(), 'host': socket.gethostname()})
    with pytest.raises(RuntimeError, match='owner is live'):
        launch_module.launch(plan_path, env, runner=lambda *args: called.append(args))
    assert len(called) == 2 and 'protocol_assets.py' in called[1][0][1]


def test_dead_local_lock_can_resume_but_live_gpu_child_blocks(tmp_path):
    root = tmp_path/'phase17_batch'
    write_json(root/'queue_state.json', {'tasks': {}})
    write_json(root/'queue.lock', {'pid': 2147483647, 'host': socket.gethostname()})
    launch_module.recover_stale_locks(root)
    assert not (root/'queue.lock').exists()
    write_json(root/'queue_logs/task/train.stdout.log.pid.json',
               {'pid': __import__('os').getpid(), 'host': socket.gethostname()})
    with pytest.raises(RuntimeError, match='GPU child process'):
        launch_module.recover_stale_locks(root)


def test_route_config_is_isolated_and_distillation_is_probe_only(monkeypatch, tmp_path):
    import yaml
    from cxr.batch.plan import ROOT
    plan = yaml.safe_load((ROOT/'configs/batch_plan.yaml').read_text(encoding='utf-8'))
    configs = prepare_probe_configs(plan, tmp_path/'probe')
    assert len(configs) == 6
    g = yaml.safe_load(configs[0][1].read_text(encoding='utf-8'))
    h = yaml.safe_load(configs[-1][1].read_text(encoding='utf-8'))
    assert 'train_subset' not in g['data']
    assert h['training']['distillation']['hard_fraction'] == 0.5
    assert h['data']['teacher_soft_labels']['formula'] == {'D17': .25, 'C17': .25, 'S17': .5}
    assert h['data']['teacher_soft_labels']['path'].startswith(str(tmp_path))
