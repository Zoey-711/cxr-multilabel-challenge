"""CPU-only dry run of the 22+4 one-launch dependency chain."""
import json
from pathlib import Path

import pytest

from cxr.batch.orchestrator import run_all
from cxr.batch.extension_queue import run_extension_queue
from cxr.batch.plan import digest


def plan():
    base = [{'id': f'BASE_{i:02d}', 'enabled': True} for i in range(22)]
    extra = [{'id': name, 'depends_on': [task['id'] for task in base]} for name in
             ('G17_FULL_01', 'G17_FULL_02', 'H17_DISTILL_01', 'H17_DISTILL_02')]
    return {'tasks': base, 'conditional_extension': {'enabled': True, 'reserved_slots': extra}}


def test_one_launch_runs_22_then_four_and_resume_does_not_repeat(tmp_path, monkeypatch):
    root = tmp_path/'phase17_batch'
    config = plan()
    calls = []

    def fake_base(*args, **kwargs):
        calls.append(('base', args[4], kwargs['resume']))
        root.mkdir()
        (root/'queue_state.json').write_text(json.dumps({'gpu_hours': 10,
            'tasks': {t['id']: {'state': 'complete'} for t in config['tasks']}}), encoding='utf-8')

    def fake_extension(*args, **kwargs):
        calls.append(('extension', kwargs['gh_budget'], kwargs['resume'], kwargs['gpu_preflight']))
        record = {'state': 'complete', 'tasks': {t['id']: {'state': 'complete'}
                  for t in config['conditional_extension']['reserved_slots']}}
        (root/'extension_state.json').write_text(json.dumps(record), encoding='utf-8')
        return record

    monkeypatch.setattr('cxr.batch.runner.run_queue', fake_base)
    monkeypatch.setattr('cxr.batch.extension_queue.run_extension_queue', fake_extension)
    run_all(config, tmp_path/'plan.yaml', root, ['0', '1'], total_budget=120,
            gh_budget=30, max_concurrent=2, preflight_gpu_hours=.2)
    assert calls == [('base', 90, False), ('extension', 30, False, True)]
    run_all(config, tmp_path/'plan.yaml', root, ['0', '1'], total_budget=120,
            gh_budget=30, max_concurrent=2, preflight_gpu_hours=.2)
    assert len(calls) == 2


def test_incomplete_base_blocks_gh_without_losing_base_results(tmp_path, monkeypatch):
    root = tmp_path/'phase17_batch'; root.mkdir()
    config = plan()
    def partial(*args, **kwargs):
        (root/'queue_state.json').write_text(json.dumps({'gpu_hours': 10,
            'tasks': {t['id']: {'state': 'failed' if t['id']=='BASE_21' else 'complete'}
                      for t in config['tasks']}}), encoding='utf-8')
    def blocked(*args, **kwargs):
        assert kwargs['gpu_preflight'] is True
        return {'state': 'BLOCKED', 'tasks': {t['id']: {'state': 'BLOCKED'}
                for t in config['conditional_extension']['reserved_slots']}}
    monkeypatch.setattr('cxr.batch.runner.run_queue', partial)
    monkeypatch.setattr('cxr.batch.extension_queue.run_extension_queue', blocked)
    result = run_all(config, tmp_path/'plan.yaml', root, ['0'], total_budget=120,
                     gh_budget=30, max_concurrent=1, preflight_gpu_hours=.2)
    assert all(x['state']=='BLOCKED' for x in result['tasks'].values())
    assert (root/'queue_state.json').is_file()


def test_budget_and_base_exception_fail_closed(tmp_path, monkeypatch):
    root = tmp_path/'phase17_batch'
    config = plan()
    with pytest.raises(ValueError, match='subbudget'):
        run_all(config, tmp_path/'plan.yaml', root, ['0'], total_budget=30,
                gh_budget=30, max_concurrent=1, preflight_gpu_hours=0)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic base failure')
    monkeypatch.setattr('cxr.batch.runner.run_queue', fail)
    with pytest.raises(RuntimeError, match='synthetic base failure'):
        run_all(config, tmp_path/'plan.yaml', root, ['0'], total_budget=120,
                gh_budget=30, max_concurrent=1, preflight_gpu_hours=.2)
    blocked = json.loads((root/'reports/extension_blocked.json').read_text(encoding='utf-8'))
    assert len(blocked['tasks']) == 4 and blocked['state'] == 'BLOCKED'


def test_one_gh_gpu_smoke_failure_does_not_cancel_other_gh_tasks(tmp_path, monkeypatch):
    root=tmp_path/'phase17_batch'; root.mkdir()
    config=plan()
    (root/'queue_state.json').write_text(json.dumps({'gpu_hours':1.,
        'tasks':{task['id']:{'state':'complete'} for task in config['tasks']}}),encoding='utf-8')
    (root/'manifests').mkdir()
    (root/'manifests/extension_selection.json').write_text('{}',encoding='utf-8')
    tasks=[]
    for slot in config['conditional_extension']['reserved_slots']:
        path=tmp_path/(slot['id']+'.yaml');path.write_text(slot['id'],encoding='utf-8')
        tasks.append({'id':slot['id'],'config':str(path),'config_sha256':digest(path),
                      'timeout_hours':1,'max_retries':0})
    monkeypatch.setattr('cxr.batch.extension_queue.prepare_extension',lambda *_:tasks)
    monkeypatch.setattr('cxr.batch.extension_queue.extension_report',lambda *_:None)
    monkeypatch.setattr('cxr.batch.extension_queue.execute',
                        lambda *a:{'state':'complete','gpu_hours':.01})
    fail={'H17_DISTILL_02':True}
    def smoke(command, env, stdout, stderr, timeout):
        name=command[command.index('--run-id')+1]
        if name.startswith('H17_DISTILL_02') and fail['H17_DISTILL_02']:
            return 1
        folder=root/'experiments'/name;folder.mkdir(parents=True)
        task=next(task for task in tasks if name.startswith(task['id']))
        (folder/'config.yaml').write_bytes(Path(task['config']).read_bytes())
        (folder/'status.json').write_text(json.dumps({'state':'preflight_passed',
            'amp_skipped_steps':0}),encoding='utf-8')
        return 0
    monkeypatch.setattr('cxr.batch.extension_queue.run_command',smoke)
    result=run_extension_queue(root,config,['0','1'],total_budget=100,
                               gh_budget=20,max_concurrent=2,pause_seconds=.001)
    assert result['state']=='partial'
    assert result['tasks']['H17_DISTILL_02']['state']=='BLOCKED'
    assert all(result['tasks'][task['id']]['state']=='complete' for task in tasks[:-1])
    fail['H17_DISTILL_02']=False
    resumed=run_extension_queue(root,config,['0','1'],total_budget=100,
                                gh_budget=20,max_concurrent=2,pause_seconds=.001,resume=True)
    assert resumed['state']=='complete'
    assert all(item['state']=='complete' for item in resumed['tasks'].values())
