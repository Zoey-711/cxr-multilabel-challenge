"""CPU-only one-task-per-GPU queue and resume rehearsal."""
import json
import threading

from cxr.batch import runner


def test_dynamic_gpu_queue_never_double_books_or_repeats(tmp_path, monkeypatch):
    plan_path=tmp_path/'plan.yaml'
    plan_path.write_text('synthetic-plan',encoding='utf-8')
    tasks=[{'id':f'T{i}','enabled':True,'depends_on':[],'timeout_hours':.001,
            'max_retries':0,'group':'A'} for i in range(4)]
    root=tmp_path/'phase17_batch'
    active=set();calls=[];lock=threading.Lock()

    def fake_execute(task, batch_root, gpu, attempt):
        with lock:
            assert gpu not in active
            active.add(gpu)
            calls.append((task['id'],gpu,attempt))
        threading.Event().wait(.02)
        with lock:
            active.remove(gpu)
        return {'state':'complete','gpu_hours':.00001}

    def fake_initialize(path, plan, filename):
        (path/'reports').mkdir(parents=True)
        (path/'manifests').mkdir(parents=True)

    monkeypatch.setattr(runner,'execute',fake_execute)
    monkeypatch.setattr(runner,'initialize_manifests',fake_initialize)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(__import__('cxr.batch.reporting',fromlist=['report']),'report',lambda _:None)
    first=runner.run_queue({'tasks':tasks},plan_path,root,['0','1'],.01,max_concurrent=2)
    assert all(item['state']=='complete' for item in first['tasks'].values())
    assert len(calls)==4 and {gpu for _,gpu,_ in calls}=={'0','1'}
    rows=(root/'reports/experiment_status.csv').read_text(encoding='utf-8')
    assert all(task['id'] in rows for task in tasks)
    resumed=runner.run_queue({'tasks':tasks},plan_path,root,['0','1'],.01,resume=True,max_concurrent=2)
    assert len(calls)==4 and all(item['state']=='complete' for item in resumed['tasks'].values())
    assert not list(root.rglob('*.tar.gz'))
