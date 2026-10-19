"""Exercise actual scheduler with tiny fake tasks, not GPU jobs."""
import threading
from pathlib import Path
from cxr.batch import runner


def setup(monkeypatch,tmp_path):
    plan_path=tmp_path/'plan';plan_path.write_text('frozen')
    monkeypatch.setattr(runner,'initialize_manifests',lambda *a:None)
    monkeypatch.setattr(runner,'record_failure',lambda *a:None)
    monkeypatch.setattr('cxr.batch.reporting.report',lambda *a:None)
    monkeypatch.setattr(runner.time,'sleep',lambda x:None)
    return plan_path


def task(name,deps=()):
    return {'id':name,'enabled':True,'group':'A','depends_on':list(deps),'timeout_hours':1,'max_retries':1}


def test_queue_finite_retry_dependencies_and_budget(monkeypatch,tmp_path):
    path=setup(monkeypatch,tmp_path);calls=[]
    def execute(t,root,gpu,attempt):
        calls.append((t['id'],gpu,attempt))
        if t['id']=='A' and attempt==1:return {'state':'failed','retryable':True,'reason':'temporary I/O','gpu_hours':.1}
        return {'state':'complete','gpu_hours':.1}
    monkeypatch.setattr(runner,'execute',execute)
    plan={'tasks':[task('A'),task('B',['A'])]}
    result=runner.run_queue(plan,path,tmp_path/'queue',['0','1'],4)
    assert [x[0] for x in calls]==['A','A','B']
    assert result['tasks']['A']['attempts']==2
    assert result['tasks']['B']['state']=='complete'
    calls.clear()
    result=runner.run_queue(plan,path,tmp_path/'budget',['0'],.5)
    assert calls==[] and result['tasks']['A']['state']=='pending'


def test_oom_never_retried_and_dependent_never_launched(monkeypatch,tmp_path):
    path=setup(monkeypatch,tmp_path);calls=[]
    def execute(t,*args):
        calls.append(t['id']);return {'state':'failed','reason':'CUDA out of memory','retryable':False,'gpu_hours':.1}
    monkeypatch.setattr(runner,'execute',execute)
    result=runner.run_queue({'tasks':[task('A'),task('B',['A'])]},path,tmp_path/'q',['0'],4)
    assert calls==['A'] and result['tasks']['A']['state']=='failed'
    assert result['tasks']['B']['state']=='pending'


def test_unique_gpu_and_stop_drain(monkeypatch,tmp_path):
    path=setup(monkeypatch,tmp_path);active=set();guard=threading.Lock();calls=[]
    root=tmp_path/'q'
    def execute(t,root,gpu,attempt):
        with guard:
            assert gpu not in active
            active.add(gpu);calls.append(t['id'])
            (root/'STOP').touch()
            active.remove(gpu)
        return {'state':'complete','gpu_hours':.01}
    monkeypatch.setattr(runner,'execute',execute)
    runner.run_queue({'tasks':[task('A'),task('B',['A'])]},path,root,['0'],4)
    assert calls==['A']
