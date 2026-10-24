"""CPU/synthetic acceptance only. No real images or CUDA execution."""
import copy
import csv
import json
import shutil
import threading
import time
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from cxr.phase32.protocol import load_plan,ROOT,server_config
from cxr.phase32.methods import study_positive_weights,loss_elements,mixup_batch
from cxr.phase32.results import csv_write,finalize,verify_done,report
from cxr.batch.plan import write_json

@pytest.mark.parametrize('task',load_plan()['tasks'],ids=lambda t:t['id'])
def test_all_pilot_configs_and_cpu_loss(task):
    cfg=yaml.safe_load((ROOT/task['config']).read_text())
    training=cfg['training'];training['_resolved_pos_weight']=[2.]*10
    x=torch.randn(4,10,requires_grad=True);y=torch.randint(0,2,(4,10)).float()
    loss=loss_elements(x,y,training).mean();loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()

def test_study_support_not_image_weighted():
    rows=[{'Subject_id':str(i),'Study_id':str(i),'Predict_class':';'.join(str(c) for c in range(1,11) if (i+c)%3==0)} for i in range(12)]
    assert study_positive_weights(rows)==study_positive_weights(rows+rows[:2]*25)

def test_mixup_resume_rng_and_gradient():
    from cxr.utils.training_resume import capture_rng_state,restore_rng_state
    from torch.utils.data import DataLoader,TensorDataset
    cfg=yaml.safe_load((ROOT/'configs/phase32/P32_NS_MIXUP.yaml').read_text())['training']
    x=torch.randn(8,4);y=torch.randint(0,2,(8,10)).float();m=torch.ones_like(y)
    loader=DataLoader(TensorDataset(x),generator=torch.Generator().manual_seed(42))
    model=torch.nn.Linear(4,10);opt=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(opt,1)
    state=capture_rng_state(loader,loader)
    a=mixup_batch(x,y,m,cfg)
    loss_elements(model(a[0]),a[1],cfg).mean().backward();opt.step();scheduler.step()
    restore_rng_state(state,loader,loader);b=mixup_batch(x,y,m,cfg)
    assert all(torch.equal(v,w) for v,w in zip(a,b))
    with pytest.raises(ValueError):mixup_batch(x,y,m*0,cfg)

def test_budget_required(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/phase32_server.example.yaml').read_text())
    for k in ['data_root','pretrained_root','protocol_root','baseline_results_root','output_root']:cfg[k]=str(tmp_path/'phase32_batch')
    p=tmp_path/'server.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='budget|GPU-hour'):server_config(p)
    cfg['gpu_hours']=100;p.write_text(yaml.safe_dump(cfg));assert server_config(p)['max_concurrent']==8

def _predictions(path,count,patients):
    csv_write(path,[{'Subject_id':f'p{i%patients}','Study_id':f's{i}','image_count':1+i%2,
        **{f'probability_{c}':(.8 if (i+c)%2 else .2) for c in range(1,11)},
        **{f'target_{c}':(i+c)%2 for c in range(1,11)}} for i in range(count)])

def test_complete_export_relocation_and_tamper(tmp_path):
    plan=load_plan();task=plan['tasks'][0];run=tmp_path/'experiments'/task['id'];run.mkdir(parents=True)
    write_json(run/'status.json',{'state':'complete'})
    shutil.copy2(ROOT/task['config'],run/'config.yaml')
    row={'epoch':1,'val':{'macro_auc':1.,'macro_f1':1.},'elapsed_seconds':1.,'gpu_peak_allocated_mib':0}
    (run/'metrics.jsonl').write_text(json.dumps(row)+'\n');(run/'train_stdout.log').write_text('synthetic CPU only')
    write_json(run/'best_metrics.json',{'epoch':1});write_json(run/'pretrained_source.json',{'source':'SYNTHETIC'})
    torch.save({'model_state_dict':{'weight':torch.ones(1)}},run/'best_model.pt')
    _predictions(run/'validation_analysis/study_predictions.csv',859,228)
    _predictions(run/'oof_analysis/study_predictions.csv',3687,2211)
    finalize(run,task,plan,tmp_path/'no-baselines')
    data=json.loads((run/'metrics.json').read_text());assert data['test_auc'] is None
    assert data['variants']['raw']['val']['all']['macro_auc']==1
    other=tmp_path/'relocated';shutil.copytree(run,other);verify_done(other)
    assert torch.load(other/'best_model.pth',weights_only=True)['weight'].item()==1
    (other/'val_predictions.csv').write_text('bad')
    with pytest.raises(ValueError):verify_done(other)

def test_queue_failure_retry_budget_resume(tmp_path):
    from cxr.phase32.queue import run
    plan=load_plan();calls=[];active=set();guard=threading.Lock()
    cfg={'output_root':str(tmp_path/'phase32_batch'),'gpu_ids':[0,1],'max_concurrent':2,'gpu_hours':100}
    def execute(task,root,gpu,attempt):
        with guard:assert gpu not in active;active.add(gpu);calls.append((task['id'],attempt))
        time.sleep(.002)
        with guard:active.remove(gpu)
        if task['id']=='P32_NS_ASL':return {'state':'FAILED','reason':'CUDA OOM','retryable':False,'gpu_hours':.01}
        if task['id']=='P32_NS_POS' and attempt==1:return {'state':'FAILED','reason':'temporary IO','retryable':True,'gpu_hours':.01}
        p=root/'experiments'/task['id'];p.mkdir(parents=True,exist_ok=True)
        write_json(p/'DONE.json',{'sha256':{}})
        return {'state':'DONE','gpu_hours':.01}
    state=run(plan,cfg,execute,poll_seconds=.001)
    assert sum(r['state']=='DONE' for r in state['tasks'].values())==7
    assert len(calls)==9
    again=run(plan,cfg,execute,poll_seconds=.001);assert len(calls)==9
    assert again['gpu_hours']==state['gpu_hours']
    cfg['output_root']=str(tmp_path/'budget');cfg['gpu_hours']=1
    assert all(r['state']=='PENDING' for r in run(plan,cfg,execute)['tasks'].values())

def test_small_export_excludes_images_weights(tmp_path):
    from cxr.phase32.upload import export_small
    run=tmp_path/'source/experiments/P32_X';run.mkdir(parents=True)
    write_json(run/'DONE.json',{'sha256':{}});write_json(run/'metrics.json',{'val_auc':.8})
    (run/'best_model.pth').write_bytes(b'not exported');(run/'image.png').write_bytes(b'not exported')
    files=export_small(tmp_path/'source',tmp_path/'export')
    assert list(files)==['experiments/P32_X/metrics.json']

def test_preflight_windows_fail_closed(tmp_path):
    import platform
    if platform.system()!='Windows':pytest.skip('Windows-specific local protection')
    from cxr.phase32.preflight import prepare
    with pytest.raises(RuntimeError,match='Linux'):prepare({},load_plan())

def test_no_phase17_entry_mutation():
    import subprocess
    assert not subprocess.check_output(['git','diff','--','scripts/batch_train.py','src/cxr/batch/runner.py'],cwd=ROOT)

@pytest.mark.parametrize('size',[384,448,512])
def test_isolated_preprocessing(size):
    from PIL import Image
    from cxr.phase32.dataset import preprocess_image
    from cxr.data.dataset import preprocess_image as original
    image=Image.fromarray(np.arange(79*121*3,dtype=np.uint8).reshape(79,121,3))
    value=preprocess_image(image,size=size)
    assert value.shape==(3,size,size) and torch.isfinite(value).all()
    if size!=448:assert torch.equal(value,original(image,size=size))

def test_preflight_identity_roundtrip():
    from cxr.phase32.protocol import fingerprint
    identity={'devices':{'0':['uuid','RTX4090','24564']},'plan':'sealed'}
    assert identity==json.loads(json.dumps(identity))
    assert fingerprint(identity)==fingerprint(json.loads(json.dumps(identity)))

def test_cpu_optimizer_checkpoint_next_step(tmp_path):
    import io
    from cxr.utils.training_resume import capture_rng_state,restore_rng_state
    from torch.utils.data import DataLoader,TensorDataset
    cfg=yaml.safe_load((ROOT/'configs/phase32/P32_NS_MIXUP.yaml').read_text())['training']
    x=torch.arange(32).reshape(8,4).float()/32;y=torch.randint(0,2,(8,10)).float()
    loader=DataLoader(TensorDataset(x),generator=torch.Generator().manual_seed(42))
    def setup():
        model=torch.nn.Linear(4,10);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        return model,opt,torch.optim.lr_scheduler.StepLR(opt,1,.9)
    def step(model,opt,scheduler):
        xx,yy,_=mixup_batch(x,y,torch.ones_like(y),cfg)
        opt.zero_grad();loss_elements(model(xx),yy,cfg).mean().backward();opt.step();scheduler.step()
    model,opt,scheduler=setup();step(model,opt,scheduler)
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
        'scheduler':scheduler.state_dict(),'rng':capture_rng_state(loader,loader)},buffer)
    step(model,opt,scheduler);expected=copy.deepcopy(model.state_dict())
    buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    m,o,s=setup();m.load_state_dict(saved['model']);o.load_state_dict(saved['optimizer']);s.load_state_dict(saved['scheduler'])
    restore_rng_state(saved['rng'],loader,loader);step(m,o,s)
    assert all(torch.equal(expected[k],v) for k,v in m.state_dict().items())
    assert o.param_groups[0]['lr']==opt.param_groups[0]['lr']
