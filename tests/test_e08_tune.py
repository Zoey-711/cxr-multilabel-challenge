"""CPU acceptance: exact interventions, EMA restore, gated expansion, aligned OOF."""
import copy
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from cxr.batch.plan import write_json,digest
from cxr.e08_tune.protocol import load_plan,ROOT,server_config
from cxr.e08_tune.selection import choose,select_pilots

@pytest.mark.parametrize('task',load_plan()['tasks'],ids=lambda t:t['id'])
def test_frozen_templates_cpu_loss(task):
    from cxr.phase32.methods import loss_elements
    config=yaml.safe_load((ROOT/task['config']).read_text(encoding='utf-8'))
    logits=torch.randn(3,10,requires_grad=True);targets=torch.randint(0,2,(3,10)).float()
    loss=loss_elements(logits,targets,config['training']).mean();loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert config['training']['epochs']==(40 if task['variant']=='LONG' else 30)
    assert config['training']['early_stopping_patience']==6
    assert config['training']['weight_decay']==1e-4
    assert task['expected_counts']['oof']['patients']==2211

@pytest.mark.parametrize('decay',[.998,.999,.9995])
def test_ema_buffer_and_checkpoint_next_update(decay):
    from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
    torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(4,10),torch.nn.BatchNorm1d(10))
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(decay),use_buffers=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    x=torch.randn(8,4);y=torch.randint(0,2,(8,10)).float()
    def update(m,e,o,s):
        o.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(m(x),y).backward()
        o.step();s.step();e.update_parameters(m)
    update(model,ema,optimizer,scheduler)
    previous=ema.module[1].running_mean.clone()
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'ema':ema.state_dict(),
        'opt':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),
        'best_epoch':1,'stale':0,'best_auc':.8},buffer)
    update(model,ema,optimizer,scheduler)
    assert torch.allclose(ema.module[1].running_mean,previous*decay+model[1].running_mean*(1-decay),atol=1e-7)
    buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    m=copy.deepcopy(model);m.load_state_dict(saved['model'])
    e=AveragedModel(m,multi_avg_fn=get_ema_multi_avg_fn(decay),use_buffers=True);e.load_state_dict(saved['ema'])
    o=torch.optim.AdamW(m.parameters(),lr=.0001);s=torch.optim.lr_scheduler.StepLR(o,1,.9)
    o.load_state_dict(saved['opt']);s.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng'])
    update(m,e,o,s)
    assert all(torch.equal(v,e.state_dict()[k]) for k,v in ema.state_dict().items())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in model.state_dict().items())

def metric(delta=.003,lower=.001,reg=-.01):
    return {'variants':{'ema':{'oof_comparison':{'differences':{'pilot_minus_baseline':{'point':delta,'ci95':[lower,.01]}}},
                                  'followup_evidence':{'worst_class_delta':reg}}}}

def test_selection_max_two_stable_and_ema_only():
    records={'LONG':metric(),'LR_LOW':metric(.004),'DECAY_HIGH':metric(.0035),
             'DECAY_LOW':metric(.02,-.001),'LR_HIGH':metric(.03,.01,-.03)}
    assert choose(records,load_plan()['selection'])==['LR_LOW','DECAY_HIGH']
    assert choose({'B':metric(),'A':metric(),'C':metric()},load_plan()['selection'])==['A','B']
    assert choose({'A':metric(.001)},load_plan()['selection'])==[]

def test_dependency_incomplete_blocks_expansion(tmp_path):
    plan=load_plan();tasks={t['id']:{'state':'DONE' if t['variant']!='LONG' else 'FAILED'} for t in plan['tasks'] if t['fold']==1}
    write_json(tmp_path/'pilot/queue_state.json',{'tasks':tasks})
    assert select_pilots(tmp_path,plan) is None
    assert json.loads((tmp_path/'reports/extension_status.json').read_text())['state']=='BLOCKED'

def test_expansion_selection_is_frozen(tmp_path):
    plan=load_plan();tasks={};
    for t in plan['tasks']:
        if t['fold']!=1:continue
        tasks[t['id']]={'state':'DONE'};run=tmp_path/'pilot/experiments'/t['id']
        write_json(run/'metrics.json',metric(.003 if t['variant']=='LR_LOW' else .001))
        write_json(run/'DONE.json',{'sha256':{'metrics.json':digest(run/'metrics.json')}})
    write_json(tmp_path/'pilot/queue_state.json',{'tasks':tasks})
    assert select_pilots(tmp_path,plan)==['LR_LOW']
    assert select_pilots(tmp_path,plan)==['LR_LOW']
    path=tmp_path/'pilot/experiments/ET_LR_LOW_F01/metrics.json';write_json(path,metric(.009))
    with pytest.raises(ValueError):select_pilots(tmp_path,plan)

def test_authorization_and_local_platform_guard(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/e08_tune_server.example.yaml').read_text())
    for k in ('data_root','pretrained_root','protocol_root','baseline_results_root','output_root'):cfg[k]=str(tmp_path/'e08_tuning')
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):server_config(path)
    cfg.update(gpu_hours=100,extension_gpu_hours=80);path.write_text(yaml.safe_dump(cfg))
    assert server_config(path)['extension_gpu_hours']==80
    import platform
    if platform.system()=='Windows':
        from cxr.e08_tune.preflight import prepare
        with pytest.raises(RuntimeError,match='Linux'):prepare(cfg,load_plan())

def test_pilot_queue_resume_and_budget(tmp_path):
    from cxr.e08_tune.queue import run
    plan=load_plan();plan['tasks']=[t for t in plan['tasks'] if t['fold']==1]
    cfg={'output_root':str(tmp_path/'pilot'),'gpu_ids':[0,1],'max_concurrent':2,'gpu_hours':70};calls=[]
    def execute(task,root,gpu,attempt):
        calls.append(task['id']);write_json(root/'experiments'/task['id']/'DONE.json',{'sha256':{}})
        return {'state':'DONE','gpu_hours':.01}
    state=run(plan,cfg,execute,poll_seconds=.001)
    run(plan,cfg,execute,poll_seconds=.001)
    assert len(calls)==5 and all(v['state']=='DONE' for v in state['tasks'].values())
    assert not any(t['fold']!=1 for t in plan['tasks'])

def test_no_change_to_phase32_seal():
    from cxr.phase32.protocol import load_plan as previous
    assert len(previous()['tasks'])==8

def test_five_fold_incomplete_is_not_full(tmp_path):
    from cxr.e08_tune.selection import complete_folds
    result=complete_folds(tmp_path,load_plan(),['LR_LOW'],tmp_path/'baseline')
    assert result=={}
    assert 'INCOMPLETE' in (tmp_path/'reports/five_fold_summary.csv').read_text()

def test_full_oof_alignment_and_bootstrap_pipeline(tmp_path,monkeypatch):
    import cxr.e08_tune.selection as module
    from cxr.e08_tune.results import csv_write
    plan=load_plan();plan['selection']['bootstrap_valid_draws']=2
    tasks=sorted([t for t in plan['tasks'] if t['variant']=='LR_LOW'],key=lambda t:t['fold'])
    baseline=tmp_path/'baseline';audit={'baselines':{},'e17_oof_sources':{}}
    offset=0
    for task in tasks:
        f=task['fold'];n=task['expected_counts']['oof']['studies'];patients=2211
        rows=[{'Subject_id':f'p{f}_{i%patients}','Study_id':f's{offset+i}','image_count':1+(i%2),
            **{f'probability_{j}':(.7 if (i+j)%2 else .3) for j in range(1,11)},
            **{f'target_{j}':(i+j)%2 for j in range(1,11)}} for i in range(n)]
        val=[{**r,'Subject_id':f'v{i%228}','Study_id':f'vs{i}'} for i,r in enumerate(rows[:859])]
        run=tmp_path/('pilot' if f==1 else 'expansion')/'experiments'/task['id']
        csv_write(run/'ema_oof_analysis/study_predictions.csv',rows)
        csv_write(run/'ema_validation_analysis/study_predictions.csv',val)
        (run/'ema_best_model.pth').write_bytes(b'SYNTHETIC');(run/'resolved_config.yaml').write_text('synthetic: true')
        write_json(run/'DONE.json',{'sha256':{p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file()}})
        nid=f'N17_F{f:02d}';p=baseline/'experiments'/nid/'ema_oof_analysis/study_predictions.csv';csv_write(p,rows)
        audit['baselines'][nid]={'sha256':{'ema_oof_analysis/study_predictions.csv':digest(p)}}
        for branch in ('D','C','S'):
            mid=f'{branch}17_F{f:02d}';p=baseline/'experiments'/mid/'oof_analysis/study_predictions.csv';csv_write(p,rows)
            audit['e17_oof_sources'][mid]={'prediction_sha256':digest(p)}
        offset+=n
    write_json(tmp_path/'docs/e08_tune/source_audit.json',audit);monkeypatch.setattr(module,'ROOT',tmp_path)
    result=module.complete_folds(tmp_path,plan,['LR_LOW'],baseline)['LR_LOW']
    assert result['oof']['studies']==18513 and result['oof']['patients']==11055
    assert result['oof']['macro_auc']==1 and result['val_ensemble']['macro_auc']==1
    assert result['full_oof_bootstrap']['valid_draws']==2
    assert result['folds2to5_confirmation']['differences']['pilot_minus_N17_EMA']['point']==0
    assert result['deployment']=='LATENCY_UNVERIFIED'
    # The pipeline must reject a modified baseline, not align silently by row number.
    p=baseline/'experiments/N17_F01/ema_oof_analysis/study_predictions.csv';p.write_text('changed')
    with pytest.raises(ValueError,match='baseline changed'):module.complete_folds(tmp_path,plan,['LR_LOW'],baseline)

def test_raw_ema_export_independent_and_portable(tmp_path):
    import shutil
    from cxr.e08_tune.results import finalize,verify_done,csv_write
    plan=load_plan();plan['selection']['bootstrap_valid_draws']=2
    task=next(t for t in plan['tasks'] if t['id']=='ET_DECAY_LOW_F01')
    run=tmp_path/'experiments'/task['id'];run.mkdir(parents=True)
    shutil.copy2(ROOT/task['config'],run/'config.yaml');write_json(run/'status.json',{'state':'complete'})
    history={'epoch':5,'val':{'macro_auc':1,'macro_f1':1},'ema_val':{'macro_auc':1},'elapsed_seconds':1,'gpu_peak_allocated_mib':0}
    (run/'metrics.jsonl').write_text(json.dumps(history)+'\n');(run/'train_stdout.log').write_text('CPU SYNTHETIC')
    write_json(run/'pretrained_source.json',{'source':'SYNTHETIC'})
    for kind in ('raw','ema'):
        prefix='' if kind=='raw' else 'ema_'
        torch.save({'model_state_dict':{'weight':torch.tensor([1. if kind=='raw' else 2.])}},run/('best_model.pt' if kind=='raw' else 'best_ema_model.pt'))
        write_json(run/('best_metrics.json' if kind=='raw' else 'best_ema_metrics.json'),{'epoch':4 if kind=='raw' else 5})
        for folder,n,patients in [('validation_analysis',859,228),('oof_analysis',3687,2211)]:
            rows=[{'Subject_id':f'p{i%patients}','Study_id':f's{i}','image_count':1+i%2,
                **{f'probability_{j}':(.8 if (i+j)%2 else .2) for j in range(1,11)},
                **{f'target_{j}':(i+j)%2 for j in range(1,11)}} for i in range(n)]
            csv_write(run/(prefix+folder)/'study_predictions.csv',rows)
            csv_write(tmp_path/'baseline/experiments/N17_F01'/(prefix+folder)/'study_predictions.csv',rows[::-1])
    finalize(run,task,plan,tmp_path/'baseline')
    data=json.loads((run/'metrics.json').read_text())
    assert data['variants']['raw']['best_epoch']==4 and data['variants']['ema']['best_epoch']==5
    assert data['test_auc'] is None
    assert data['variants']['ema']['oof_comparison']['valid_draws']==2
    assert digest(run/'best_model.pth')!=digest(run/'ema_best_model.pth')
    target=tmp_path/'relocated';shutil.copytree(run,target);verify_done(target)
    assert torch.load(target/'ema_best_model.pth',weights_only=True)['weight'].item()==2
