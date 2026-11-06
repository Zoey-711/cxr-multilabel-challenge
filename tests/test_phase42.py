"""CPU-only architecture, statistical selection, checkpoint and queue regression."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
from cxr.phase42.models import IDS,build_model,parameter_groups
from cxr.phase42.protocol import load_plan,tasks,resolved_config,server_config,digest,verify_receipt
from cxr.phase42.analysis import paired_bootstrap,weighted_auc,select

torch.set_num_threads(2)


@pytest.mark.parametrize('variant', ('B01', 'C03'))
def test_gem_heads_backward_with_training_dropout(variant):
    model = build_model(variant).train()
    assert not model.dropout.inplace
    logits = model(torch.rand(2, 3, 64, 64))
    nn.functional.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert model.pool.p.grad is not None
    assert torch.isfinite(model.pool.p.grad).all()


@pytest.mark.parametrize('variant',IDS)
def test_forward_backward_export_ema(variant,tmp_path):
    torch.manual_seed(42)
    model=build_model(variant).eval()
    size=384 if variant=='F04' else 64
    image=torch.rand(2,3,size,size)
    logits=model(image)
    assert logits.shape==(2,10) and torch.isfinite(logits).all()
    nn.functional.binary_cross_entropy_with_logits(logits,torch.rand_like(logits)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    back,head=parameter_groups(model)
    assert set(map(id,back)).isdisjoint(set(map(id,head)))
    assert set(map(id,back+head))==set(map(id,model.parameters()))
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
    ema.update_parameters(model)
    assert int(ema.n_averaged)==1
    path=tmp_path/'model.pth';torch.save(model.state_dict(),path)
    restored=build_model(variant).eval();restored.load_state_dict(torch.load(path,weights_only=True))
    with torch.no_grad():assert torch.equal(model(image),restored(image))


def test_initial_backbones_and_rng_match():
    torch.manual_seed(42);base=build_model('BASE');reference=torch.get_rng_state()
    for variant in IDS:
        torch.manual_seed(42);model=build_model(variant)
        backbone=model.backbone if hasattr(model,'backbone') else model
        assert all(torch.equal(v,backbone.features.state_dict()[k]) for k,v in base.features.state_dict().items())
        assert torch.equal(torch.get_rng_state(),reference)


def test_plan_counts_and_confirmation_never_repeats_pilot():
    plan=load_plan();assert len(tasks(plan))==19
    assert all(t['fold']==2 for t in tasks(plan))
    expanded=tasks(plan,['A01','B01','C01','D01'])
    assert len(expanded)==16 and all(t['fold']!=2 for t in expanded)
    assert {t['id'] for t in expanded}.isdisjoint(t['id'] for t in tasks(plan))
    for t in tasks(plan):
        cfg=resolved_config(plan,t)
        assert cfg['micro_batch']==4 and cfg['accumulation']==4 and cfg['seed']==42
    with pytest.raises(ValueError):tasks(plan,['A01']*2)


def test_smoothing_does_not_change_mask_or_original_targets(monkeypatch):
    from cxr.phase42 import methods
    seen=[]
    monkeypatch.setattr(methods,'original_step',lambda cfg,m,e,micro,*args:seen.extend(micro))
    y=torch.tensor([[0.,1.]])
    batch=dict(target=y,valid_mask=torch.tensor([[False,True]]))
    methods.step_group({'label_smoothing':.05},None,None,[batch],None,None,None,None)
    assert torch.equal(y,torch.tensor([[0.,1.]]))
    assert torch.allclose(seen[0]['target'],torch.tensor([[.025,.975]]))
    assert torch.equal(seen[0]['valid_mask'],batch['valid_mask'])


def test_phase41r_amp_diagnostics_reach_phase42_step(monkeypatch):
    from cxr.phase42 import methods
    recorded = []
    def fake_step(*args):
        recorded.append(args)
        args[-1]['optimizer_skipped'] = True
        return 1.0, True
    monkeypatch.setattr(methods, 'original_step', fake_step)
    diagnostic = {}
    value, skipped = methods.step_group({'label_smoothing': 0.}, None, None,
                                        [], None, None, None, None, diagnostic)
    assert (value, skipped) == (1.0, True)
    assert recorded[0][-1] is diagnostic
    assert diagnostic['optimizer_skipped'] is True


def test_cluster_bootstrap_matches_explicit_duplication():
    from cxr.engine.metrics import binary_auc
    y=np.array([0,1,1,0]);p=np.array([.3,.3,.9,.8]);w=np.array([2,1,3,1])
    assert weighted_auc(y,p,w)==pytest.approx(binary_auc(np.repeat(y,w),np.repeat(p,w)))
    keys=[(str(i//2),str(i)) for i in range(40)]
    truth={k:[i%2]*10 for i,k in enumerate(keys)}
    base={k:np.full(10,.5) for k in keys};better={k:np.full(10,.9 if i%2 else .1) for i,k in enumerate(keys)}
    a=paired_bootstrap(truth,base,better,keys,draws=50)
    assert a==paired_bootstrap(truth,base,better,keys,draws=50)
    assert a['delta']==.5 and a['patients']==20 and a['valid']==50


def test_frozen_selection_holm_and_mechanism_limit():
    plan=load_plan()
    rows=[dict(candidate=c['id'],mechanism=c['mechanism'],state='DONE',delta=.01,ci_lower=.002,p=.00001,
               worst_class_delta=-.001,mac_ratio=1.01,weight_bytes=90000000) for c in plan['candidates']]
    selected=select(rows,plan)
    assert len(selected)==4
    assert len({r['mechanism'] for r in rows if r['candidate'] in selected})==4
    failed=copy.deepcopy(rows);failed[0].update(state='FAILED',delta=None,p=1.)
    assert failed[0]['candidate'] not in select(failed,plan)
    for r in rows:r['p']=.1
    assert select(rows,plan)==[]


def test_budget_and_output_protection(tmp_path,monkeypatch):
    import yaml
    from cxr.phase42.protocol import ROOT
    cfg=yaml.safe_load((ROOT/'configs/phase42_server.example.yaml').read_text())
    monkeypatch.setattr('cxr.phase42.protocol.ROOT',tmp_path/'separate_code')
    for key in ('data_root','pretrained_root','protocol_root','phase41_results_root','output_root'):
        cfg[key]=str(tmp_path/key)
    path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='budget'):server_config(path)
    cfg.update(gpu_hours=100,pilot_gpu_hours=90,preflight_gpu_hours=10)
    path.write_text(yaml.safe_dump(cfg));assert server_config(path)['max_concurrent']==8
    cfg['confirmation']=dict(enabled=True,gpu_hours=50);path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='exceed'):server_config(path)


def test_queue_failure_isolation_resume_no_double_charge(tmp_path):
    from cxr.phase42.queue import run
    from cxr.phase39_support import write_json
    items=tasks(load_plan())[:3];calls=[]
    def worker(t,root,config,gpu,identity,attempt):
        calls.append(t['id']);folder=root/'experiments'/t['id'];folder.mkdir(parents=True,exist_ok=True)
        if t==items[0]:return dict(state='FAILED',reason='synthetic OOM',retryable=False,gpu_hours=.01)
        (folder/'metrics.json').write_text('{}')
        write_json(folder/'DONE.json',dict(identity_sha256=identity,sha256={'metrics.json':digest(folder/'metrics.json')}))
        return dict(state='DONE',gpu_hours=.01)
    cfg=dict(output_root=str(tmp_path),pilot_gpu_hours=150.,gpu_ids=[0,1],max_concurrent=2,upload=dict(enabled=False))
    a=run(items,cfg,tmp_path/'config.yaml','test','pilot',executor=worker,poll=.001)
    assert [r['state'] for r in a['tasks'].values()]==['FAILED','DONE','DONE']
    b=run(items,cfg,tmp_path/'config.yaml','test','pilot',executor=worker,poll=.001)
    assert len(calls)==3 and a['gpu_hours_used']==b['gpu_hours_used']


def test_insufficient_budget_does_not_launch(tmp_path):
    from cxr.phase42.queue import run
    cfg=dict(output_root=str(tmp_path),pilot_gpu_hours=1.,gpu_ids=[0],max_concurrent=1,upload=dict(enabled=False))
    def forbidden(*args):raise AssertionError('Worker should never launch')
    result=run(tasks(load_plan())[:1],cfg,tmp_path/'cfg','id','pilot',executor=forbidden,poll=.001)
    assert next(iter(result['tasks'].values()))['state']=='PENDING'


def test_receipt_detects_tamper_and_path_escape(tmp_path):
    (tmp_path/'metrics.json').write_text('{}')
    marker=tmp_path/'DONE.json';marker.write_text(json.dumps(dict(sha256={'metrics.json':digest(tmp_path/'metrics.json')})))
    verify_receipt(tmp_path)
    (tmp_path/'metrics.json').write_text('changed')
    with pytest.raises(ValueError,match='hash'):verify_receipt(tmp_path)
    marker.write_text(json.dumps(dict(sha256={'../bad':'x'})))
    with pytest.raises(ValueError,match='Unsafe'):verify_receipt(tmp_path)


def test_real_smoke_cpu_exercises_restore_and_val(monkeypatch,tmp_path):
    from cxr.phase42 import train
    from torch.utils.data import DataLoader
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.features=nn.Sequential(nn.Conv2d(3,4,1),nn.AdaptiveAvgPool2d(1));self.classifier=nn.Linear(4,10)
        def forward(self,x):return self.classifier(self.features(x).flatten(1))
    def loader():
        data=[dict(image=torch.rand(3,8,8),target=torch.rand(10),valid_mask=torch.ones(10,dtype=torch.bool)) for _ in range(4)]
        return DataLoader(data,batch_size=2,generator=torch.Generator().manual_seed(42))
    monkeypatch.setattr(train,'fold_rows',lambda *a:([{}]*16,[{}]*4))
    monkeypatch.setattr(train,'loaders',lambda *a:(loader(),loader()))
    monkeypatch.setattr(train,'build',lambda *a,**kw:Tiny())
    monkeypatch.setattr(train,'read_split_rows',lambda *a:[{},{}])
    monkeypatch.setattr(train,'make_dataloader',lambda *a,**kw:loader())
    cfg=dict(pretrained_root=str(tmp_path),data_root=str(tmp_path),protocol_root=str(tmp_path))
    result=train.real_smoke(load_plan(),tasks(load_plan())[0],cfg,tmp_path/'probe',device='cpu')
    assert result['state']=='PASS' and result['resume']=='PASS' and result['oof_val_forward']=='PASS'
    assert np.isfinite(result['resumed_loss'])


def test_frozen_patient_selection_rejects_crossfold_study(tmp_path):
    from cxr.data.patient_folds import select_fold_rows
    from cxr.data.archive import image_member
    from cxr.phase39_support import write_csv
    rows=[dict(Subject_id=f'subject_{i:05d}',Study_id=f'study_{i:05d}',Dicom_id=f'pic_{i:06d}',Predict_class='1;2') for i in range(10)]
    patients=[dict(Subject_id=r['Subject_id'],fold=i%5+1) for i,r in enumerate(rows)]
    studies=[dict(Subject_id=r['Subject_id'],Study_id=r['Study_id'],fold=i%5+1,Predict_class=r['Predict_class']) for i,r in enumerate(rows)]
    images=[dict(**r,fold=i%5+1,archive_member=image_member(r,'train')) for i,r in enumerate(rows)]
    for name,entries in [('patient_folds.csv',patients),('study_folds.csv',studies),('image_folds.csv',images)]:write_csv(tmp_path/name,entries)
    cfg=dict(directory=str(tmp_path),fold=2,sha256={p.name:digest(p) for p in tmp_path.glob('*.csv')},
             expected_counts=dict(train=dict(patients=8,studies=8,images=8),oof=dict(patients=2,studies=2,images=2)))
    train=select_fold_rows(rows,cfg,role='train');oof=select_fold_rows(rows,cfg,role='oof')
    assert not {r['Subject_id'] for r in train}&{r['Subject_id'] for r in oof}
    studies[0]['fold']=2;write_csv(tmp_path/'study_folds.csv',studies)
    cfg['sha256']['study_folds.csv']=digest(tmp_path/'study_folds.csv')
    with pytest.raises(ValueError,match='mismatch'):select_fold_rows(rows,cfg,role='oof')


def test_training_export_interruption_resume_end_to_end_cpu(monkeypatch,tmp_path):
    from cxr.phase42 import train
    from cxr.phase39_support import write_json
    from torch.utils.data import DataLoader
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.features=nn.Sequential(nn.Conv2d(3,4,1),nn.AdaptiveAvgPool2d(1));self.classifier=nn.Linear(4,10)
        def forward(self,x):return self.classifier(self.features(x).flatten(1))
    rows=[dict(Subject_id=f'subject_{i:05d}',Study_id=f'study_{i:05d}',Dicom_id=f'pic_{i:06d}',
               Predict_class='1;3;5;7;9' if i%2 else '2;4;6;8;10') for i in range(20)]
    def loader():
        data=[dict(image=torch.rand(3,8,8),target=torch.tensor([(i+j)%2 for j in range(10)],dtype=torch.float32),
                   valid_mask=torch.ones(10,dtype=torch.bool),subject_id=r['Subject_id'],study_id=r['Study_id'],image_id=r['Dicom_id']) for i,r in enumerate(rows)]
        return DataLoader(data,batch_size=4,generator=torch.Generator().manual_seed(42))
    plan=load_plan();item=tasks(plan)[0];conf=resolved_config(plan,item);conf['epochs']=1
    monkeypatch.setattr(train,'resolved_config',lambda *a:conf)
    monkeypatch.setattr(train,'fold_rows',lambda *a:(rows,rows))
    monkeypatch.setattr(train,'loaders',lambda *a:(loader(),loader()))
    monkeypatch.setattr(train,'build',lambda *a,**kw:Tiny())
    monkeypatch.setattr(train,'build_model',lambda *a,**kw:Tiny())
    monkeypatch.setattr(train,'read_split_rows',lambda *a:rows)
    monkeypatch.setattr(train,'make_dataloader',lambda *a,**kw:loader())
    monkeypatch.setattr('cxr.phase42.benchmark.benchmark',lambda model,dataset,run,cfg:write_json(run/'inference_benchmark.json',{'status':'CPU_TEST_ONLY'}))
    cfg=dict(output_root=str(tmp_path),pretrained_root=str(tmp_path),data_root=str(tmp_path),protocol_root=str(tmp_path),max_vram_gib=20)
    original=train.infer
    def interrupt(model,loader,targets,output,device):
        if output is not None:raise RuntimeError('synthetic inference interruption')
        return original(model,loader,targets,output,device)
    monkeypatch.setattr(train,'infer',interrupt)
    with pytest.raises(RuntimeError,match='interruption'):train.run_task(plan,item,cfg,'identity',device='cpu')
    folder=tmp_path/'experiments'/item['id']
    state=torch.load(folder/'last_training.pt',weights_only=True)
    assert state['training_complete'] and len(state['history'])==1 and state['optimizer']['state']
    best_hash=digest(folder/'ema_best.pth')
    monkeypatch.setattr(train,'infer',original)
    result=train.run_task(plan,item,cfg,'identity',device='cpu')
    assert result['state']=='DONE' and digest(folder/'ema_best.pth')==best_hash
    verify_receipt(folder)
    assert train.run_task(plan,item,cfg,'identity',device='cpu')['reused']
    for file in ('oof_predictions.csv','val_predictions.csv','metrics.json','resolved_config.yaml','per_class_metrics.csv'):
        assert (folder/file).is_file()
    assert not (folder/'test_predictions.csv').exists()


def test_remote_preflight_and_upload_fail_closed(tmp_path):
    from cxr.phase42.preflight import prepare
    from cxr.phase42.upload import upload
    import platform
    if platform.system()!='Linux':
        with pytest.raises(RuntimeError,match='Linux'):prepare(load_plan(),{})
    with pytest.raises(ValueError,match='authorized'):upload(tmp_path,dict(enabled=False))


def test_preflight_baseline_timing_and_cached_integrity(tmp_path,monkeypatch):
    from cxr.phase42 import preflight as remote
    from cxr.phase39_support import write_json
    plan=load_plan();calls=[]
    def probe(item,gpu,cfg,config,folder):
        calls.append(item.get('mode','smoke'))
        folder.mkdir(parents=True,exist_ok=True)
        hashes={}
        if item.get('mode')=='baseline':
            write_json(folder/'inference_benchmark.json',dict(mean_ms=100.,gpu='MOCK',status='T4_UNVERIFIED'))
            hashes={'inference_benchmark.json':digest(folder/'inference_benchmark.json')}
        write_json(folder/'receipt.json',dict(state='PASS',sha256=hashes))
        return dict(task=item['id'],hours=.001,path=(folder/'receipt.json').relative_to(tmp_path).as_posix(),sha256=digest(folder/'receipt.json'))
    monkeypatch.setattr(remote,'run_probe',probe)
    cfg=dict(output_root=str(tmp_path),gpu_ids=list(range(8)),max_concurrent=8,preflight_gpu_hours=10)
    result=remote.preflight(plan,cfg,tmp_path/'config',{'identity':'mock'})
    assert result['state']=='PASS' and calls.count('smoke')==19 and calls.count('baseline')==1
    assert remote.preflight(plan,cfg,tmp_path/'config',{'identity':'mock'})==result
    assert len(calls)==20
    (tmp_path/'reports/baseline_reference/inference_benchmark.json').write_text('{}')
    with pytest.raises(ValueError,match='Baseline timing changed'):
        remote.preflight(plan,cfg,tmp_path/'config',{'identity':'mock'})


def test_complete_fivefold_export_and_excluded_pilot(tmp_path,monkeypatch):
    import yaml
    from cxr.phase42 import analysis
    from cxr.phase42.protocol import task
    from cxr.phase39_support import write_json,write_csv
    plan=load_plan()
    keys=[(f'subject_{i%11055:05d}',f'study_{i:05d}') for i in range(18513)]
    folds={k:int(k[0].split('_')[1])%5+1 for k in keys}
    truth={k:[(i+j)%2 for j in range(10)] for i,k in enumerate(keys)}
    vk=[(f'subject_val_{i%228}',f'val_{i}') for i in range(859)]
    vt={k:[(i+j)%2 for j in range(10)] for i,k in enumerate(vk)}
    monkeypatch.setattr(analysis,'labels_and_folds',lambda *a:(truth,vt,folds))
    def rows(selected,labels):
        return [dict(Subject_id=k[0],Study_id=k[1],**{f'Prob_{j+1}':.8 if labels[k][j] else .2 for j in range(10)}) for k in selected]
    for fold in range(1,6):
        folder=tmp_path/'experiments'/f'P42_A01_F{fold:02d}';folder.mkdir(parents=True)
        write_csv(folder/'oof_predictions.csv',rows([k for k in keys if folds[k]==fold],truth))
        write_csv(folder/'val_predictions.csv',rows(vk,vt))
        (folder/'resolved_config.yaml').write_text(yaml.safe_dump(resolved_config(plan,task(plan,'A01',fold))))
        write_json(folder/'run_manifest.json',dict(variant='A01',fold=fold,fold_sha256=plan['fold_sha256'],
            official_val_used_for_training=False,test_labels_used=False,config_sha256=digest(folder/'resolved_config.yaml')))
        write_json(folder/'model_size.json',dict(inference_weights_and_config_bytes=81000000))
        write_json(folder/'DONE.json',dict(sha256={p.name:digest(p) for p in folder.iterdir()}))
    monkeypatch.setattr(analysis,'verify_baseline',lambda p,c,f,**kw:(tmp_path/'experiments'/f'P42_A01_F{f:02d}',{}))
    calls=[]
    def fake_boot(truth,base,candidate,keys,*args,**kw):
        calls.append(set(keys));return dict(delta=0.,ci95=[0.,0.])
    monkeypatch.setattr(analysis,'paired_bootstrap',fake_boot)
    result=analysis.confirmation(tmp_path,plan,{},['A01'])
    assert result['candidates']['A01']['full_oof']['studies']==18513
    assert result['candidates']['A01']['full_oof']['patients']==11055
    assert calls[1]=={k for k in keys if folds[k]!=2}
    assert result['candidates']['A01']['state']=='COMPLETED_NO_GAIN'
    assert (tmp_path/'oof/A01_fivefold.csv').is_file()
    assert json.loads((tmp_path/'reports/confirmation_analysis.json').read_text())['third_stage_recommendations']==[]
