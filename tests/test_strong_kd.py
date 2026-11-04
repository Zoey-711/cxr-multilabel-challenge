"""CPU-only tests; no download, official data or GPU training in this suite."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from cxr.phase39_support import write_csv, write_json
from cxr.batch.artifacts import write_probabilities
from cxr.strong_kd import teachers, train
from cxr.strong_kd.loss import step_group
from cxr.strong_kd.protocol import RECIPES, TRAINING, digest, config, training_config
from cxr.strong_kd.runner import authorize
from cxr.strong_kd.runner import student_decision

torch.set_num_threads(2)


def fixture(root):
    pf=[dict(Subject_id=f'p{i}',fold=i//2+1) for i in range(10)]
    labels=[dict(Subject_id=r['Subject_id'],Study_id=f's{i}_{s}',fold=r['fold'],image_count=1+s,
                 **{f'Target_{j}':(s+j)%2 for j in range(1,11)}) for i,r in enumerate(pf) for s in range(2)]
    index=[dict(Subject_id=r['Subject_id'],Study_id=r['Study_id'],Dicom_id=f"{r['Study_id']}_{i}",fold=r['fold'])
           for r in labels for i in range(r['image_count'])]
    write_csv(root/'manifests/patient_folds.csv',pf)
    write_csv(root/'manifests/study_folds.csv',[{k:r[k] for k in ('Subject_id','Study_id','fold')} for r in labels])
    write_csv(root/'manifests/image_folds.csv',index)
    write_csv(root/'manifests/train_labels.csv',labels)
    write_csv(root/'manifests/val_labels.csv',[dict(labels[0],Subject_id='val_patient',Study_id='val_study')])
    folds={name:digest(root/'manifests'/name) for name in ('patient_folds.csv','study_folds.csv','image_folds.csv')}
    for b in ('D17','C17','S17','N17'):
        for f in range(1,6):
            name=f'{b}_F{f:02d}';folder=root/'experiments'/name;folder.mkdir(parents=True)
            cfg=dict(data=dict(train_subset=dict(fold=f,sha256=folds)))
            (folder/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
            write_csv(folder/'training_image_index.csv',[r for r in index if r['fold']!=f])
            held=[r for r in labels if r['fold']==f]
            prob='ema_oof_predictions.csv' if b=='N17' else 'oof_predictions.csv'
            weight='ema_best_model.pth' if b=='N17' else 'best_model.pth'
            p=np.array([[.1+.8*r[f'Target_{j}'] for j in range(1,11)] for r in held])
            if b!='N17':p[:]=.5
            write_probabilities(folder/prob,dict(keys=[(r['Subject_id'],r['Study_id']) for r in held],p=p),{})
            manifest=dict(model_id=name,fold_hashes=folds,
                training_image_index_sha256=digest(folder/'training_image_index.csv'),
                files={prob:dict(sha256=digest(folder/prob)),weight:dict(sha256='1'*64)})
            write_json(folder/'run_manifest.json',manifest)
            write_json(folder/'DONE.json',dict(state='complete',model_id=name,
                verified_sha256={**{n:digest(folder/n) for n in ('run_manifest.json','resolved_config.yaml',prob)},weight:'1'*64}))
    return dict(studies=20,patients=10,images=30,source_commit='test',
                files={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()})


def test_teacher_patient_provenance_and_complementarity(tmp_path):
    lock=fixture(tmp_path/'source')
    result=teachers.prepare(tmp_path/'source',tmp_path/'out',draws=40,lock=lock)
    assert result['state']=='PASS' and result['studies']==20
    assert result['student_oof']=='N/A' and result['outer_fold_student_training']=='BLOCKED'
    assert result['metrics']['SKD_COMPLEMENT']['macro_auc']>result['metrics']['SKD_CONTROL']['macro_auc']
    assert result['bootstrap']['valid']==40
    assert len(result['sources'])==20
    # Repeat does not change a frozen soft-target file.
    sha=digest(tmp_path/'out/teachers/SKD_COMPLEMENT.csv')
    teachers.prepare(tmp_path/'source',tmp_path/'out',draws=40,lock=lock)
    assert digest(tmp_path/'out/teachers/SKD_COMPLEMENT.csv')==sha


def test_tampered_teacher_is_rejected(tmp_path):
    lock=fixture(tmp_path)
    p=tmp_path/'experiments/N17_F01/ema_oof_predictions.csv'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA-256'):teachers.audit(tmp_path,lock)


def test_train_patient_leak_rejected_even_if_hashes_resealed(tmp_path):
    lock=fixture(tmp_path);folder=tmp_path/'experiments/D17_F01'
    path=folder/'training_image_index.csv';data=teachers.rows(path)
    data[0]['Subject_id']='p0';write_csv(path,data)
    # Prove semantic rejection, not merely the hash gate.
    lock['files'][path.relative_to(tmp_path).as_posix()]=digest(path)
    with pytest.raises(ValueError,match='isolation'):teachers.audit(tmp_path,lock)


def test_outer_fold_and_empty_budget_blocked(tmp_path):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/strong_kd_server.example.yaml').read_text())
    with pytest.raises(ValueError,match='authorize_training'):authorize(cfg)
    cfg['authorize_training']=True
    with pytest.raises(ValueError,match='budget'):authorize(cfg)
    cfg['mode']='outer_fold';p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='BLOCKED_OUTER_TEACHERS'):config(p)


def test_control_only_changes_teacher():
    a=training_config('SKD_CONTROL');b=training_config('SKD_COMPLEMENT')
    assert {k for k in a if a[k]!=b[k]}=={'experiment_id','teacher_formula'}
    assert all(sum(v.values())==1 for v in RECIPES.values())
    assert a['epochs']==4 and a['micro_batch']*a['accumulation']==16


def parts(model, scaler_device='cpu'):
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler=torch.amp.GradScaler(scaler_device,enabled=scaler_device=='cpu',init_scale=1024)
    return model,ema,opt,sch,scaler


def batch(n=4):
    g=torch.Generator().manual_seed(5)
    return dict(image=torch.rand(n,3,generator=g),target=torch.randint(0,2,(n,10),generator=g).float(),
        valid_mask=torch.ones(n,10,dtype=torch.bool),soft_target=torch.rand(n,10,generator=g),
        soft_weight=torch.tensor([1.,.5,.5,1.])[:n])


def test_accumulation_matches_combined_group_and_weights():
    torch.manual_seed(4);m=nn.Linear(3,10);p=parts(m);q=parts(copy.deepcopy(m));b=batch()
    a=step_group(*p[:2],[b],*p[2:],torch.device('cpu'),{})
    split=[{k:v[s] for k,v in b.items()} for s in (slice(0,2),slice(2,None))]
    c=step_group(*q[:2],split,*q[2:],torch.device('cpu'),{})
    assert a['total']==pytest.approx(c['total'])
    assert not a['skipped'] and int(p[1].n_averaged)==1
    for x,y in zip(p[0].parameters(),q[0].parameters()):torch.testing.assert_close(x,y)


def test_real_gradscaler_overflow_skips_optimizer_ema_and_scheduler():
    p=parts(nn.Linear(3,10));before=copy.deepcopy(p[0].state_dict())
    hook=p[0].weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    diag={};result=step_group(*p[:2],[batch()],*p[2:],torch.device('cpu'),diag);hook.remove()
    assert result['skipped'] and int(p[1].n_averaged)==0 and p[3].last_epoch==0
    assert p[4].get_scale()==512 and diag['scaler_before']==1024
    assert all(torch.equal(before[k],v) for k,v in p[0].state_dict().items())
    assert all(x.grad is None for x in p[0].parameters())


def test_bad_soft_target_blocks_step():
    p=parts(nn.Linear(3,10));b=batch();b['soft_target'][0,0]=float('nan')
    with pytest.raises(FloatingPointError):step_group(*p[:2],[b],*p[2:],torch.device('cpu'),{})
    assert int(p[1].n_averaged)==0


class ToyData(Dataset):
    def __init__(self):
        self.x=torch.rand(8,3,generator=torch.Generator().manual_seed(10))
    def __len__(self):return 8
    def __getitem__(self,i):
        y=torch.tensor([float((i+j)%2) for j in range(10)])
        return dict(image=self.x[i],target=y,valid_mask=torch.ones(10,dtype=torch.bool),
            soft_target=.1+.8*y,soft_weight=torch.tensor(1.),subject_id=f'p{i}',study_id=f's{i}',image_id=f'i{i}')


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Linear(3,4);self.classifier=nn.Linear(4,10)
    def forward(self,x):return self.classifier(torch.relu(self.features(x)))


@pytest.mark.parametrize('task', ['STUDENT_BASE','SKD_CONTROL','SKD_COMPLEMENT',
                                  'STUDENT_BASE_SEED2026','SKD_CONTROL_SEED2026','SKD_COMPLEMENT_SEED2026'])
def test_full_training_checkpoint_resume_export_without_retraining(tmp_path,monkeypatch,task):
    rows=[dict(Subject_id=f'p{i}',Study_id=f's{i}',Predict_class=';'.join(str(j+1) for j in range(10) if (i+j)%2)) for i in range(8)]
    monkeypatch.setattr(train,'data',lambda *_:(rows,rows,{},{}))
    def loaders(*_):
        return tuple(DataLoader(ToyData(),batch_size=4,generator=torch.Generator().manual_seed(s)) for s in (42,43))
    monkeypatch.setattr(train,'loaders',loaders)
    monkeypatch.setattr(train,'create_model',lambda *a,**k:ToyModel())
    cfg=dict(output_root=str(tmp_path),teacher_root=str(tmp_path),pretrained_root=str(tmp_path))
    (tmp_path/'teachers').mkdir()
    for name in RECIPES: (tmp_path/'teachers'/f'{name}.csv').write_text('synthetic')
    (tmp_path/'teacher_audit.json').write_text('{}')
    original=train.infer
    monkeypatch.setattr(train,'infer',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('export interrupted')))
    with pytest.raises(RuntimeError,match='export interrupted'):
        train.run_task(cfg,task,'test',torch.device('cpu'))
    resume=tmp_path/'experiments'/task/'last_training.pt';sha=digest(resume)
    monkeypatch.setattr(train,'infer',original)
    monkeypatch.setattr(train,'step_group',lambda *a,**k:pytest.fail('Completed training repeated'))
    receipt=train.run_task(cfg,task,'test',torch.device('cpu'))
    assert digest(resume)==sha and 'student_raw.pth' in receipt['sha256']
    assert len(teachers.rows(tmp_path/'experiments'/task/'train_history.csv'))==4
    metrics=json.loads((tmp_path/'experiments'/task/'metrics.json').read_text())
    assert metrics['student_oof']=='N/A' and set(metrics['val'])=={'raw','ema'}
    assert not (tmp_path/'experiments'/task/'oof_predictions.csv').exists()
    if task.startswith('STUDENT_BASE'):
        assert json.loads((tmp_path/'experiments'/task/'run_manifest.json').read_text())['teacher_sha256'] is None
        assert all(float(r['soft_loss'])==0 for r in teachers.rows(tmp_path/'experiments'/task/'train_history.csv'))
    with pytest.raises(ValueError,match='identity'):
        train.run_task(cfg,task,'changed',torch.device('cpu'))


def test_nonfinite_checkpoint_rejected():
    assert not train.finite_state({'optimizer':{'value':torch.tensor(float('inf'))}})
    assert train.finite_state({'a':[torch.tensor(1.),2,{}]} )


def test_image_mapping_checks_patient_study_and_image_together():
    good=[dict(Subject_id='p',Study_id='s',Dicom_id='i')]
    train.verify_image_mapping(good,good)
    with pytest.raises(ValueError,match='mapping'):
        train.verify_image_mapping([dict(good[0],Study_id='other')],good)
    with pytest.raises(ValueError,match='mapping'):
        train.verify_image_mapping(good+good,good)


def test_actual_student_is_single_efficientnet_with_ten_outputs():
    model,ema,*_=train.build(training_config('SKD_CONTROL'),torch.device('cpu'),1,pretrained=False)
    model.eval();ema.eval()
    with torch.inference_mode():
        output=model(torch.zeros(1,3,64,64))
    assert output.shape==(1,10) and torch.isfinite(output).all()
    assert not any('teacher' in name for name,_ in model.named_parameters())


def test_changed_pretrained_cache_blocks_before_any_cuda_model(tmp_path,monkeypatch):
    monkeypatch.setattr(train,'data',lambda *_:([],[],{},{}))
    monkeypatch.setattr(train,'loaders',lambda *_:([],[]))
    monkeypatch.setattr(train,'build',lambda *a,**k:pytest.fail('Must not initialize CUDA model'))
    weight=tmp_path/'cache/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth'
    weight.parent.mkdir(parents=True);weight.write_bytes(b'not-the-approved-weights')
    cfg=dict(output_root=str(tmp_path/'out'),pretrained_root=str(tmp_path/'cache'))
    with pytest.raises(ValueError,match='initialization hash'):
        train.run_task(cfg,'SKD_CONTROL','test',torch.device('cuda'))


def test_student_rebuild_gate_requires_paired_macro_ci_and_all_classes():
    primary=dict(bootstrap=dict(delta=.003,ci95=[.0002,.006]),
                 control=dict(per_class_auc=[.8]*10),
                 complement=dict(per_class_auc=[.803]*10))
    assert student_decision(primary)[0]=='CONSIDER_NESTED_TEACHER_PILOT'
    primary['bootstrap']['ci95'][0]=-.0001
    assert student_decision(primary)[0]=='STOP_TEACHER_REBUILD'
    primary['bootstrap']['ci95'][0]=.0002
    primary['complement']['per_class_auc'][0]=.789
    assert student_decision(primary)[0]=='STOP_TEACHER_REBUILD'
    primary['complement']['per_class_auc']=[.803]*9
    with pytest.raises(ValueError,match='Ten'):student_decision(primary)


def test_base_does_not_consume_soft_probabilities():
    p=parts(nn.Linear(3,10));q=parts(copy.deepcopy(p[0]))
    b=batch();bad=copy.deepcopy(b);bad['soft_target'][:]=float('nan')
    a=step_group(*p[:2],[b],*p[2:],torch.device('cpu'),{},hard_fraction=1.)
    c=step_group(*q[:2],[bad],*q[2:],torch.device('cpu'),{},hard_fraction=1.)
    assert a['soft']==c['soft']==0 and a['total']==pytest.approx(a['hard'])
    for x,y in zip(p[0].parameters(),q[0].parameters()):torch.testing.assert_close(x,y)


def test_fixed_triplets_and_explicit_replication_budget():
    from cxr.strong_kd.protocol import tasks
    assert len(tasks())==3 and len(tasks({'replicate_seed_enabled':True}))==6
    for t in tasks({'replicate_seed_enabled':True}):
        c=training_config(t['id'])
        assert c['seed']==(2026 if 'SEED2026' in t['id'] else 42)
        assert c['hard_fraction']==(1. if t['id'].startswith('STUDENT_BASE') else .5)
    cfg=dict(authorize_training=True,gpu_hours=38,preflight_gpu_hours=2,
             minimum_free_disk_gib=15,max_vram_gib=20,gpu_ids=[0],max_concurrent=1)
    authorize(cfg)
    with pytest.raises(ValueError,match='replicate_gpu_hours'):
        authorize(dict(cfg,replicate_seed_enabled=True))
    with pytest.raises(ValueError,match='Six tasks'):
        authorize(dict(cfg,replicate_seed_enabled=True,replicate_gpu_hours=36))
    authorize(dict(cfg,replicate_seed_enabled=True,replicate_gpu_hours=36,gpu_hours=74))


def test_loader_seed_changes_both_dataloader_streams(monkeypatch):
    seen=[]
    monkeypatch.setattr(train,'make_dataloader',lambda *a,**kw:seen.append(kw['seed']))
    train.loaders({'data_root':'.'},[],[],{}, {},2026)
    assert seen==[2026,2027]


def test_analysis_requires_all_triplet_results_and_writes_reports(tmp_path,monkeypatch):
    from cxr.strong_kd import analysis
    from cxr.strong_kd.protocol import tasks
    rr=[dict(Subject_id=f'p{i//2}',Study_id=f's{i}',image_count=1,**{f'Target_{j}':(i+j)%2 for j in range(1,11)}) for i in range(20)]
    write_csv(tmp_path/'source/manifests/val_labels.csv',rr)
    cfg=dict(output_root=str(tmp_path/'out'),teacher_root=str(tmp_path/'source'))
    original=analysis.paired_bootstrap
    monkeypatch.setattr(analysis,'paired_bootstrap',lambda *a,**kw:original(*a,draws=40,seed=5417))
    for name in ('SKD_CONTROL','SKD_COMPLEMENT'):
        folder=tmp_path/'out/experiments'/name
        for v in ('raw','ema'):
            write_probabilities(folder/f'{v}_val_predictions.csv',dict(keys=[(r['Subject_id'],r['Study_id']) for r in rr],p=np.full((20,10),.5)),{})
        write_json(folder/'DONE.json',dict(sha256={p.name:digest(p) for p in folder.iterdir()}))
    result=analysis.analyze(cfg)
    assert result['decision']=='WAIT_FOR_PAIRED_STUDENTS' and result['missing']==['STUDENT_BASE']
    folder=tmp_path/'out/experiments/STUDENT_BASE'
    for v in ('raw','ema'):
        write_probabilities(folder/f'{v}_val_predictions.csv',dict(keys=[(r['Subject_id'],r['Study_id']) for r in rr],p=np.full((20,10),.5)),{})
    write_json(folder/'DONE.json',dict(sha256={p.name:digest(p) for p in folder.iterdir()}))
    result=analysis.analyze(cfg)
    assert result['state']=='COMPLETE_DEVELOPMENT_ONLY' and len(result['comparisons'])==6
    assert result['decision']=='STOP_TEACHER_REBUILD'
    assert (tmp_path/'out/reports/final_decision.md').is_file()
    assert len(teachers.rows(tmp_path/'out/reports/student_per_class.csv'))==60
    result=analysis.analyze(dict(cfg,replicate_seed_enabled=True))
    assert len(result['missing'])==3 and result['decision']=='WAIT_FOR_PAIRED_STUDENTS'
