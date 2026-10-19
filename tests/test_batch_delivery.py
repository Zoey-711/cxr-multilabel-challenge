"""Small CPU-only delivery rehearsal. Does not initialize CUDA or use real training data."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cxr.batch.plan import write_json,digest
from cxr.batch.artifacts import write_probabilities,read_probabilities,label_keys,PROB_COLUMNS,record_failure
from cxr.batch.results import finalize_task,verify_frozen,read_table,scores
from cxr.batch.reporting import report
from cxr.batch.direct_delivery import verify_direct_folder
from cxr.batch.test_results import write_test


def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fixture_root(root):
    for d in ('manifests','experiments','validation','oof','analysis','deployment','reports','test_predictions','submissions'):(root/d).mkdir(parents=True)
    tasks=[{'id':f'{b}17_F{i:02d}','group':'A','enabled':True,'fold':i,'ema':False,'seed':42,'fold_hashes':{},'expected_test_studies':2} for b in 'DCS' for i in range(1,6)]
    tasks += [{'id':f'N17_F{i:02d}','group':'N','enabled':True,'fold':i,'ema':True,'seed':42,'fold_hashes':{},'expected_test_studies':2} for i in range(1,6)]
    tasks += [{'id':b+'_F01','group':'P','enabled':True,'fold':1,'ema':False,'seed':42,'fold_hashes':{},'expected_test_studies':2} for b in ('CS17','NM17')]
    plan={'tasks':tasks,'bootstrap_draws':10,'fold_hashes':{},'data_sha256':'synthetic','expected_test_studies':2,
          'fixed_fusions':{'E17':{'D5':.25,'C5':.25,'S5':.5}},
          'conditional_extension':{'reserved_slots':[{'id':name,'group':name[0]} for name in
              ('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02')]},
          'deployment_candidates':{'E17_F01':{'D17_F01':.25,'C17_F01':.25,'S17_F01':.5}}}
    (root/'manifests/batch_plan.yaml').write_text(yaml.safe_dump(plan),encoding='utf-8')
    fields=['Subject_id','Study_id','image_count','fold']+[f'Target_{i}' for i in range(1,11)]
    labels=[]
    for patient in range(10):
        for j in range(2):
            labels.append({'Subject_id':f'{patient:04d}','Study_id':f'{patient*2+j:06d}', 'image_count':1+j,'fold':patient//2+1,
                           **{f'Target_{i}':j for i in range(1,11)}})
    for name in ('train_labels.csv','val_labels.csv'):write_csv(root/'manifests'/name,fields,labels)
    write_json(root/'manifests/labels_sha256.json',{n:digest(root/'manifests'/n) for n in ('train_labels.csv','val_labels.csv')})
    write_json(root/'manifests/environment.json',{'git_commit':None,'data_sha256':'synthetic',
               'fold_hashes':{},'code_sha256':{}})
    image_rows=[{**{k:r[k] for k in ('Subject_id','Study_id','fold')},'Dicom_id':'image_'+r['Study_id']} for r in labels]
    write_csv(root/'manifests/image_folds.csv',list(image_rows[0]),image_rows)
    table={'keys':[(r['Subject_id'],r['Study_id']) for r in labels], 'p':np.full((20,10),.5),
           'y':np.asarray([[r[f'Target_{i}'] for i in range(1,11)] for r in labels]),'counts':np.asarray([r['image_count'] for r in labels])}
    write_probabilities(root/'validation/E07_val_probabilities.csv',table,{'model_id':'synthetic_reference'})
    return tasks,table


def train_synthetic(run,task,table):
    run.mkdir();torch.manual_seed(42)
    model=torch.nn.Linear(2,10)
    x=torch.tensor([[float(i%2),1.] for i in range(20)])
    y=torch.tensor(table['y'],dtype=torch.float32)
    mask=torch.tensor([int(s)//2+1 != task['fold'] for s,_ in table['keys']])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.02)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    history=[]
    for epoch in (1,2):
        optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[mask]),y[mask]);loss.backward();optimizer.step();scheduler.step()
        p=model(x).sigmoid().detach().numpy();m=scores({**table,'p':p})
        history.append({'epoch':epoch,'train_loss':loss.item(),'val':{'macro_auc':m['macro_auc'],'macro_f1':m['macro_f1_at_0_5']},
                        'learning_rates':[optimizer.param_groups[0]['lr']],'elapsed_seconds':.01,'best_epoch':2,'stale':0,
                        'optimizer_steps':1,'amp_skipped_steps':0,'gpu_peak_allocated_mib':0,'gpu_peak_reserved_mib':0})
        if epoch==1:
            # Real optimizer/scheduler/scaler/model serialization and reload in the rehearsal.
            state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                   'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'stale':0,'python_random_state':__import__('random').getstate()}
            torch.save(state,run/'last_training.pt')
            restored=torch.load(run/'last_training.pt',weights_only=True)
            model.load_state_dict(restored['model_state_dict']);optimizer.load_state_dict(restored['optimizer_state_dict']);scheduler.load_state_dict(restored['scheduler_state_dict']);scaler.load_state_dict(restored['scaler_state_dict'])
    torch.save({'model_state_dict':model.state_dict()},run/'best_model.pt')
    write_json(run/'best_metrics.json',history[-1])
    write_json(run/'status.json',{'state':'complete','best_epoch':2,'stop_epoch':2,'stale':0})
    write_json(run/'environment.json',{'python':'synthetic test','torch':torch.__version__,'gpu':'none (unit test)'})
    write_json(run/'pretrained_source.json',{'source':'synthetic fixture; not a competition model'})
    (run/'config.yaml').write_text('model: {name: synthetic_linear, num_classes: 10}\n',encoding='utf-8')
    (run/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in history)+'\n',encoding='utf-8')
    (run/'train_stdout.log').write_text('synthetic training completed\n');(run/'train_stderr.log').write_text('')
    for folder,indices in [('validation_analysis',list(range(20))),('oof_analysis',[i for i,(s,_) in enumerate(table['keys']) if int(s)//2+1==task['fold']])]:
        rows=[]
        for i in indices:
            s,st=table['keys'][i]
            rows.append({'Subject_id':s,'Study_id':st,'image_count':int(table['counts'][i]),
                         **{f'probability_{j}':float(p[i,j-1]) for j in range(1,11)},
                         **{f'target_{j}':int(table['y'][i,j-1]) for j in range(1,11)}})
        write_csv(run/folder/'study_predictions.csv',list(rows[0]),rows)
    if task['ema']:
        # Export pipeline fixture; EMA arithmetic is independently tested on real CPU BatchNorm.
        shutil.copy2(run/'best_model.pt',run/'best_ema_model.pt')
        shutil.copy2(run/'best_metrics.json',run/'best_ema_metrics.json')
        for folder in ('validation_analysis','oof_analysis'):
            shutil.copytree(run/folder,run/('ema_'+folder))
    for kind in (['raw','ema'] if task['ema'] else ['raw']):
        source=run/('best_model.pt' if kind=='raw' else 'best_ema_model.pt')
        target=run/('test_predictions.csv' if kind=='raw' else 'ema_test_predictions.csv')
        write_test(target,{'keys':[('subject_00001','study_000001'),('subject_00002','study_000002')],
                           'p':np.full((2,10),.4)},
                   {'checkpoint_sha256':digest(source),'checkpoint_kind':kind,'no_test_labels':True})
    write_json(run/'latency.json',{'device':'synthetic CPU','mean_ms':1,'scope':'unit test'})
    finalize_task(run,task,expected_val=(20,10));verify_frozen(run)
    from cxr.batch.submission_reporting import emit_single_submission
    emit_single_submission(run,expected_studies=2)
    return model


def test_end_to_end_train_resume_export_direct_folder_recompute(tmp_path):
    root=tmp_path/'outputs/phase17_batch';tasks,table=fixture_root(root)
    expected={}
    for task in tasks:
        model=train_synthetic(root/'experiments'/task['id'],task,table)
        expected[task['id']]={k:v.clone() for k,v in model.state_dict().items()}
    before=report(root,draws=10)
    assert before['missing_folds']=={'D17':[],'C17':[],'S17':[],'N17':[]}
    assert (root/'test_predictions/D17_test_5fold.csv').is_file()
    assert (root/'test_predictions/N17_EMA_test_5fold.csv').is_file()
    assert (root/'test_predictions/E17_test_probabilities.csv').is_file()
    with (root/'reports/experiment_summary.csv').open(encoding='utf-8',newline='') as stream:
        summary=list(csv.DictReader(stream))
    summary_by_id={row['model_id']:row for row in summary}
    assert {'D5','C5','S5','N5','E17'} <= set(summary_by_id)
    assert {'G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02'} <= set(summary_by_id)
    assert all(summary_by_id[name]['state']=='PENDING' for name in
               ('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02'))
    assert summary_by_id['D17_F01']['val_macro_auc']
    assert summary_by_id['D17_F01']['oof_macro_auc']
    assert summary_by_id['D5']['oof_auc_class_1']
    assert summary_by_id['E17']['val_auc_class_10']
    assert all(row['test_auc']=='N/A: hidden Test labels unavailable' for row in summary)
    assert summary_by_id['D17_F01']['test_prediction_status']=='probabilities_saved'
    from cxr.batch.test_results import validate_submission
    assert validate_submission(root/'submissions/ensembles/E17_submission.csv',expected_studies=2)['rows']==20
    assert all((root/'experiments'/t['id']/'model_size.json').is_file() for t in tasks)
    assert not list(root.rglob('*.tar.gz'))
    assert not (root/'oof/CS17_oof_all.csv').exists()
    assert (root/'analysis/backbone_pilot_comparison.csv').exists()
    assert all(r['status']!='PASS' for r in before['deployment_compliance'])
    manifest=verify_direct_folder(root)
    assert manifest['model_count']==22 and manifest['no_archives'] is True
    destination=tmp_path/'independent'
    shutil.copytree(root,destination)
    recovered=destination
    verify_direct_folder(recovered)
    after=report(recovered,draws=10)
    assert before['official_val']==after['official_val']
    assert before['bootstrap_val']==after['bootstrap_val']
    assert before['full_oof']==after['full_oof']
    for task in tasks:
        restored=torch.nn.Linear(2,10)
        restored.load_state_dict(torch.load(recovered/'experiments'/task['id']/'best_model.pth',weights_only=True))
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected[task['id']].items())
    with (recovered/'reports/experiment_summary.json').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError):verify_direct_folder(recovered)


def test_incomplete_failed_tasks_explicit_and_no_false_oof(tmp_path):
    root=tmp_path/'batch';tasks,table=fixture_root(root)
    train_synthetic(root/'experiments'/tasks[0]['id'],tasks[0],table)
    record_failure(root,tasks[1],{'reason':'CUDA OOM','retryable':False,'gpu_hours':.1})
    result=report(root,draws=5)
    assert result['tasks'][1]['state']=='failed'
    assert result['tasks'][2]['val_auc'] is None
    assert result['missing_folds']['D17']
    assert not (root/'oof/D17_oof_all.csv').exists()
    assert not (root/'test_predictions/D17_test_5fold.csv').exists()
    assert json.loads((root/'test_predictions/D17_test_5fold.status.json').read_text())['state']=='INCOMPLETE'
    assert not (root/'experiments'/tasks[1]['id']/'DONE.json').exists()
    from scripts.validate_batch_artifacts import main as validate_artifacts
    import sys
    old=sys.argv
    try:
        sys.argv=['validate_batch_artifacts.py','--config',str(root/'manifests/batch_plan.yaml'),'--root',str(root)]
        validate_artifacts()
    finally:
        sys.argv=old
    audit=json.loads((root/'reports/artifact_validation.json').read_text())
    assert audit[0]=={'id':tasks[0]['id'],'state':'verified'}
    assert audit[1]=={'id':tasks[1]['id'],'state':'failed'}
    partial_manifest=verify_direct_folder(root)
    assert partial_manifest['completed_tasks']==[tasks[0]['id']]
    assert len(partial_manifest['missing_tasks'])==21
    assert (root/'experiments'/tasks[0]['id']/'best_model.pth').is_file()
    assert not (root/'experiments'/tasks[1]['id']/'DONE.json').exists()


def test_portable_probability_columns_and_label_id_join(tmp_path):
    root=tmp_path/'batch';tasks,t=fixture_root(root)
    path=root/'validation/E07_val_probabilities.csv'
    with path.open(newline='') as f:rows=list(csv.reader(f))
    assert rows[0]==PROB_COLUMNS
    write_csv(path,PROB_COLUMNS,[dict(zip(PROB_COLUMNS,r)) for r in rows[:0:-1]])
    restored=read_probabilities(path,root/'manifests/val_labels.csv')
    assert restored['keys']==t['keys'][::-1]
    assert np.array_equal(restored['y'],t['y'][::-1])
