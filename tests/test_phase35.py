"""CPU only: exact continuation, failure isolation, reusable outputs and full OOF."""
import copy,csv,importlib.util,json,threading,time
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from cxr.batch.plan import ROOT,digest,write_json
from cxr.phase35.protocol import load_plan,server_config,reuse_fold1
from cxr.phase35.results import csv_write,finalize,verify_done
from cxr.batch.artifacts import write_probabilities
from cxr.batch.test_results import read_test_probabilities

@pytest.mark.parametrize('task',load_plan()['tasks'],ids=lambda t:t['id'])
def test_exact_eight_continuations_and_cpu_loss(task):
    from cxr.phase32.methods import loss_elements,study_positive_weights
    plan=load_plan();cfg=yaml.safe_load((ROOT/task['config']).read_text())
    base=yaml.safe_load((ROOT/plan['reused_fold1'][task['variant']]['config']).read_text())
    assert cfg['training']==base['training'] and cfg['experiment']['seed']==42
    assert task['fold'] in (2,3,4,5) and cfg['training']['epochs']==30
    assert cfg['training']['early_stopping_patience']==6 and cfg['training']['ema']['decay']==.999
    assert cfg['data']['input_size']==(384 if task['variant']=='POS' else 512)
    settings=copy.deepcopy(cfg['training'])
    if task['variant']=='POS':
        rows=[{'Subject_id':'a','Study_id':'a','Predict_class':'1;2;3;4;5'},
              {'Subject_id':'b','Study_id':'b','Predict_class':'6;7;8;9;10'}]
        w=study_positive_weights(rows)
        assert w==study_positive_weights(rows+[rows[0]]*4)
        settings['_resolved_pos_weight']=w
    logits=torch.randn(4,10,requires_grad=True);labels=torch.randint(0,2,(4,10)).float()
    loss=loss_elements(logits,labels,settings).mean();loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

def test_original_seals_and_server_training_snapshot_unchanged():
    from cxr.phase32.protocol import load_plan as p32
    from cxr.e08_tune.protocol import load_plan as et
    assert len(p32()['tasks'])==8 and len(et()['tasks'])==25
    import subprocess
    original=subprocess.check_output(['git','show','01fa68bdac65ce56d04a1fc99e8f5996ab09aa20:src/cxr/phase32/train.py'],cwd=ROOT)
    assert (ROOT/'src/cxr/phase35/train.py').read_bytes()==original

def test_budget_and_local_training_gate(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/phase35_server.example.yaml').read_text())
    for k in ('data_root','pretrained_root','protocol_root','baseline_results_root','phase32_results_root'):cfg[k]=str(tmp_path/k)
    cfg['output_root']=str(tmp_path/'phase35_batch');p=tmp_path/'server.yaml'
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='GPU hours'):server_config(p)
    cfg['gpu_hours']=100;p.write_text(yaml.safe_dump(cfg));assert server_config(p)['gpu_hours']==100
    from cxr.phase35.preflight import prepare
    import platform
    if platform.system()=='Windows':
        with pytest.raises(RuntimeError,match='Linux'):prepare(cfg,load_plan())
    cfg['gpu_hours']=float('inf');p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):server_config(p)

def test_fold1_copy_hash_and_no_retraining(tmp_path):
    source=tmp_path/'phase32/experiments/P32';source.mkdir(parents=True)
    (source/'config.yaml').write_text('frozen: true');(source/'best_model.pth').write_bytes(b'SYNTHETIC')
    sealed={p.name:digest(p) for p in source.iterdir()}
    plan={'pilot_results_commit':'synthetic','reused_fold1':{'POS':{'id':'P32','sha256':sealed}}}
    cfg={'phase32_results_root':str(tmp_path/'phase32'),'output_root':str(tmp_path/'phase35')}
    reuse_fold1(cfg,plan);reuse_fold1(cfg,plan)
    assert {p.name:digest(p) for p in source.iterdir()}==sealed
    assert json.loads((tmp_path/'phase35/reused_fold1/P32/status.json').read_text())['state']=='complete'
    (tmp_path/'phase35/reused_fold1/P32/best_model.pth').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):reuse_fold1(cfg,plan)

def test_queue_eight_gpu_assignment_retry_budget_resume(tmp_path):
    from cxr.phase35.queue import run
    plan=load_plan();cfg={'output_root':str(tmp_path/'queue'),'gpu_ids':[0,1],'max_concurrent':2,'gpu_hours':110}
    active=set();calls=[];guard=threading.Lock()
    def execute(task,root,gpu,attempt):
        with guard:
            assert gpu not in active;active.add(gpu);calls.append((task['id'],attempt))
        time.sleep(.002)
        with guard:active.remove(gpu)
        if task==plan['tasks'][0] and attempt==1:return {'state':'FAILED','reason':'recoverable transport','retryable':True,'gpu_hours':.01}
        if task==plan['tasks'][1]:return {'state':'FAILED','reason':'CUDA OOM','retryable':False,'gpu_hours':.01}
        write_json(root/'experiments'/task['id']/'DONE.json',{'state':'DONE','sha256':{}})
        return {'state':'DONE','gpu_hours':.01}
    s=run(plan,cfg,execute,poll_seconds=.001);before=len(calls)
    again=run(plan,cfg,execute,poll_seconds=.001)
    assert before==len(calls)==9 and again['gpu_hours']==s['gpu_hours']
    assert sum(r['state']=='DONE' for r in s['tasks'].values())==7
    assert s['tasks'][plan['tasks'][1]['id']]['state']=='FAILED'
    assert (tmp_path/'queue/reports/experiment_summary.csv').exists()
    cfg['output_root']=str(tmp_path/'no_budget');cfg['gpu_hours']=1
    s=run(plan,cfg,execute,poll_seconds=.001)
    assert all(r['state']=='PENDING' for r in s['tasks'].values())

def table(n,patients,prefix):
    y=np.tile(np.asarray([i%2 for i in range(n)])[:,None],(1,10))
    return {'keys':[(f'{prefix}p{i%patients}',f'{prefix}s{i}') for i in range(n)],'p':.2+.6*y,'y':y,'counts':np.ones(n,int)}

def rich(path,t):
    csv_write(path,[{'Subject_id':s,'Study_id':st,'image_count':int(t['counts'][i]),
                   **{f'probability_{j}':float(t['p'][i,j-1]) for j in range(1,11)},
                   **{f'target_{j}':int(t['y'][i,j-1]) for j in range(1,11)}} for i,(s,st) in enumerate(t['keys'])])

def test_failed_test_inference_resumes_without_training(tmp_path,monkeypatch):
    import cxr.phase35.test_stage as stage
    run=tmp_path/'run';run.mkdir();calls=[];data=table(1414,1414,'test')
    def command(args,env,stdout,stderr,timeout):
        assert 'train' not in args[0];kind=args[args.index('--checkpoint-kind')+1];calls.append(kind)
        if len(calls)==2:return 1
        prefix='' if kind=='raw' else 'ema_';write_probabilities(run/(prefix+'test_predictions.csv'),data,{'kind':kind});return 0
    monkeypatch.setattr(stage,'run_command',command)
    with pytest.raises(RuntimeError):stage.test_stage(run,tmp_path/'config',{},tmp_path/'logs',time.monotonic()+10,pure=True)
    stage.test_stage(run,tmp_path/'config',{},tmp_path/'logs',time.monotonic()+10,pure=True)
    stage.test_stage(run,tmp_path/'config',{},tmp_path/'logs',time.monotonic()+10,pure=True)
    assert calls==['raw','ema','ema']
    assert len(read_test_probabilities(run/'ema_test_predictions.csv')['keys'])==1414

def test_complete_model_export_requires_test_and_is_portable(tmp_path):
    plan=load_plan();plan['selection']['bootstrap_valid_draws']=2;t=plan['tasks'][0]
    run=tmp_path/'experiments'/t['id'];run.mkdir(parents=True)
    (run/'config.yaml').write_bytes((ROOT/t['config']).read_bytes());write_json(run/'status.json',{'state':'complete'})
    with pytest.raises(FileNotFoundError):finalize(run,t,plan,tmp_path/'baseline')
    (run/'metrics.jsonl').write_text(json.dumps({'epoch':1,'val':{'macro_auc':1,'macro_f1':1},'ema_val':{'macro_auc':1},'elapsed_seconds':1,'gpu_peak_allocated_mib':0})+'\n')
    (run/'train_stdout.log').write_text('SYNTHETIC CPU ONLY');write_json(run/'pretrained_source.json',{'source':'SYNTHETIC'})
    for kind in ('raw','ema'):
        pre='' if kind=='raw' else 'ema_'
        torch.save({'model_state_dict':{'w':torch.ones(2)*(1 if kind=='raw' else 2)}},run/('best_model.pt' if kind=='raw' else 'best_ema_model.pt'))
        write_json(run/('best_metrics.json' if kind=='raw' else 'best_ema_metrics.json'),{'epoch':1})
        for folder,n,patients in [('validation_analysis',859,228),('oof_analysis',t['expected_counts']['oof']['studies'],2211)]:
            data=table(n,patients,folder);rich(run/(pre+folder)/'study_predictions.csv',data)
            rich(tmp_path/'baseline/experiments'/t['baseline_id']/(pre+folder)/'study_predictions.csv',data)
        p=run/(pre+'test_predictions.csv');write_probabilities(p,table(1414,1414,'test'),{'test_auc':None})
        write_json(run/(pre+'test.done.json'),{p.name:digest(p),p.with_suffix('.provenance.json').name:digest(p.with_suffix('.provenance.json'))})
    finalize(run,t,plan,tmp_path/'baseline');verify_done(run)
    data=json.loads((run/'metrics.json').read_text());assert data['test_auc'] is None
    assert data['variants']['ema']['oof']['all']['studies']==3731
    assert data['variants']['raw']['test']['studies']==1414
    import shutil
    moved=tmp_path/'relocated';shutil.copytree(run,moved);verify_done(moved)
    assert torch.load(moved/'ema_best_model.pth',weights_only=True)['w'][0]==2

def test_full_fivefold_pipeline_and_confirmation(tmp_path,monkeypatch):
    import cxr.phase35.analysis as m
    plan=load_plan();plan['selection']['bootstrap_valid_draws']=2
    # Exercise one complete branch and one incomplete branch, all real coverage counts.
    plan['reused_fold1']['POS']['sha256']={}
    first=plan['reused_fold1']['POS']['id'];runroot=tmp_path/'output';base=tmp_path/'baseline'
    index=[];audit={'baselines':{}}
    for f in range(1,6):
        task=next((t for t in plan['tasks'] if t['variant']=='POS' and t['fold']==f),None)
        count=3687 if f==1 else task['expected_counts']['oof']['studies']
        ot=table(count,2211,f'f{f}_');vt=table(859,228,'val_')
        run=runroot/('reused_fold1' if f==1 else 'experiments')/(first if f==1 else task['id'])
        run.mkdir(parents=True);(run/'resolved_config.yaml').write_text('synthetic: true')
        audit['baselines'][f'N17_F{f:02d}']={'sha256':{}}
        for i,(s,st) in enumerate(ot['keys']):index.append({'Subject_id':s,'Study_id':st,'fold':f,'Predict_class':';'.join(map(str,range(1,11))) if ot['y'][i,0] else ''})
        # Study parser requires a nonblank class list. Use alternating groups instead.
        for j in range(10):ot['y'][:,j]=(np.arange(count)+j)%2;vt['y'][:,j]=(np.arange(859)+j)%2
        ot['p']=.2+.6*ot['y'];vt['p']=.2+.6*vt['y']
        for i in range(count):index[-count+i]['Predict_class']=';'.join(str(j+1) for j in range(10) if ot['y'][i,j])
        for pre in ('','ema_'):
            for folder,tbl in [('oof_analysis',ot),('validation_analysis',vt)]:rich(run/(pre+folder)/'study_predictions.csv',tbl)
            p=base/f'experiments/N17_F{f:02d}'/(pre+'oof_analysis/study_predictions.csv');rich(p,ot)
            audit['baselines'][f'N17_F{f:02d}']['sha256'][pre+'oof_analysis/study_predictions.csv']=digest(p)
            write_probabilities(run/(pre+'test_predictions.csv'),table(1414,1414,'test'),{})
            (run/(pre+'best_model.pth')).write_bytes(b'SYNTHETIC')
            write_json(run/(pre+'latency.json'),{'mean_ms':1,'device':'SYNTHETIC_CPU','scope':'SYNTHETIC'})
        if f!=1:write_json(run/'DONE.json',{'state':'DONE','sha256':{p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file()}})
    p=tmp_path/'outputs/analysis/phase17_patient_folds/study_folds.csv';csv_write(p,index);plan['fold_hashes']['study_folds.csv']=digest(p)
    write_json(tmp_path/'docs/e08_tune/source_audit.json',audit);monkeypatch.setattr(m,'ROOT',tmp_path)
    result=m.complete(runroot,plan,base);m.complete(runroot,plan,base)
    assert result['POS_ema']['oof']['studies']==18513 and result['POS_ema']['oof']['patients']==11055
    assert result['POS_raw']['val_fivefold_probability_mean']['macro_auc']==1
    assert result['POS_ema']['folds2to5_confirmation']['differences']['candidate_minus_N17']['point']==0
    assert 'INCOMPLETE' in (runroot/'reports/five_fold_summary.csv').read_text()
    assert not list(runroot.rglob('*submission*'))
    # Wrong patient identity must fail even if total row count is preserved.
    bad=copy.deepcopy(ot);bad['keys'][0]=('outsider','wrong')
    with pytest.raises(ValueError):m.validate_oof(bad,5,index)

def test_mac_counter_cpu_and_cli_files():
    spec=importlib.util.spec_from_file_location('p35lat',ROOT/'scripts/phase35_latency.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    model=torch.nn.Sequential(torch.nn.Conv2d(3,2,3,bias=False),torch.nn.Flatten(),torch.nn.Linear(8,10,bias=False)).eval()
    assert m.convolution_linear_macs(model,torch.zeros(1,3,4,4))==216+80
    for p in list((ROOT/'src/cxr/phase35').glob('*.py'))+list((ROOT/'scripts').glob('phase35_*.py')):compile(p.read_text(encoding='utf-8'),str(p),'exec')

def test_execution_failure_after_train_does_not_retrain(tmp_path,monkeypatch):
    import cxr.phase35.execution as ex
    import cxr.phase35.test_stage as ts
    plan=load_plan();t=plan['tasks'][0];calls=[];test_calls=[]
    from cxr.phase35.protocol import fingerprint
    from cxr.batch.artifacts import commit
    write_json(tmp_path/'preflight/REMOTE_PREFLIGHT.json',{'identity':{'plan':fingerprint(plan),'code_commit':commit()}})
    monkeypatch.setenv('CXR_PHASE35_BASELINES',str(tmp_path/'baseline'))
    def command(args,env,stdout,stderr,timeout):
        calls.append(args[0]);run=tmp_path/'experiments'/t['id']
        if args[0].endswith('train.py'):
            write_json(run/'status.json',{'state':'complete'});(run/'train_stderr.log').write_text('')
        elif args[0].endswith('latency.py'):
            prefix='ema_' if args[-1]=='ema' else '';write_json(run/(prefix+'latency.json'),{'state':'SYNTHETIC'})
        else:
            prefix='ema_' if '--checkpoint-kind' in args else ''
            folder='validation_analysis' if 'validation' in args[0] else 'oof_analysis'
            write_json(run/(prefix+folder)/'analysis.json',{'state':'SYNTHETIC'})
        return 0
    def test_stage(*args,**kwargs):
        test_calls.append(1)
        if len(test_calls)==1:raise RuntimeError('Synthetic Test interruption')
    monkeypatch.setattr(ex,'run_command',command);monkeypatch.setattr(ts,'test_stage',test_stage)
    monkeypatch.setattr(ex,'finalize',lambda run,*a:write_json(run/'DONE.json',{'state':'DONE','sha256':{}}))
    assert ex.execute(t,tmp_path,'0',1)['state']=='FAILED'
    assert ex.execute(t,tmp_path,'0',2)['state']=='DONE'
    assert sum(x.endswith('train.py') for x in calls)==1
    assert len(test_calls)==2

def test_missing_fold1_dependency_blocks_before_gpu_probe(tmp_path):
    plan=load_plan();cfg={'phase32_results_root':str(tmp_path/'absent'),'output_root':str(tmp_path/'output')}
    with pytest.raises(ValueError,match='frozen Fold1'):reuse_fold1(cfg,plan)
    assert not list(tmp_path.rglob('best_model.pth'))

def test_upload_private_branch_and_release_no_overwrite(tmp_path,monkeypatch):
    import cxr.phase35.upload as m
    root=tmp_path/'phase35_batch';run=root/'experiments/P35_POS_F02';run.mkdir(parents=True)
    for name in ('best_model.pth','ema_best_model.pth'):(run/name).write_bytes(b'SYNTHETIC_'+name.encode())
    (run/'metrics.json').write_text('{"test_auc":null}')
    write_json(run/'DONE.json',{'state':'DONE','sha256':{p.name:digest(p) for p in run.iterdir()}})
    cfg={'enabled':True,'repository':'Zoey-711/cxr-multilabel-challenge','branch':'results/phase35-fivefold',
         'checkout':str(tmp_path/'checkout'),'release_tag':'phase35-test'}
    actions=[];assets={};published=[False]
    def output(args,**kwargs):
        actions.append(args)
        if args[:3]==['gh','repo','view']:return json.dumps({'isPrivate':True,'viewerPermission':'WRITE'})
        if args[:2]==['gh','api']:return json.dumps({'assets':list(assets.values())})
        if args[:3]==['gh','release','upload']:
            spec=args[4];path,label=spec.split('#');p=Path(path)
            assert p.suffix=='.pth' and '--clobber' not in args
            assets[p.name]={'name':p.name,'size':p.stat().st_size,'label':label};return ''
        if args[:3]==['gh','release','create']:published[0]=True;return ''
        if args[0]=='git':
            rest=args[3:]
            if rest[:2]==['remote','get-url']:return 'https://github.com/Zoey-711/cxr-multilabel-challenge.git'
            if rest[:2]==['branch','--show-current']:return cfg['branch']
            if rest[:2]==['diff','--cached']:return 'results/phase35_batch/export_manifest.json'
            return ''
        raise AssertionError(args)
    monkeypatch.setattr(m.subprocess,'check_output',output)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':0 if published[0] else 1})())
    monkeypatch.setattr(m,'commit',lambda:'a'*40)
    m.upload(root,cfg);m.upload(root,cfg)
    assert len([a for a in actions if a[:3]==['gh','release','upload']])==2
    assert not list((tmp_path/'checkout').rglob('*.pth'))
    assert (tmp_path/'checkout/results/phase35_batch/experiments/P35_POS_F02/DONE.json').is_file()
    assert not any('--force' in a or '--clobber' in a for a in actions)
    assets['P35_POS_F02__best_model.pth']['label']='wrong'
    with pytest.raises(ValueError,match='differs'):m.upload(root,cfg)
    cfg['branch']='results/phase32-pilots'
    with pytest.raises(ValueError,match='Dedicated'):m.upload(root,cfg)
