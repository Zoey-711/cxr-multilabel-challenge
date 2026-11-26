"""CPU-only analysis of three fixed N17 EMA fusions; never infer Test labels."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location('phase27_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alternate_probabilities(path, keys, prefix):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['Subject_id', 'Study_id'] + [f'{prefix}{i}' for i in range(1,11)]
        if reader.fieldnames[:12] != expected:
            raise ValueError(f'Class mapping changed: {path}')
        rows = list(reader)
    mapping = {(r['Subject_id'],r['Study_id']): [float(r[f'{prefix}{i}']) for i in range(1,11)] for r in rows}
    if len(rows) != len(keys) or set(mapping) != set(keys):
        raise ValueError(f'Study IDs differ: {path}')
    result = np.asarray([mapping[k] for k in keys],dtype=np.float64)
    if not np.isfinite(result).all() or (result<0).any() or (result>1).any():
        raise ValueError(f'Invalid probabilities: {path}')
    return result


def write_probabilities(path, keys, matrix):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Subject_id','Study_id']+[f'Prob_{i}' for i in range(1,11)])
        for key, vector in zip(keys,matrix):
            w.writerow([*key,*[repr(float(v)) for v in vector]])


def write_submission(path, keys, matrix):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Subject_id','Study_id','Predict_class','Probability'])
        for key, vector in zip(keys,matrix):
            for i, v in enumerate(vector,1):
                w.writerow([*key,i,repr(float(v))])
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        rows=list(reader)
    actual={(r['Subject_id'],r['Study_id'],int(r['Predict_class'])) for r in rows}
    expected={(s,t,i) for s,t in keys for i in range(1,11)}
    if len(rows)!=14140 or len(actual)!=14140 or actual!=expected:
        raise ValueError('Submission row/Study/class coverage differs')
    if any(not np.isfinite(float(r['Probability'])) or not 0<=float(r['Probability'])<=1 for r in rows):
        raise ValueError('Submission probabilities invalid')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',required=True,type=Path)
    ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--phase27',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root=args.version/'results/phase17_batch'
    project=args.project
    helper=load_module(Path(__file__).with_name('Analyze-Phase27.py'))
    prior=json.loads((args.phase27/'phase27_audit.json').read_text(encoding='utf-8'))
    if prior['source']['commit']!='cbfc5997438f371086cd1fbe232890590cd67ffb' or prior['counts']!={'PASS':26}:
        raise ValueError('Phase27 source audit differs')
    val_keys,yv=helper.labels(root/'manifests/val_labels.csv',859)
    train_keys,yt=helper.labels(root/'manifests/train_labels.csv',18513)
    n_val=helper.probs(root/'validation/N17_EMA_val_ensemble.csv',val_keys)
    n_oof=helper.probs(root/'oof/N17_EMA_oof_all.csv',train_keys)
    f_val=helper.probs(root/'validation/E17_val_probabilities.csv',val_keys)
    f_oof=helper.probs(root/'analysis/gh_teacher_oof.csv',train_keys)
    local_oof_path=project/'outputs/analysis/phase17_final_handoff_20261023/e17_oof_study_probabilities.csv'
    l_oof=helper.probs(local_oof_path,train_keys)
    local_freeze=json.loads((local_oof_path.parent/'freeze_manifest.json').read_text(encoding='utf-8'))
    local_val_path=project/'outputs/analysis/phase17_five_fold_final_20261023/e17_val_fivefold.csv'
    if digest(local_val_path)!=local_freeze['source_outputs']['e17_val_fivefold.csv']['sha256']:
        raise ValueError('Local frozen E17 Val hash differs')
    l_val=alternate_probabilities(local_val_path,val_keys,'probability_')
    e08_val_path=project/'outputs/submissions/E08_EMA_20261020/val_predictions.csv'
    e08_manifest=json.loads((e08_val_path.parent/'manifest.json').read_text(encoding='utf-8'))
    if digest(e08_val_path)!=e08_manifest['files']['val_predictions.csv']['sha256']:
        raise ValueError('Frozen E08 Val hash differs')
    e_val=alternate_probabilities(e08_val_path,val_keys,'probability_')
    h_val=helper.probs(root/'experiments/H17_DISTILL_02/val_predictions.csv',val_keys)
    if abs(helper.score(yv,e_val)[0]-0.8116686461316818)>1e-9:
        raise ValueError('Frozen E08 Val AUC differs')
    if abs(helper.score(yt,l_oof)[0]-0.8030492475894413)>1e-9:
        raise ValueError('Frozen local E17 OOF AUC differs')
    if abs(helper.score(yt,n_oof)[0]-0.8076721227703633)>1e-9:
        raise ValueError('Friend N17 EMA OOF AUC differs')
    # The Phase27 audit checked each patient's OOF source against the same frozen fold hash.
    fold_hashes=prior['fold_hashes']
    if fold_hashes!=local_freeze['fold_hashes']:
        raise ValueError('Friend and local fivefold splits differ')
    candidates={
        'F_E17_N17EMA': {'val':(f_val+n_val)/2,'oof':(f_oof+n_oof)/2,
                          'source_val':[root/'validation/E17_val_probabilities.csv',root/'validation/N17_EMA_val_ensemble.csv'],
                          'source_oof':[root/'analysis/gh_teacher_oof.csv',root/'oof/N17_EMA_oof_all.csv']},
        'L_E17_N17EMA': {'val':(l_val+n_val)/2,'oof':(l_oof+n_oof)/2,
                          'source_val':[local_val_path,root/'validation/N17_EMA_val_ensemble.csv'],
                          'source_oof':[local_oof_path,root/'oof/N17_EMA_oof_all.csv']},
        'E08_N17EMA': {'val':(e_val+n_val)/2,'oof':None,
                         'source_val':[e08_val_path,root/'validation/N17_EMA_val_ensemble.csv'],
                         'source_oof':[]},
    }
    args.output.mkdir(parents=True)
    results={'source_commit':prior['source']['commit'],'fold_hashes':fold_hashes,'models':{},'candidates':{},
             'latency_target':'T4 batch=1 full read-to-write, 100 samples x 3 runs after 5 warmup images',
             'test_auc':None}
    model_arrays={'FRIEND_E17':(f_val,f_oof),'LOCAL_E17':(l_val,l_oof),
                  'N17_EMA':(n_val,n_oof),'E08':(e_val,None),'H17_02':(h_val,None)}
    for name,(vp,op) in model_arrays.items():
        va,vper=helper.score(yv,vp)
        oa,oper=helper.score(yt,op) if op is not None else (None,None)
        results['models'][name]={'val_auc':va,'val_per_class_auc':vper,'oof_auc':oa,'oof_per_class_auc':oper}
    for name,c in candidates.items():
        va,vper=helper.score(yv,c['val'])
        oa,oper=helper.score(yt,c['oof']) if c['oof'] is not None else (None,None)
        corr=[]
        if c['oof'] is not None:
            left,right=(f_oof,n_oof) if name=='F_E17_N17EMA' else (l_oof,n_oof)
            corr=[float(np.corrcoef(left[:,i],right[:,i])[0,1]) for i in range(10)]
        results['candidates'][name]={
            'formula':'0.5 * '+name.split('_N17EMA')[0]+' + 0.5 * N17_EMA',
            'val_auc':va,'val_delta_vs_e08':va-results['models']['E08']['val_auc'],
            'val_per_class_auc':vper,
            'val_per_class_delta_vs_e08':[x-y for x,y in zip(vper,results['models']['E08']['val_per_class_auc'])],
            'oof_auc':oa,'oof_per_class_auc':oper,'oof_prediction_correlation_per_class':corr,
            'val_bootstrap_vs_e08':helper.bootstrap(yv,c['val'],e_val,val_keys,2000,42),
            'source_val_sha256':{str(p):digest(p) for p in c['source_val']},
            'source_oof_sha256':{str(p):digest(p) for p in c['source_oof']},
            'test_status':'NOT_CHECKED', 'deployment_status':'LATENCY_UNVERIFIED'}
        if c['oof'] is not None:
            base=f_oof if name=='F_E17_N17EMA' else l_oof
            results['candidates'][name]['oof_delta_vs_e17']=oa-helper.score(yt,base)[0]
            results['candidates'][name]['oof_bootstrap_vs_e17']=helper.bootstrap(yt,c['oof'],base,train_keys,1000,43)
            results['candidates'][name]['n17_minus_e17_oof_bootstrap']=helper.bootstrap(yt,n_oof,base,train_keys,1000,44)
    n_bytes=sum(int(json.loads((root/'experiments'/f'N17_F{i:02d}'/'run_manifest.json').read_text(encoding='utf-8'))['files']['ema_best_model.pth']['bytes']) for i in range(1,6))
    f_bytes=1251159645
    l_bytes=sum(int(x['checkpoint_bytes']) for x in local_freeze['models'])
    e_bytes=int(e08_manifest['total_checkpoint_bytes'])
    for name,base in [('F_E17_N17EMA',f_bytes),('L_E17_N17EMA',l_bytes),('E08_N17EMA',e_bytes)]:
        total=base+n_bytes
        item=results['candidates'][name]
        item.update({'reported_inference_weight_bytes':total,'weight_count':20 if name!='E08_N17EMA' else 10,
                     'size_status':'SIZE_FAIL' if total>500000000 else 'SIZE_PENDING_CONFIG_FILES',
                     'weights_downloaded':False})
        if total>500000000:
            item['deployment_status']='SIZE_FAIL'
    results['weight_sources']={'friend_E17_bytes':f_bytes,'local_E17_bytes':l_bytes,
                               'E08_frozen_bytes':e_bytes,'N17_EMA_fivefold_manifest_bytes':n_bytes,
                               'friend_weights_not_downloaded':True}
    # Only existing deterministic Study-level Test probabilities may be fused.
    n_test_path=root/'test_predictions/N17_EMA_test_5fold.csv'
    f_test_path=root/'test_predictions/E17_test_probabilities.csv'
    e_test_path=project/'outputs/submissions/E08_EMA_20261020/test_predictions.csv'
    for path,provenance in [(n_test_path,n_test_path.with_suffix('.provenance.json')),
                            (f_test_path,f_test_path.with_suffix('.provenance.json'))]:
        recorded=json.loads(provenance.read_text(encoding='utf-8'))['csv_sha256']
        if digest(path)!=recorded:
            raise ValueError(f'Test probability source hash differs: {path}')
    if digest(e_test_path)!=e08_manifest['files']['test_predictions.csv']['sha256']:
        raise ValueError('Frozen E08 Test probability hash differs')
    with e_test_path.open(encoding='utf-8-sig',newline='') as f:
        er=list(csv.DictReader(f))
    test_keys=sorted({(r['Subject_id'],r['Study_id']) for r in er})
    if len(er)!=1414 or len(test_keys)!=1414:
        raise ValueError('E08 Test Study coverage differs')
    n_test=helper.probs(n_test_path,test_keys)
    f_test=helper.probs(f_test_path,test_keys)
    e_test=alternate_probabilities(e_test_path,test_keys,'probability_')
    research=args.output/'research_only'
    research.mkdir()
    for name,base,source in [('F_E17_N17EMA',f_test,f_test_path),('E08_N17EMA',e_test,e_test_path)]:
        matrix=(base+n_test)/2
        prob_path=research/f'{name}_test_probabilities.csv'
        submission_path=research/f'{name}_submission.csv'
        write_probabilities(prob_path,test_keys,matrix)
        write_submission(submission_path,test_keys,matrix)
        results['candidates'][name].update({'test_status':'FORMAT_PASS_RESEARCH_ONLY','test_studies':1414,
            'submission_rows':14140,'submission_path':str(submission_path),
            'submission_sha256':digest(submission_path),'test_probabilities_sha256':digest(prob_path),
            'test_source_sha256':{str(source):digest(source),str(n_test_path):digest(n_test_path)}})
    results['candidates']['L_E17_N17EMA']['test_status']='UNAVAILABLE_FULL_LOCAL_E17_TEST_PROBABILITIES'
    (args.output/'phase28_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (args.output/'fixed_fusion_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        columns=['candidate','val_auc','val_delta_vs_e08','oof_auc','reported_inference_weight_bytes','size_status','deployment_status','test_status','submission_path']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name,item in results['candidates'].items():
            w.writerow({'candidate':name,**{k:item.get(k,'') for k in columns[1:]}})
    print(json.dumps({'models':{k:[v['val_auc'],v['oof_auc']] for k,v in results['models'].items()},
                      'candidates':{k:[v['val_auc'],v['oof_auc'],v['reported_inference_weight_bytes'],v['test_status']] for k,v in results['candidates'].items()},
                      'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':
    main()
