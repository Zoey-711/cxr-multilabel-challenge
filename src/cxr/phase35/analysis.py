"""Pinned Fold1 plus four new held-out folds; full and confirmation estimates."""
import csv,json
from pathlib import Path
import numpy as np
from cxr.batch.plan import ROOT,digest,write_json
from cxr.batch.results import read_table,align,combine,concatenate_oof,scores,bootstrap
from cxr.batch.artifacts import write_probabilities
from cxr.batch.test_results import read_test_probabilities,average_test
from cxr.phase35.results import verify_done,csv_write
from cxr.phase35.protocol import verify_reused

def validate_oof(table,fold,study_index):
    expected={(r['Subject_id'],r['Study_id']):r for r in study_index if int(r['fold'])==fold}
    if len(table['keys'])!=len(expected) or set(table['keys'])!=set(expected):raise ValueError('OOF fold coverage differs')
    held={s for s,_ in expected};training={r['Subject_id'] for r in study_index if int(r['fold'])!=fold}
    if held&training:raise ValueError('Patient crosses OOF and training')
    for i,k in enumerate(table['keys']):
        positive=set(map(int,expected[k]['Predict_class'].split(';')))
        if not np.array_equal(table['y'][i],[int(j in positive) for j in range(1,11)]):raise ValueError('OOF labels differ')

def immutable_predictions(path,table,provenance):
    path=Path(path)
    if path.exists():
        old=read_test_probabilities(path,expected_keys=table['keys'],expected_studies=len(table['keys']))
        ix={k:i for i,k in enumerate(old['keys'])}
        if not np.array_equal(old['p'][[ix[k] for k in table['keys']]],table['p']):raise ValueError('Existing aggregate differs; no overwrite')
        return
    write_probabilities(path,table,provenance)

def complete(root,plan,baseline_root):
    root,baseline_root=Path(root),Path(baseline_root);rows=[];details={};classes=[];costs=[]
    foldpath=ROOT/'outputs/analysis/phase17_patient_folds/study_folds.csv'
    if digest(foldpath)!=plan['fold_hashes']['study_folds.csv']:raise ValueError('Fold index changed')
    with foldpath.open(encoding='utf-8-sig',newline='') as f:index=list(csv.DictReader(f))
    source_audit=json.loads((ROOT/'docs/e08_tune/source_audit.json').read_text(encoding='utf-8'))
    for variant,reuse in plan['reused_fold1'].items():
        tasks=sorted([t for t in plan['tasks'] if t['variant']==variant],key=lambda t:t['fold'])
        first=root/'reused_fold1'/reuse['id'];runs=[first]+[root/'experiments'/t['id'] for t in tasks]
        if not all((r/'DONE.json').exists() for r in runs[1:]):
            rows.append({'Variant':variant,'Status':'INCOMPLETE','OOF_AUC':None,'Val_Ensemble_AUC':None,'Confirmation_Delta':None});continue
        verify_reused(first,reuse)
        for r in runs[1:]:verify_done(r)
        for kind in ('raw','ema'):
            prefix='' if kind=='raw' else 'ema_';name=variant+'_'+kind
            oo=[read_table(r/(prefix+'oof_analysis')/'study_predictions.csv') for r in runs]
            for f,t in enumerate(oo,1):validate_oof(t,f,index)
            whole=concatenate_oof(oo)
            if len(whole['keys'])!=18513 or len({s for s,_ in whole['keys']})!=11055:raise ValueError('Incomplete full OOF')
            vv=[read_table(r/(prefix+'validation_analysis')/'study_predictions.csv') for r in runs]
            val=combine(vv,[.2]*5)
            if len(val['keys'])!=859 or len({s for s,_ in val['keys']})!=228:raise ValueError('Val coverage changed')
            base=[]
            for f in range(1,6):
                rel=prefix+'oof_analysis/study_predictions.csv';p=baseline_root/'experiments'/f'N17_F{f:02d}'/rel
                if digest(p)!=source_audit['baselines'][f'N17_F{f:02d}']['sha256'][rel]:raise ValueError('N17 reference changed')
                t=read_table(p);align(oo[f-1],t);base.append(t)
            allbase=concatenate_oof(base);draws=plan['selection']['bootstrap_valid_draws'];seed=plan['selection']['bootstrap_seed']
            ci=bootstrap({'candidate':whole,'N17':allbase},[('candidate','N17')],draws=draws,seed=seed)
            confirm=bootstrap({'candidate':concatenate_oof(oo[1:]),'N17':concatenate_oof(base[1:])},[('candidate','N17')],draws=draws,seed=seed)
            metrics=scores(whole);vs=scores(val)
            detail={'oof':metrics,'val_fivefold_probability_mean':vs,'full_oof_bootstrap':ci,'folds2to5_confirmation':confirm,
                    'scope':'Fold1 selected in Phase32; Folds2-5 confirmation; official Val repeatedly used for epoch selection',
                    'test_auc':None,'deployment':'T4_UNVERIFIED'}
            immutable_predictions(root/f'oof/{name}_all.csv',whole,{'fold1_source_commit':plan['pilot_results_commit'],'formula':'concatenate corresponding held-out fold only'})
            immutable_predictions(root/f'validation/{name}_fivefold.csv',val,{'formula':'equal mean of five Study-level probability matrices'})
            # ID-aligned Test only. No submission generation or Test labels.
            test=average_test([read_test_probabilities(r/(prefix+'test_predictions.csv')) for r in runs],[.2]*5)
            immutable_predictions(root/f'test_predictions/{name}_fivefold.csv',test,{'formula':'equal five-fold Study Mean probability average','test_auc':None})
            size=sum((r/(prefix+'best_model.pth')).stat().st_size+(r/'resolved_config.yaml').stat().st_size for r in runs)
            detail['inference_weights_and_config_bytes']=size
            detail['size_status']='PASS' if size<=500000000 else 'SIZE_FAIL'
            detail['sources']=[{'model':r.name,'sha256':digest(r/(prefix+'best_model.pth'))} for r in runs]
            details[name]=detail
            rows.append({'Variant':name,'Status':'COMPLETE_OOF','OOF_AUC':metrics['macro_auc'],'Val_Ensemble_AUC':vs['macro_auc'],
                         'Confirmation_Delta':confirm['differences']['candidate_minus_N17']['point']})
            for split,m in [('oof',metrics),('val',vs)]:classes.extend({'variant':name,'split':split,**c} for c in m['per_class'])
            for f,r in enumerate(runs,1):
                latency=json.loads((r/(prefix+'latency.json')).read_text())
                costs.append({'model':r.name,'variant':kind,'fold':f,'weight_bytes':(r/(prefix+'best_model.pth')).stat().st_size,
                    'input_size':384 if variant=='POS' else 512,'macs':latency.get('conv_linear_macs'),'mean_ms':latency.get('mean_ms'),
                    'p95_ms':latency.get('p95_ms'),'device':latency.get('device'),'scope':latency.get('scope'),
                    'protocol':'PHASE32_LEGACY' if f==1 else 'PHASE35_PAIRED_INPUTS','status':'T4_UNVERIFIED'})
    csv_write(root/'reports/five_fold_summary.csv',rows);write_json(root/'reports/five_fold_analysis.json',details)
    csv_write(root/'analysis/per_class_metrics.csv',classes);csv_write(root/'deployment/cost_comparison.csv',costs)
    comparisons={}
    for kind in ('raw','ema'):
        pair={size:[r for r in costs if r['variant']==kind and r['fold']!=1 and r['input_size']==size] for size in (384,512)}
        if all(pair.values()):
            mean={size:float(np.mean([r['mean_ms'] for r in group])) for size,group in pair.items()}
            macs={size:float(np.mean([r['macs'] for r in group])) if all(r['macs'] is not None for r in group) else None for size,group in pair.items()}
            comparisons[kind]={'mean_ms_384':mean[384],'mean_ms_512':mean[512],'delta_ms_512_minus_384':mean[512]-mean[384],
                               'conv_linear_macs_384':macs[384],'conv_linear_macs_512':macs[512],
                               'limitation':'Same fixed Val input images and timing code; concurrent server load may differ. Not controlled T4 measurement.'}
    write_json(root/'deployment/resolution_comparison.json',comparisons)
    write_json(root/'reports/delivery_manifest.json',{'training_tasks':8,'reused_fold1':2,'complete_variants':list(details),
        'files':{p.relative_to(root).as_posix():digest(p) for folder in ('reports','analysis','deployment','oof','validation','test_predictions')
                 for p in (root/folder).rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}})
    return details
