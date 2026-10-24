"""Incremental CPU export; single-Fold evidence never becomes complete OOF."""
import csv
import io
import json
import shutil
from pathlib import Path
import numpy as np
from cxr.batch.plan import digest, write_json
from cxr.batch.artifacts import commit, utcnow, write_probabilities
from cxr.batch.results import read_table, scores, grouped, bootstrap, align

def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO(newline='')
    if rows:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(stream.getvalue(),encoding='utf-8');temporary.replace(path)

def verify_done(run):
    run=Path(run)
    marker=json.loads((run/'DONE.json').read_text())
    for rel,h in marker['sha256'].items():
        if digest(run/rel)!=h: raise ValueError(f'Completed artifact changed: {rel}')
    return marker

def finalize(run,task,plan,baseline_root):
    import torch
    run=Path(run)
    if (run/'DONE.json').exists():return verify_done(run)
    status=json.loads((run/'status.json').read_text())
    if status['state']!='complete':raise ValueError('Incomplete training')
    shutil.copy2(run/'config.yaml',run/'resolved_config.yaml')
    if (run/'class_weights.json').exists():
        import yaml
        resolved=yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
        resolved['derived_training_class_weights']=json.loads((run/'class_weights.json').read_text())
        (run/'resolved_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    csv_write(run/'train_history.csv',[{**r,'val_macro_auc':r['val']['macro_auc'],
        'val_macro_f1':r['val'].get('macro_f1'),'ema_val_macro_auc':(r.get('ema_val') or {}).get('macro_auc')} for r in history])
    shutil.copy2(run/'train_stdout.log',run/'train.log')
    metrics={'task_id':task['id'],'training_status':status,'oof_scope':'PARTIAL_OOF: Fold 1 only',
             'test_auc':None,'variants':{},'peak_allocated_mib':max(r['gpu_peak_allocated_mib'] for r in history),
             'training_hours':sum(r['elapsed_seconds'] for r in history)/3600}
    per_class=[]
    for kind in (['raw','ema'] if task['ema'] else ['raw']):
        prefix='' if kind=='raw' else 'ema_'
        source=run/('best_model.pt' if kind=='raw' else 'best_ema_model.pt')
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        destination=run/(prefix+'best_model.pth')
        torch.save(checkpoint['model_state_dict'],destination)
        restored=torch.load(destination,map_location='cpu',weights_only=True)
        if any(not torch.equal(restored[k],v.cpu()) for k,v in checkpoint['model_state_dict'].items()):
            raise ValueError('Pure inference export differs')
        destination.with_suffix('.sha256').write_text(digest(destination)+'  '+destination.name+'\n',encoding='ascii')
        result={}
        for split,folder,count in [('val','validation_analysis',859),('oof','oof_analysis',3687)]:
            table=read_table(run/(prefix+folder)/'study_predictions.csv')
            if len(table['keys'])!=count:raise ValueError('Study coverage mismatch')
            if len({s for s,_ in table['keys']}) != (228 if split=='val' else 2211):raise ValueError('Patient coverage mismatch')
            write_probabilities(run/(prefix+split+'_predictions.csv'),table,{
                'model_id':task['id'],'kind':kind,'checkpoint_sha256':digest(destination),
                'fold_hashes':plan['fold_hashes'],'config_sha256':task['config_sha256'],'fold':1})
            result[split]=grouped(table)
            per_class += [{'variant':kind,'split':split,**r} for r in scores(table)['per_class']]
            baseline=Path(baseline_root)/'experiments'/task['baseline_id']/(prefix+folder)/'study_predictions.csv'
            if baseline.is_file():
                base=read_table(baseline);align(table,base)
                result[split+'_comparison']=bootstrap({'pilot':table,'baseline':base},[('pilot','baseline')],
                    draws=plan['selection']['bootstrap_valid_draws'],seed=42)
            else: result[split+'_comparison']={'status':'UNAVAILABLE','reason':'Matching baseline predictions absent'}
        result['weight_bytes']=destination.stat().st_size
        result['deployment_status']='SIZE_FAIL' if result['weight_bytes']+ (run/'resolved_config.yaml').stat().st_size>500000000 else 'LATENCY_UNVERIFIED'
        best=json.loads((run/('best_metrics.json' if kind=='raw' else 'best_ema_metrics.json')).read_text())
        result['best_epoch']=best['epoch']
        latency=run/('latency.json' if kind=='raw' else 'ema_latency.json')
        result['research_latency']=json.loads(latency.read_text()) if latency.exists() else {'state':'PENDING'}
        comparison=result['oof_comparison']
        if 'differences' in comparison:
            difference=comparison['differences']['pilot_minus_baseline']
            baseline_table=read_table(Path(baseline_root)/'experiments'/task['baseline_id']/(prefix+'oof_analysis')/'study_predictions.csv')
            base_classes=scores(baseline_table)['per_class']
            regression=min(a['auc']-b['auc'] for a,b in zip(result['oof']['all']['per_class'],base_classes))
            result['followup_evidence']={'criterion_met':difference['point']>=plan['selection']['min_delta'] and
                difference['ci95'][0]>0 and regression>=-plan['selection']['max_class_regression'],
                'worst_class_delta':regression,'scope':'exploratory Fold1; never automatically launch followup'}
        else:result['followup_evidence']={'criterion_met':None,'state':'UNAVAILABLE'}
        metrics['variants'][kind]=result
    csv_write(run/'per_class_metrics.csv',per_class)
    write_json(run/'metrics.json',metrics)
    write_json(run/'model_size.json',{k:{'bytes':v['weight_bytes'],'status':v['deployment_status']} for k,v in metrics['variants'].items()})
    write_json(run/'run_manifest.json',{'model_id':task['id'],'commit':commit(),'config_sha256':task['config_sha256'],
        'fold_hashes':plan['fold_hashes'],'source_results_commit':plan['source_results_commit'],
        'completed':utcnow(),'kind':'single_fold_pilot','training_pretrained':json.loads((run/'pretrained_source.json').read_text()),
        'limitations':['Official Val repeatedly used for model selection.','Fold1 comparisons exploratory; no multiplicity correction.',
                      '4090 timing cannot prove T4 <=100ms.'],'state':'DONE'})
    files={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file() and p.name not in ('DONE.json','FAILED.json') and not p.name.endswith('.pid.json')}
    write_json(run/'DONE.json',{'state':'DONE','sha256':files,'model_id':task['id']})
    return verify_done(run)

def report(root,state,plan):
    root=Path(root);rows=[]
    for task in plan['tasks']:
        rec=state['tasks'][task['id']];path=root/'experiments'/task['id']/'metrics.json'
        data=json.loads(path.read_text()) if rec['state']=='DONE' and path.exists() else {}
        for kind in (['raw','ema'] if task['ema'] else ['raw']):
            v=data.get('variants',{}).get(kind,{})
            rows.append({'Model_ID':task['id'],'Variant':kind,'State':rec['state'],'Best_Epoch':v.get('best_epoch'),
                         'Val_Macro_AUC':v.get('val',{}).get('all',{}).get('macro_auc'),
                         'Fold1_OOF_Macro_AUC':v.get('oof',{}).get('all',{}).get('macro_auc'),
                         'OOF_Scope':'PARTIAL_OOF','Test_AUC':None,'Weight_Bytes':v.get('weight_bytes'),
                         'Deployment_Status':v.get('deployment_status'),'Reason':rec.get('reason','')})
    csv_write(root/'reports/experiment_summary.csv',rows)
    csv_write(root/'reports/experiment_status.csv',[{'Model_ID':k,**v} for k,v in state['tasks'].items()])
    write_json(root/'reports/experiment_summary.json',rows)
    write_json(root/'reports/delivery_manifest.json',{'commit':commit(),'state':state,'files':{
        p.relative_to(root).as_posix():digest(p) for p in (root/'reports').glob('*.csv')}})
    (root/'reports/phase32_report.md').write_text('Phase32 单折探索性实验\n\n'+
        '\n'.join(f"- {t['id']}: {state['tasks'][t['id']]['state']}" for t in plan['tasks'])+
        '\n\n未完成指标留空；没有 Test AUC。官方 Val 多次选模；Fold1 不代表完整五折。'
        '\n所有候选仍须 T4 batch=1 完整端到端 ≤100ms 和全部权重 ≤500MB 验收。\n',encoding='utf-8')
