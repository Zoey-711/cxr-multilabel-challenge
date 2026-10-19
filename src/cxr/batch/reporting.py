"""CPU-only final/partial report from portable Study probabilities and fixed labels."""
import csv
import json
from pathlib import Path

import yaml

from cxr.batch.artifacts import read_probabilities, write_probabilities, label_keys
from cxr.batch.plan import digest, write_json, deployment_bytes
from cxr.batch.results import combine, concatenate_oof, grouped, scores, bootstrap, align


def csv_rows(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or (list(rows[0]) if rows else [])); w.writeheader(); w.writerows(rows)


def _summary_row(candidate, val=None, oof=None, task=None):
    """Flatten one model/ensemble into the unified summary; never score hidden Test."""
    val = val or {}
    oof = oof or {}
    task = task or {}
    row = {
        'model_id': candidate, 'state': task.get('state', 'derived' if val or oof else 'missing'),
        'best_epoch': task.get('best_epoch'), 'stop_epoch': task.get('stop_epoch'),
        'epoch_hours': task.get('epoch_hours'), 'weights_bytes': task.get('weights_bytes'),
        'val_studies': val.get('studies'), 'val_patients': val.get('patients'),
        'val_macro_auc': val.get('macro_auc'), 'val_macro_ap': val.get('macro_ap'),
        'val_auc': val.get('macro_auc'),
        'val_macro_f1_at_0_5': val.get('macro_f1_at_0_5'),
        'oof_studies': 'N/A' if task.get('oof_not_applicable') else oof.get('studies'),
        'oof_patients': 'N/A' if task.get('oof_not_applicable') else oof.get('patients'),
        'oof_macro_auc': 'N/A' if task.get('oof_not_applicable') else oof.get('macro_auc'),
        'oof_macro_ap': 'N/A' if task.get('oof_not_applicable') else oof.get('macro_ap'),
        'oof_auc': 'N/A' if task.get('oof_not_applicable') else oof.get('macro_auc'),
        'oof_macro_f1_at_0_5': 'N/A' if task.get('oof_not_applicable') else oof.get('macro_f1_at_0_5'),
        'test_auc': 'N/A: hidden Test labels unavailable',
        'test_prediction_status': task.get('test_prediction_status', 'not_scored'),
    }
    for split, metrics in (('val', val), ('oof', oof)):
        per_class = metrics.get('per_class', [])
        for class_id in range(1, 11):
            item = next((x for x in per_class if x.get('class_id') == class_id), {})
            row[f'{split}_auc_class_{class_id}'] = item.get('auc')
            row[f'{split}_ap_class_{class_id}'] = item.get('ap')
            row[f'{split}_positive_studies_class_{class_id}'] = item.get('positive_studies')
            row[f'{split}_negative_studies_class_{class_id}'] = item.get('negative_studies')
            row[f'{split}_positive_patients_class_{class_id}'] = item.get('positive_patients')
            if split == 'oof' and task.get('oof_not_applicable'):
                for suffix in ('auc', 'ap', 'positive_studies', 'negative_studies', 'positive_patients'):
                    row[f'oof_{suffix}_class_{class_id}'] = 'N/A'
    return row


def report(root, *, draws=None):
    root=Path(root)
    plan=yaml.safe_load((root/'manifests/batch_plan.yaml').read_text(encoding='utf-8'))
    label_hashes=json.loads((root/'manifests/labels_sha256.json').read_text(encoding='utf-8'))
    for name,h in label_hashes.items():
        if digest(root/'manifests'/name)!=h:
            raise ValueError('Fixed evaluation labels changed')
    draws=plan['bootstrap_draws'] if draws is None else draws
    val_label=root/'manifests/val_labels.csv'; train_label=root/'manifests/train_labels.csv'
    val_keys=label_keys(val_label)
    tables={}; oofs={}; files={}; rows=[]; failures={}
    for task in plan['tasks']:
        tid=task['id']; run=root/'experiments'/tid
        row={'model_id':tid,'group':task['group'],'state':'missing','best_epoch':None,'val_auc':None,'oof_auc':None,'epoch_hours':None,'weights_bytes':None}
        if (run/'DONE.json').exists():
            manifest=json.loads((run/'DONE.json').read_text(encoding='utf-8'))
            # Result-only delivery intentionally omits .pth; validate all available important files.
            for name,h in manifest['verified_sha256'].items():
                if (run/name).exists() and digest(run/name)!=h:
                    raise ValueError(f'Changed artifact: {tid}/{name}')
            t=read_probabilities(run/'val_predictions.csv',val_label,val_keys)
            tables[tid]=t; files[tid]=[run/'best_model.pth']
            metric=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
            row.update(state='complete',best_epoch=metric['best_epoch'],val_auc=scores(t)['macro_auc'],
                       stop_epoch=metric.get('stop_epoch'), epoch_hours=metric['resources']['epoch_hours'],
                       weights_bytes=metric['weights_bytes'])
            if task['fold']:
                oo=read_probabilities(run/'oof_predictions.csv',train_label,label_keys(train_label,task['fold']))
                if {s for s,_ in oo['keys']} & training_patients(run/'training_image_index.csv'):
                    raise ValueError(f'OOF training-patient leakage: {tid}')
                oofs[tid]=oo; row['oof_auc']=scores(oo)['macro_auc']
            if task['ema']:
                tables[tid+'_EMA']=read_probabilities(run/'ema_val_predictions.csv',val_label,val_keys)
                files[tid+'_EMA']=[run/'ema_best_model.pth']
                if task['fold']:
                    oofs[tid+'_EMA']=read_probabilities(run/'ema_oof_predictions.csv',train_label,label_keys(train_label,task['fold']))
        elif (run/'FAILED.json').exists():
            row['state']='failed'; failures[tid]=json.loads((run/'FAILED.json').read_text(encoding='utf-8'))
        elif (root/'queue_state.json').exists():
            state=json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
            row['state']=state['tasks'].get(tid,{}).get('state','disabled' if not task['enabled'] else 'missing')
        rows.append(row)
    ensemble_tables={}; complete_oof={}; source_maps={}; missing={}
    for branch in ('D17','C17','S17','N17'):
        ids=[f'{branch}_F{i:02d}' for i in range(1,6)]
        missing[branch]=[n for n in ids if n not in tables or n not in oofs]
        if missing[branch]:
            write_json(root/'oof'/(branch+'_status.json'),{'state':'incomplete','missing':missing[branch]})
            continue
        name=branch[0]+'5'
        ensemble_tables[name]=combine([tables[n] for n in ids],[.2]*5)
        complete_oof[name]=concatenate_oof([oofs[n] for n in ids])
        if set(complete_oof[name]['keys'])!=label_keys(train_label):
            raise ValueError('Full OOF coverage differs from frozen official Train labels')
        files[name]=[p for n in ids for p in files[n]]
        provenance={'sources':ids,'weights':[.2]*5,'source_hashes':{n:digest(root/'experiments'/n/'val_predictions.csv') for n in ids}}
        write_probabilities(root/'validation'/(branch+'_val_ensemble.csv'),ensemble_tables[name],provenance)
        write_probabilities(root/'oof'/(branch+'_oof_all.csv'),complete_oof[name],{'sources':ids,'rule':'concatenate unseen fold only'})
        index=[{'Subject_id':s,'Study_id':st,'model_id':n,'fold':i} for i,n in enumerate(ids,1) for s,st in oofs[n]['keys']]
        csv_rows(root/'oof'/(branch+'_source_index.csv'),index)
        write_json(root/'oof'/(branch+'_status.json'),{'state':'complete','studies':len(index),'patients':len({r['Subject_id'] for r in index})})
    for branch in ('C17','N17'):
        ids=[f'{branch}_F{i:02d}_EMA' for i in range(1,6)]
        if not all(n in tables and n in oofs for n in ids):
            write_json(root/'oof'/(branch+'_EMA_status.json'),{'state':'incomplete','missing':[n for n in ids if n not in oofs]})
            continue
        name=branch[0]+'5_EMA'
        ensemble_tables[name]=combine([tables[n] for n in ids],[.2]*5)
        complete_oof[name]=concatenate_oof([oofs[n] for n in ids])
        if set(complete_oof[name]['keys'])!=label_keys(train_label):raise ValueError('EMA OOF coverage mismatch')
        files[name]=[p for n in ids for p in files[n]]
        write_probabilities(root/'validation'/(branch+'_EMA_val_ensemble.csv'),ensemble_tables[name],{'sources':ids,'weights':[.2]*5})
        write_probabilities(root/'oof'/(branch+'_EMA_oof_all.csv'),complete_oof[name],{'sources':ids,'rule':'unseen corresponding fold only'})
        csv_rows(root/'oof'/(branch+'_EMA_source_index.csv'),[{'Subject_id':s,'Study_id':st,'model_id':n,'fold':i} for i,n in enumerate(ids,1) for s,st in oofs[n]['keys']])
    e07=root/'validation/E07_val_probabilities.csv'
    ensemble_tables['E07']=read_probabilities(e07,val_label,val_keys)
    candidates=dict(plan.get('deployment_candidates',{}))
    candidates['E17']={'D5':.25,'C5':.25,'S5':.5}
    if all(f'C17_F{i:02d}_EMA' in tables for i in range(1,6)):
        ids=[f'C17_F{i:02d}_EMA' for i in range(1,6)]
        ensemble_tables['C5_EMA']=combine([tables[n] for n in ids],[.2]*5)
        files['C5_EMA']=[p for n in ids for p in files[n]]
        candidates['E17_EMA_C']={'D5':.25,'C5_EMA':.25,'S5':.5}
    for name,formula in candidates.items():
        available={**tables,**ensemble_tables}
        if not set(formula)<=set(available):
            continue
        ensemble_tables[name]=combine([available[n] for n in formula],list(formula.values()))
        files[name]=list({p for n in formula for p in files[n]})
        write_probabilities(root/'validation'/(name+'_val_probabilities.csv'),ensemble_tables[name],{'formula':formula,'sources':list(formula)})
    if all(n in complete_oof for n in ('D5','C5','S5')):
        complete_oof['E17']=combine([complete_oof[n] for n in ('D5','C5','S5')],[.25,.25,.5])
    official={n:grouped(t) for n,t in {**tables, **ensemble_tables}.items()}
    pairs=[(n,'E07') for n in ensemble_tables if n!='E07']
    boot=bootstrap(ensemble_tables,pairs,draws) if len(ensemble_tables)>1 else None
    oof_metrics={n:grouped(t) for n,t in complete_oof.items()}
    fold_oof_metrics={n:grouped(t) for n,t in oofs.items()}
    all_oof_metrics={**fold_oof_metrics, **oof_metrics}
    oof_boot=bootstrap(complete_oof,[("E17",n) for n in ('D5','C5','S5')],draws) if 'E17' in complete_oof else None
    fold_boot={}
    for i in range(1,6):
        fold_tables={n:oofs[n] for n in (f'D17_F{i:02d}',f'C17_F{i:02d}',f'S17_F{i:02d}') if n in oofs}
        if fold_tables:
            fold_boot[str(i)]=bootstrap(fold_tables,[],draws)
    all_val_boot=bootstrap(tables,[(n,n[:-4]) for n in tables if n.endswith('_EMA')],draws) if tables else None
    deployment=[]
    for name in [n for n in {**tables,**ensemble_tables} if n!='E07']:
        dependencies=[]
        for p in files[name]:
            run=p.parent; man=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
            meta=man['files'][p.name]
            if p.exists() and (p.stat().st_size!=meta['bytes'] or digest(p)!=meta['sha256']):
                raise ValueError('Deployment weight hash mismatch')
            dependencies.append({'path':p.relative_to(root).as_posix(),**meta})
        size=sum(x['bytes'] for x in dependencies)
        latency=[]
        for p in files[name]:
            f=p.parent/('ema_latency.json' if p.name.startswith('ema_') else 'latency.json')
            if f.exists():latency.append(json.loads(f.read_text(encoding='utf-8')))
        full_measurement=root/'deployment/research_benchmarks'/(name+'.json')
        if full_measurement.exists():latency.append(json.loads(full_measurement.read_text(encoding='utf-8')))
        deployment.append({'candidate':name,'bytes':size,'fits_500MB':size<=500_000_000,
                           'val_auc':scores(({**tables,**ensemble_tables})[name])['macro_auc'],'dependencies':dependencies,
                           'latency':latency,'latency_note':'Per-model measurements only; summed latency is not measured ensemble end-to-end latency.'})
    provenance=root/'manifests/e07_provenance.json'
    if provenance.exists():
        sources=json.loads(provenance.read_text(encoding='utf-8'))
        deployment.append({'candidate':'E07','val_auc':official['E07']['all']['macro_auc'],
                           'dependencies':[{'path':'frozen_reference/'+n,'sha256':v['checkpoint_sha256'],
                                            'bytes':v['checkpoint_bytes']} for n,v in sources.items()],
                           'latency':[]})
    from cxr.batch.deployment import export_reports
    import hashlib
    for candidate in deployment:
        # Independent validation uses the same FP16 raw/EMA checkpoint inference pipeline.
        hashes=sorted(d['sha256'] for d in candidate['dependencies'])
        recipe=hashlib.sha256(json.dumps({'weights':hashes,'precision':'fp16_amp',
            'fusion':candidates.get(candidate['candidate'],candidate['candidate']),
            'study_aggregation':'mean_probability'},sort_keys=True).encode()).hexdigest()
        candidate.update(inference_recipe_sha256=recipe,accuracy_recipe_sha256=recipe,
                         accuracy_checkpoint_sha256=hashes)
    deployment_status=export_reports(root,deployment,plan.get('deployment_protocol',{}))
    from cxr.batch.additional import analyze
    analyze(root,tables,oofs,rows,draws)
    result={'tasks':rows,'failures':failures,'official_val':official,'fold_oof':fold_oof_metrics,
            'full_oof':oof_metrics,'missing_folds':missing,
            'bootstrap_val':boot,'bootstrap_oof':oof_boot,'fold_oof_bootstrap':fold_boot,'model_val_bootstrap':all_val_boot,
            'deployment':deployment,'deployment_compliance':deployment_status,'gpu_epoch_hours':sum(r['epoch_hours'] or 0 for r in rows),
            'limitations':['官方Val多次用于模型与Epoch选择；置信区间仅为探索性证据，非新独立测试结果。',
                          '预设多个候选相互相关，不把未校正的多个置信区间当作独立显著性证据。',
                          'OOF每名患者只使用对应未见患者的Fold模型，不是每患者五模型平均。',
                          '单图/多图类别构成不同；未使用Phase15 Holdout；未读取隐藏测试标签。']}
    if (root/'queue_state.json').exists():
        result['queue_accounting_and_retries']=json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
    import numpy as np
    result['error_correlations']={}
    if 'E17' in ensemble_tables:
        reference=ensemble_tables['E07']; a=align(reference,ensemble_tables['E17'])-reference['y']; b=reference['p']-reference['y']
        result['error_correlations']['E17_vs_E07']=[float(np.corrcoef(a[:,j],b[:,j])[0,1]) if a[:,j].std() and b[:,j].std() else None for j in range(10)]
    write_json(root/'analysis/statistics.json',result)
    from cxr.batch.submission_reporting import generate_test_reports
    generate_test_reports(root, plan, rows, ensemble_tables, deployment, deployment_status)
    perclass=[{'candidate':n,'split':split,'group':g,**r} for split,items in [('official_val',official),('oof',all_oof_metrics)] for n,groups in items.items() for g,m in groups.items() for r in m['per_class']]
    if perclass:csv_rows(root/'analysis/per_class_metrics.csv',perclass)
    task_by_id = {r['model_id']: r for r in rows}
    for reserved in plan.get('conditional_extension', {}).get('reserved_slots', []):
        task_by_id.setdefault(reserved['id'], {
            'model_id': reserved['id'], 'state': 'PENDING', 'best_epoch': None,
            'stop_epoch': None, 'epoch_hours': None, 'weights_bytes': None,
        })
    summary = []
    for candidate in sorted(set(official) | set(all_oof_metrics) | set(task_by_id)):
        base_id = candidate.removesuffix('_EMA')
        task = dict(task_by_id.get(base_id, {}))
        if candidate.endswith('_EMA'):
            run = root/'experiments'/base_id
            metric_path = run/'ema_metrics.json'
            if metric_path.is_file():
                ema_metric = json.loads(metric_path.read_text(encoding='utf-8'))
                task.update(best_epoch=ema_metric.get('best_epoch'), stop_epoch=ema_metric.get('stop_epoch'),
                            epoch_hours=ema_metric.get('resources', {}).get('epoch_hours'),
                            weights_bytes=ema_metric.get('weights_bytes'))
        if candidate in tables and base_id in task_by_id:
            test_file = root/'experiments'/base_id/(
                'ema_test_predictions.csv' if candidate.endswith('_EMA') else 'test_predictions.csv')
            task['test_prediction_status'] = 'probabilities_saved' if test_file.is_file() else 'not_available'
        summary.append(_summary_row(candidate,
            official.get(candidate, {}).get('all'), all_oof_metrics.get(candidate, {}).get('all'), task))
    csv_rows(root/'reports/experiment_summary.csv',summary)
    write_json(root/'reports/experiment_summary.json',result)
    lines=['# Phase17 批量最终/阶段报告','',*result['limitations'],'',
           '| Task | State | Best Epoch | Val AUC | OOF AUC |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["model_id"]} | {r["state"]} | {r["best_epoch"]} | {r["val_auc"]} | {r["oof_auc"]} |')
    lines+=['','## 固定候选','']+[f'- {n}: {m["all"]["macro_auc"]}' for n,m in official.items()]
    lines+=['','## 类别1、4、8（相对E07）','']
    for n,m in official.items():
        if n=='E07':continue
        for j in (0,3,7):
            lines.append(f'- {n}, class {j+1}: ΔAUC={m["all"]["per_class"][j]["auc"]-official["E07"]["all"]["per_class"][j]["auc"]:.6f}')
    lines+=['','## 候选保留建议','', '仅PASS候选可进入部署复核：≤500MB且按官方口径端到端实测<100ms；不自动更换正式基准，不按最高Val分数自动提交。',
             '完整多折模型仅作研究对照；失败/缺失任务不填入估计指标。详见analysis/statistics.json和ensembles/deployment_candidates.json。']
    (root/'reports/phase17_final_report.md').write_text('\n'.join(lines),encoding='utf-8')
    from cxr.batch.direct_delivery import write_delivery_manifest
    write_delivery_manifest(root)
    return result


def training_patients(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return {r['Subject_id'] for r in csv.DictReader(f)}
