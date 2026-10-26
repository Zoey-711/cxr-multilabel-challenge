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
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


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
                       epoch_hours=metric['resources']['epoch_hours'],weights_bytes=metric['weights_bytes'])
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
    for branch in ('D17','C17','S17'):
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
    official={n:grouped(t) for n,t in ensemble_tables.items()}
    pairs=[(n,'E07') for n in ensemble_tables if n!='E07']
    boot=bootstrap(ensemble_tables,pairs,draws) if len(ensemble_tables)>1 else None
    oof_metrics={n:grouped(t) for n,t in complete_oof.items()}
    oof_boot=bootstrap(complete_oof,[("E17",n) for n in ('D5','C5','S5')],draws) if 'E17' in complete_oof else None
    fold_boot={}
    for i in range(1,6):
        fold_tables={n:oofs[n] for n in (f'D17_F{i:02d}',f'C17_F{i:02d}',f'S17_F{i:02d}') if n in oofs}
        if fold_tables:
            fold_boot[str(i)]=bootstrap(fold_tables,[],draws)
    all_val_boot=bootstrap(tables,[(n,n[:-4]) for n in tables if n.endswith('_EMA')],draws) if tables else None
    deployment=[]
    for name in [n for n in ensemble_tables if n!='E07']:
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
            f=p.parent/'latency.json'
            if f.exists():latency.append(json.loads(f.read_text(encoding='utf-8')))
        deployment.append({'candidate':name,'bytes':size,'fits_500MB':size<=500_000_000,
                           'val_auc':official[name]['all']['macro_auc'],'dependencies':dependencies,
                           'latency':latency,'latency_note':'Per-model measurements only; summed latency is not measured ensemble end-to-end latency.'})
    write_json(root/'ensembles/deployment_candidates.json',deployment)
    csv_rows(root/'ensembles/deployment_candidates.csv',
             [{k:r[k] for k in ('candidate','bytes','fits_500MB','val_auc')} for r in deployment],
             ['candidate','bytes','fits_500MB','val_auc'])
    (root/'ensembles/deployment_report.md').write_text('# 部署预算\n\n仅统计纯推理权重，不含resume。完整15模型不能默认部署。\n\n'+
        '\n'.join(f'- {r["candidate"]}: {r["bytes"]} bytes; <=500MB: {r["fits_500MB"]}' for r in deployment)+
        '\n\n延迟设备与单模型实测见 latency 字段；未在官方T4实测，不宣称满足100ms。',encoding='utf-8')
    result={'tasks':rows,'failures':failures,'official_val':official,'full_oof':oof_metrics,'missing_folds':missing,
            'bootstrap_val':boot,'bootstrap_oof':oof_boot,'fold_oof_bootstrap':fold_boot,'model_val_bootstrap':all_val_boot,
            'deployment':deployment,'gpu_epoch_hours':sum(r['epoch_hours'] or 0 for r in rows),
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
    perclass=[{'candidate':n,'split':split,'group':g,**r} for split,items in [('official_val',official),('oof',oof_metrics)] for n,groups in items.items() for g,m in groups.items() for r in m['per_class']]
    if perclass:csv_rows(root/'analysis/per_class_metrics.csv',perclass)
    csv_rows(root/'reports/experiment_summary.csv',rows)
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
    lines+=['','## 候选保留建议','', '保留预先指定且≤500MB的候选供人工复核；不自动更换正式基准，不按最高Val分数自动提交。',
             '完整15模型仅研究对照；失败/缺失任务不填入估计指标。详见analysis/statistics.json和ensembles/deployment_candidates.json。']
    (root/'reports/phase17_final_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def training_patients(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return {r['Subject_id'] for r in csv.DictReader(f)}
