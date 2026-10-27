"""Fold2-only selection; excluded-Fold2 confirmation; Val never selects heads."""
import json
from pathlib import Path
import numpy as np
from cxr.data.archive import read_split_rows
from cxr.data.dataset import study_targets_from_rows
from cxr.data.patient_folds import read_index
from cxr.phase39_support import write_json,write_csv
from cxr.phase41.analysis import matrix,auc_summary
from cxr.phase42.protocol import ROOT,digest,fingerprint,verify_baseline,verify_receipt


def weighted_auc(y,p,weights):
    _,bins=np.unique(p,return_inverse=True)
    pos=np.bincount(bins,weights=weights*y);neg=np.bincount(bins,weights=weights*(1-y))
    if pos.sum()==0 or neg.sum()==0:raise ValueError('Bootstrap class lacks positive/negative support')
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))


def paired_bootstrap(truth,base,candidate,keys,draws=2000,seed=4202):
    y=np.asarray([truth[k] for k in keys]);a=np.asarray([base[k] for k in keys]);b=np.asarray([candidate[k] for k in keys])
    patients,inverse=np.unique([k[0] for k in keys],return_inverse=True)
    # Precompute probability tie groups; patient weights exactly reproduce a
    # cluster bootstrap with every Study of the sampled patient duplicated.
    prepared=[]
    for p in (a,b):prepared.append([np.unique(p[:,j],return_inverse=True)[1] for j in range(10)])
    def score(groups,w):
        scores=[]
        for j,g in enumerate(groups):
            pos=np.bincount(g,weights=w*y[:,j]);neg=np.bincount(g,weights=w*(1-y[:,j]))
            if not pos.sum() or not neg.sum():raise ValueError('Missing class')
            scores.append(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
        return float(np.mean(scores))
    point=score(prepared[1],np.ones(len(keys)))-score(prepared[0],np.ones(len(keys)))
    rng=np.random.default_rng(seed);deltas=[];invalid=0
    for _ in range(draws):
        weights=np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))[inverse]
        try:deltas.append(score(prepared[1],weights)-score(prepared[0],weights))
        except ValueError:invalid+=1
    if len(deltas)<.9*draws:raise ValueError('Too many invalid bootstrap draws')
    delta=np.asarray(deltas)
    return dict(delta=point,ci95=np.percentile(delta,[2.5,97.5]).tolist(),requested=draws,valid=len(deltas),invalid=invalid,
        seed=seed,patients=len(patients),studies=len(keys),invalid_policy='Discard whole draw if any class is undefined; no macro over fewer classes',
        centered_one_sided_p=float((1+np.sum(delta-point>=point))/(1+len(delta))),
        p_method='Approximate centered paired-cluster bootstrap; exploratory, not an independent confirmatory test')


def select(rows,plan):
    """Frozen deterministic ranking; absent/failed candidates stay in family size."""
    rule=plan['selection'];family=len(plan['candidates'])
    ordered=sorted(rows,key=lambda r:(r.get('p',1.),r['candidate']))
    previous=0.
    for i,row in enumerate(ordered):
        previous=max(previous,min(1.,(family-i)*row.get('p',1.)))
        row['holm_approx_p']=previous
        row['eligible']=bool(row.get('state')=='DONE' and row['delta']>=rule['min_delta'] and
            row['ci_lower']>rule['min_ci_lower'] and previous<=rule['holm_alpha'] and
            row['worst_class_delta']>=rule['min_class_delta'] and row['mac_ratio']<=rule['max_mac_ratio'] and
            row['weight_bytes']<=rule['max_weight_bytes'])
    selected=[];groups=set()
    for row in sorted(ordered,key=lambda r:(-(r['delta'] if r.get('delta') is not None else -999),r.get('mac_ratio',999),r['candidate'])):
        if row['eligible'] and row['mechanism'] not in groups and len(selected)<rule['max_candidates']:
            selected.append(row['candidate']);groups.add(row['mechanism'])
    return selected


def labels_and_folds(cfg,plan):
    for name,sha in plan['fold_sha256'].items():
        if digest(Path(cfg['protocol_root'])/name)!=sha:raise ValueError('Fold index hash changed')
    train=read_split_rows(Path(cfg['data_root']),'train');val=read_split_rows(Path(cfg['data_root']),'val')
    truth={(s.subject_id,s.study_id):s.labels for s in study_targets_from_rows(train)}
    vtruth={(s.subject_id,s.study_id):s.labels for s in study_targets_from_rows(val)}
    entries=read_index(Path(cfg['protocol_root'])/'study_folds.csv')
    folds={(r['Subject_id'],r['Study_id']):int(r['fold']) for r in entries}
    patients={r['Subject_id']:int(r['fold']) for r in read_index(Path(cfg['protocol_root'])/'patient_folds.csv')}
    if len(folds)!=len(entries) or len(folds)!=18513 or len(patients)!=11055 or set(folds)!=set(truth):
        raise ValueError('Full Train IDs do not match frozen folds')
    if any(patients[k[0]]!=f for k,f in folds.items()):raise ValueError('Patient crosses folds')
    if len(vtruth)!=859 or len({k[0] for k in vtruth})!=228:raise ValueError('Official Val coverage differs')
    return truth,vtruth,folds


def candidate_predictions(root,plan,candidate,fold,keys,valkeys):
    import yaml
    from cxr.phase42.protocol import resolved_config,task
    folder=Path(root)/'experiments'/f'P42_{candidate}_F{fold:02d}'
    receipt=verify_receipt(folder,require_weights=False)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    actual=yaml.safe_load((folder/'resolved_config.yaml').read_text(encoding='utf-8'))
    if actual!=resolved_config(plan,task(plan,candidate,fold)) or manifest['config_sha256']!=digest(folder/'resolved_config.yaml'):
        raise ValueError('Resolved candidate protocol changed')
    if (manifest['variant'],manifest['fold'],manifest['fold_sha256'])!=(candidate,fold,plan['fold_sha256']):
        raise ValueError('Model/Fold provenance mismatch')
    if manifest['official_val_used_for_training'] or manifest['test_labels_used']:raise ValueError('Prohibited model selection')
    return folder,matrix(folder/'oof_predictions.csv',keys),matrix(folder/'val_predictions.csv',valkeys),receipt


def pilot(root,plan,cfg,queue_state):
    root=Path(root);truth,vtruth,folds=labels_and_folds(cfg,plan)
    keys=sorted(k for k,f in folds.items() if f==plan['pilot_fold']);expected=set(keys)
    baseline,_=verify_baseline(plan,cfg,plan['pilot_fold'],require_weights=False)
    base=matrix(baseline/'oof_predictions.csv',expected);base_metrics=auc_summary(truth,base,keys)
    costs=json.loads((ROOT/'docs/phase42/model_costs.json').read_text())
    costmap={r['candidate']:r for r in costs['candidates']}
    rows=[];details={};sources={};perclass=[];timings=[]
    baseline_timing=json.loads((root/'reports/baseline_reference/inference_benchmark.json').read_text())
    for spec in plan['candidates']:
        cid=spec['id'];tid=f'P42_{cid}_F02';status=queue_state['tasks'][tid]['state']
        row=dict(candidate=cid,mechanism=spec['mechanism'],state=status,delta=None,ci_lower=None,ci_upper=None,
                 p=1.,worst_class_delta=None,mac_ratio=costmap[cid]['mac_ratio'],weight_bytes=None)
        if status=='DONE':
            folder,pred,_,receipt=candidate_predictions(root,plan,cid,2,expected,set(vtruth))
            metrics=auc_summary(truth,pred,keys)
            boot=paired_bootstrap(truth,base,pred,keys,plan['selection']['draws'],plan['selection']['seed'])
            size=json.loads((folder/'model_size.json').read_text())['inference_weights_and_config_bytes']
            differences=np.asarray(metrics['per_class_auc'])-base_metrics['per_class_auc']
            row.update(delta=boot['delta'],ci_lower=boot['ci95'][0],ci_upper=boot['ci95'][1],
                       p=boot['centered_one_sided_p'],worst_class_delta=float(differences.min()),weight_bytes=size)
            details[cid]=dict(oof=metrics,bootstrap=boot)
            sources[tid]=fingerprint(receipt)
            timing=json.loads((folder/'inference_benchmark.json').read_text())
            timings.append(dict(candidate=cid,mean_ms=timing['mean_ms'],baseline_mean_ms=baseline_timing['mean_ms'],
                delta_ms=timing['mean_ms']-baseline_timing['mean_ms'],device=timing['gpu'],
                limitation='Same input/protocol; server contention may differ; not a selection criterion',status='T4_UNVERIFIED'))
            perclass.extend(dict(candidate=cid,class_id=j+1,auc=metrics['per_class_auc'][j],
                base_auc=base_metrics['per_class_auc'][j],delta=float(differences[j]),positive_studies=metrics['positive_studies'][j]) for j in range(10))
        rows.append(row)
    # Missing task results cannot be ignored to trigger premature winner selection.
    terminal=all(r['state'] in ('DONE','FAILED','BLOCKED') for r in rows)
    selected=select(rows,plan) if terminal else []
    result=dict(state='COMPLETE' if terminal else 'PENDING',selected=selected,pilot_fold=2,
        baseline=base_metrics,candidates=details,source_receipts=sources,plan_sha256=fingerprint(plan),
        limitation='19 correlated comparisons, repeated epoch selection and previously studied cohort. Approximate Holm bootstrap screening is exploratory. Val/Test/leaderboard excluded from selection.')
    if terminal:
        sealed=root/'reports/selection.json'
        if sealed.exists() and json.loads(sealed.read_text())!=result:raise ValueError('Sealed selection changed; do not retune')
        if not sealed.exists():write_json(sealed,result)
    write_json(root/'reports/pilot_analysis.json',result)
    write_csv(root/'reports/pilot_comparison.csv',rows)
    if perclass:write_csv(root/'analysis/pilot_per_class_auc.csv',perclass)
    if timings:write_csv(root/'reports/latency_comparison.csv',timings)
    return result


def confirmation(root,plan,cfg,selected):
    root=Path(root);truth,vtruth,folds=labels_and_folds(cfg,plan)
    baselines={};baseline_val=[]
    for f in range(1,6):
        folder,_=verify_baseline(plan,cfg,f,require_weights=False)
        baselines.update(matrix(folder/'oof_predictions.csv',{k for k,v in folds.items() if v==f}))
        baseline_val.append(matrix(folder/'val_predictions.csv',set(vtruth)))
    keys=sorted(truth);held=[k for k in keys if folds[k]!=2]
    baseline_val={k:np.mean([p[k] for p in baseline_val],axis=0) for k in vtruth}
    results={};classes=[];suggest=[]
    for cid in selected:
        pred={};parts=[];size=0;missing=[];source=[]
        for fold in range(1,6):
            folder=root/'experiments'/f'P42_{cid}_F{fold:02d}'
            if not (folder/'DONE.json').exists():missing.append(fold);continue
            folder,part,val,receipt=candidate_predictions(root,plan,cid,fold,{k for k,f in folds.items() if f==fold},set(vtruth))
            if set(pred)&set(part):raise ValueError('Duplicate OOF Study')
            pred.update(part);parts.append(val)
            size+=json.loads((folder/'model_size.json').read_text())['inference_weights_and_config_bytes']
            source.append(dict(fold=fold,receipt_sha256=fingerprint(receipt)))
        if missing:
            results[cid]=dict(state='PARTIAL_OOF',missing_folds=missing,studies=len(pred));continue
        if len(pred)!=18513 or len({k[0] for k in pred})!=11055:raise ValueError('Full OOF coverage incomplete')
        full=auc_summary(truth,pred,keys);hold=auc_summary(truth,pred,held)
        full_boot=paired_bootstrap(truth,baselines,pred,keys)
        confirm_boot=paired_bootstrap(truth,baselines,pred,held)
        val={k:np.mean([p[k] for p in parts],axis=0) for k in vtruth}
        vm=auc_summary(vtruth,val,sorted(vtruth))
        worst=float(np.min(np.asarray(hold['per_class_auc'])-auc_summary(truth,baselines,held)['per_class_auc']))
        rule=plan['confirmation']
        promising=confirm_boot['delta']>=rule['min_delta'] and confirm_boot['ci95'][0]>rule['min_ci_lower'] and worst>=rule['min_class_delta']
        result=dict(state='PROMISING_CROSSFOLD' if promising else 'COMPLETED_NO_GAIN',full_oof=full,
            confirmation_excluding_fold2=hold,full_bootstrap=full_boot,confirmation_bootstrap=confirm_boot,
            val_fivefold_probability_mean=vm,val_minus_e08_reference=vm['macro_auc']-.8116686461,
            e08_comparison='Scalar reference only; no paired E08 claims without its aligned frozen CSV',
            fivefold_weights_and_configs_bytes=size,size_status='PASS' if size<=500000000 else 'SIZE_FAIL',
            latency_status='T4_UNVERIFIED',source=source)
        results[cid]=result
        if promising:suggest.append((cid,confirm_boot['delta']))
        for name,table in (('oof',pred),('validation',val)):
            write_csv(root/name/f'{cid}_fivefold.csv',[dict(Subject_id=k[0],Study_id=k[1],**{f'Prob_{j+1}':float(table[k][j]) for j in range(10)}) for k in sorted(table)])
        classes.extend(dict(candidate=cid,class_id=j+1,auc=full['per_class_auc'][j],positive_studies=full['positive_studies'][j],
                            confirmation_auc=hold['per_class_auc'][j]) for j in range(10))
    report=dict(candidates=results,third_stage_recommendations=[c for c,_ in sorted(suggest,key=lambda v:(-v[1],v[0]))[:2]],
        third_stage_training='NOT_IMPLEMENTED_NOT_AUTHORIZED: requires a separately frozen full-Train/student protocol',
        baseline_val=auc_summary(vtruth,baseline_val,sorted(vtruth)),
        limitation='Excluded Fold2 confirms structure selection only; per-Fold checkpoint early stopping still uses that Fold. Historical cohort reuse and up to4 confirmation comparisons remain exploratory.')
    write_json(root/'reports/confirmation_analysis.json',report)
    if classes:write_csv(root/'analysis/confirmation_per_class_auc.csv',classes)
    return report
