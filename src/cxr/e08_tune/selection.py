"""Predeclared EMA-only selection and ID-aligned five-fold analysis."""
import json
from pathlib import Path
from cxr.batch.plan import digest,write_json,ROOT
from cxr.batch.results import read_table,concatenate_oof,combine,scores,bootstrap,align
from cxr.e08_tune.results import verify_done,csv_write

def choose(records,criteria):
    eligible=[]
    for variant,metrics in records.items():
        raw=metrics['variants']['ema'];comparison=raw['oof_comparison']
        if 'differences' not in comparison:continue
        diff=comparison['differences']['pilot_minus_baseline'];reg=raw['followup_evidence']['worst_class_delta']
        if diff['point']>=criteria['min_delta'] and diff['ci95'][0]>criteria['ci_lower_must_exceed'] and reg>=-criteria['max_class_regression']:
            eligible.append((variant,diff['point'],diff['ci95'][0]))
    return [v for v,_,_ in sorted(eligible,key=lambda r:(-r[1],-r[2],r[0]))[:criteria['max_variants']]]

def select_pilots(root,plan):
    root=Path(root);state=json.loads((root/'pilot/queue_state.json').read_text())
    pilots=[t for t in plan['tasks'] if t['fold']==1]
    # Incomplete comparison never launches followup; no silent dropping failed pilots.
    if any(state['tasks'][t['id']]['state']!='DONE' for t in pilots):
        write_json(root/'reports/extension_status.json',{'state':'BLOCKED','reason':'All five pilot comparisons must be complete'})
        return None
    records={};provenance={}
    for t in pilots:
        run=root/'pilot/experiments'/t['id'];verify_done(run)
        records[t['variant']]=json.loads((run/'metrics.json').read_text())
        provenance[t['id']]=digest(run/'metrics.json')
    selected=choose(records,plan['selection'])
    result={'selected':selected,'source_metrics_sha256':provenance,'rules':plan['selection'],
            'scope':'Exploratory Fold1 selection; no independent confirmation claim'}
    path=root/'manifests/selection.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Frozen selection differs; no reselection')
    if not path.exists():write_json(path,result)
    return selected

def complete_folds(root,plan,selected,baseline_root):
    root,baseline_root=Path(root),Path(baseline_root)
    audit=json.loads((ROOT/'docs/e08_tune/source_audit.json').read_text(encoding='utf-8'))
    rows=[];details={}
    for variant in selected:
        tasks=sorted([t for t in plan['tasks'] if t['variant']==variant],key=lambda t:t['fold'])
        runs=[root/('pilot' if t['fold']==1 else 'expansion')/'experiments'/t['id'] for t in tasks]
        if not all((r/'DONE.json').exists() for r in runs):
            rows.append({'Variant':variant,'Status':'INCOMPLETE','OOF_AUC':None,'Val_Ensemble_AUC':None});continue
        for r in runs:verify_done(r)
        pilot_oof=[read_table(r/'ema_oof_analysis/study_predictions.csv') for r in runs]
        oof=concatenate_oof(pilot_oof)
        if len(oof['keys'])!=18513 or len({s for s,_ in oof['keys']})!=11055:raise ValueError('Complete OOF coverage mismatch')
        val=combine([read_table(r/'ema_validation_analysis/study_predictions.csv') for r in runs],[.2]*5)
        if len(val['keys'])!=859:raise ValueError('Official Val coverage differs')
        baseline_tables=[];e17_tables=[]
        for f in range(1,6):
            nid=f'N17_F{f:02d}';base=baseline_root/'experiments'/nid
            rel='ema_oof_analysis/study_predictions.csv'
            if digest(base/rel)!=audit['baselines'][nid]['sha256'][rel]:raise ValueError('N17 baseline changed')
            baseline_tables.append(read_table(base/rel))
            members=[]
            for branch in ('D','C','S'):
                mid=f'{branch}17_F{f:02d}';p=baseline_root/'experiments'/mid/'oof_analysis/study_predictions.csv'
                if digest(p)!=audit['e17_oof_sources'][mid]['prediction_sha256']:raise ValueError('E17 baseline changed')
                members.append(read_table(p))
            e17_tables.append(combine(members,[.25,.25,.5]))
            align(baseline_tables[-1],pilot_oof[f-1]) # same patient/Study/labels/image counts in corresponding fold
        base=concatenate_oof(baseline_tables);e17=concatenate_oof(e17_tables)
        from cxr.batch.artifacts import write_probabilities
        write_probabilities(root/f'oof/{variant}_ema_oof_all.csv',oof,{'tasks':[t['id'] for t in tasks],
             'fold_hashes':plan['fold_hashes'],'scope':'COMPLETE_OOF; patient-held-out models, hyperparameters selected using Fold1'})
        write_probabilities(root/f'validation/{variant}_ema_5fold.csv',val,{'formula':'equal five-fold Study probability mean'})
        scope={'pilot':oof,'N17_EMA':base,'E17':e17}
        uncertainty=bootstrap(scope,[('pilot','N17_EMA'),('pilot','E17')],draws=plan['selection']['bootstrap_valid_draws'],seed=42)
        confirmation={'pilot':concatenate_oof(pilot_oof[1:]),'N17_EMA':concatenate_oof(baseline_tables[1:])}
        confirmation_ci=bootstrap(confirmation,[('pilot','N17_EMA')],draws=plan['selection']['bootstrap_valid_draws'],seed=42)
        size=sum((r/'ema_best_model.pth').stat().st_size+(r/'resolved_config.yaml').stat().st_size for r in runs)
        details[variant]={'oof':scores(oof),'val_ensemble':scores(val),'full_oof_bootstrap':uncertainty,
             'folds2to5_confirmation':confirmation_ci,'weight_and_config_bytes':size,
             'deployment':'SIZE_FAIL' if size>500000000 else 'LATENCY_UNVERIFIED',
             'limitation':'Fold1 selected hyperparameters. Full OOF is not a fresh confirmatory test. E08 Train predictions never used as OOF.'}
        e03_bytes=audit.get('e08',{}).get('e03_checkpoint_bytes')
        details[variant]['fixed_e08_replacement']={
            'formula':'0.5 E03 + 0.5 new EfficientNet EMA component; weights must not be searched',
            'five_fold_weight_bytes_lower_bound':size+e03_bytes if e03_bytes is not None else None,
            'status':('SIZE_FAIL' if size+e03_bytes>500000000 else 'LATENCY_UNVERIFIED') if e03_bytes is not None else 'UNAVAILABLE',
            'submission':'NOT_GENERATED; a five-model component is not a validated single-model E08 replacement'}
        rows.append({'Variant':variant,'Status':'COMPLETE_OOF','OOF_AUC':scores(oof)['macro_auc'],
            'Val_Ensemble_AUC':scores(val)['macro_auc']})
    csv_write(root/'reports/five_fold_summary.csv',rows)
    write_json(root/'reports/five_fold_analysis.json',details)
    return details
