"""Finite, predeclared extra analyses. Never chooses a backbone or starts training."""
import itertools
import numpy as np
from cxr.batch.results import scores, combine, bootstrap, align, grouped
from cxr.batch.plan import write_json


def analyze(root, tables, oofs, rows, draws):
    from cxr.batch.reporting import csv_rows
    folder=root/'analysis'
    csv_rows(folder/'additional_experiments_summary.csv',
             [r for r in rows if r['group'] in ('N','P')], list(rows[0]) if rows else ['model_id','state'])
    ema=[]
    for n in tables:
        if n.endswith('_EMA'):
            raw=n[:-4]
            for split,data in [('val',tables),('fold_oof',oofs)]:
                if raw not in data or n not in data: continue
                b=bootstrap({raw:data[raw],n:data[n]},[(n,raw)],draws)
                ema.append({'model_id':raw,'split':split,'raw_auc':scores(data[raw])['macro_auc'],
                            'ema_auc':scores(data[n])['macro_auc'],'bootstrap':b})
    csv_rows(folder/'ema_comparison.csv',ema,['model_id','split','raw_auc','ema_auc','bootstrap'])
    pilots=[]; reference_ids=['D17_F01','C17_F01','S17_F01']
    if all(n in oofs for n in reference_ids):
        reference=combine([oofs[n] for n in reference_ids],[.25,.25,.5])
        for pilot in ('CS17_F01','NM17_F01'):
            if pilot not in oofs:continue
            t=oofs[pilot]; align(reference,t)  # exact Study/Subject/labels/image-count agreement
            fusion=combine([reference,t],[.5,.5])
            comparison={n:oofs[n] for n in reference_ids}
            comparison.update(reference=reference,pilot=t,reference_plus_pilot=fusion)
            b=bootstrap(comparison,[('pilot','reference'),('reference_plus_pilot','reference')]+
                        [('pilot',n) for n in reference_ids],draws)
            write_json(folder/(pilot+'_same_fold_comparison.json'),
                       {'scope':'Fold1 only; exploratory; not full fivefold OOF',
                        'metrics':{n:grouped(v) for n,v in comparison.items()},'bootstrap':b})
            pilots.append({'model_id':pilot,'studies':len(t['keys']),
                           'patients':len({s for s,_ in t['keys']}),'oof_auc':scores(t)['macro_auc'],
                           'reference_auc':scores(reference)['macro_auc'],'fixed_fusion_auc':scores(fusion)['macro_auc'],
                           'conditional_extension':'DISABLED_NO_FROZEN_CRITERIA_OR_AUTHORIZED_BUDGET'})
    csv_rows(folder/'backbone_pilot_comparison.csv',pilots,
             ['model_id','studies','patients','oof_auc','reference_auc','fixed_fusion_auc','conditional_extension'])
    correlations=[]
    for a,b in itertools.combinations(sorted(tables),2):
        pa=tables[a]['p'];pb=align(tables[a],tables[b])
        for j in range(10):
            corr=float(np.corrcoef(pa[:,j],pb[:,j])[0,1]) if pa[:,j].std() and pb[:,j].std() else None
            correlations.append({'split':'official_val','model_a':a,'model_b':b,'class_id':j+1,'probability_correlation':corr})
    csv_rows(folder/'architecture_complementarity.csv',correlations,
             ['split','model_a','model_b','class_id','probability_correlation'])
