"""Confirmation-fold concatenation with patient clusters; never average OOF."""
import json
from pathlib import Path
import numpy as np
from cxr.data.dataset import study_targets_from_rows
from cxr.engine.metrics import binary_auc
from cxr.phase39_support import write_json,write_csv
from cxr.phase42.protocol import verify_receipt
from cxr.regularization_confirmation.core import audit,csvrows
from cxr.regularization_confirmation.support import task_specs


def compare(y,base,candidate,patients,seed=20261108,draws=2000):
    if y.shape!=base.shape or y.shape!=candidate.shape or y.ndim!=2 or y.shape[1]!=10:raise ValueError('Shape mismatch')
    if not np.isin(y,[0,1]).all() or not np.isfinite(base).all() or not np.isfinite(candidate).all():raise ValueError('Invalid values')
    if ((base<0)|(base>1)|(candidate<0)|(candidate>1)).any():raise ValueError('Probability range')
    ba=np.array([binary_auc(y[:,j],base[:,j]) for j in range(10)])
    ca=np.array([binary_auc(y[:,j],candidate[:,j]) for j in range(10)])
    unique=sorted(set(patients));index={p:np.flatnonzero(np.asarray(patients)==p) for p in unique}
    rng=np.random.default_rng(seed);values=[];invalid=0
    for _ in range(draws):
        ix=np.concatenate([index[p] for p in rng.choice(unique,len(unique),replace=True)])
        support=y[ix].sum(0)
        if (support==0).any() or (support==len(ix)).any():invalid+=1;continue
        values.append(np.mean([binary_auc(y[ix,j],candidate[ix,j])-binary_auc(y[ix,j],base[ix,j]) for j in range(10)]))
    return dict(base_auc=float(ba.mean()),challenger_auc=float(ca.mean()),delta=float((ca-ba).mean()),
        ci95=np.quantile(values,[.025,.975]).tolist() if values else None,
        ci975_two_directions=np.quantile(values,[.0125,.9875]).tolist() if values else None,
        class_deltas=(ca-ba).tolist(),positive_studies=y.sum(0).astype(int).tolist(),valid_replicates=len(values),invalid_replicates=invalid,
        invalid_policy='discard entire replicate if any class lacks positives or negatives',studies=len(y),patients=len(unique))


def confirmation_analysis(cfg):
    root=Path(cfg['output_root']);tasks=task_specs(cfg)
    if any(not (root/'experiments'/t['id']/'DONE.json').exists() for t in tasks):
        write_json(root/'reports/confirmation.json',dict(state='PENDING',reason='All predeclared folds must finish'));return
    keys=[];ys=[];bs=[];cs=[];fold_deltas=[];patient_fold={}
    for t in tasks:
        out=root/'experiments'/t['id'];verify_receipt(out)
        cp=root/'configs'/(t['id']+'.json');rc=json.loads(cp.read_text(encoding='utf-8'));groups=audit(rc,cp,out)
        labels={(a.subject_id,a.study_id):a.labels for a in study_targets_from_rows(groups['outer'])}
        maps=[]
        for arm in rc['arms']:
            rows=csvrows(out/arm/'outer_oof_predictions.csv');m={(r['Subject_id'],r['Study_id']):r for r in rows}
            if len(rows)!=len(m) or set(m)!=set(labels):raise ValueError('Duplicate/missing outer predictions')
            maps.append(m)
        for k in sorted(labels):
            if k in keys:raise ValueError('Study repeated across outer folds')
            if k[0] in patient_fold and patient_fold[k[0]]!=t['fold']:raise ValueError('Patient crosses outer folds')
            patient_fold[k[0]]=t['fold'];keys.append(k);ys.append(labels[k])
            bs.append([float(maps[0][k][f'Prob_{j}']) for j in range(1,11)])
            cs.append([float(maps[1][k][f'Prob_{j}']) for j in range(1,11)])
        fold_deltas.append(json.loads((out/'bootstrap_comparison.json').read_text())['delta'])
    report=compare(np.asarray(ys),np.asarray(bs),np.asarray(cs),[k[0] for k in keys])
    required_positive=2
    interval=report['ci95'] if cfg['stage']=='mixup_confirmation' else report['ci975_two_directions']
    passed=report['delta']>=.002 and interval is not None and interval[0]>0 and sum(d>0 for d in fold_deltas)>=required_positive and min(report['class_deltas'])>=-.01
    report.update(scope='CONFIRMATION_FOLDS_ONLY_HOLDOUT_EXCLUDED_PARTIAL_OOF',folds=[t['fold'] for t in tasks],fold_deltas=fold_deltas,
        state='PROMISING_CONFIRMATION' if passed else 'COMPLETED_NO_GAIN',full_fivefold_oof=False,official_val_used=False,
        selection_bias_note='Historical patients were used by other experiments; bootstrap does not include training-seed variance',
        further_training_started=False,T4='T4_UNVERIFIED')
    write_json(root/'reports/confirmation.json',report)
    write_csv(root/'reports/confirmation_per_class.csv',[dict(class_id=j+1,delta=report['class_deltas'][j],positive_studies=report['positive_studies'][j]) for j in range(10)])
