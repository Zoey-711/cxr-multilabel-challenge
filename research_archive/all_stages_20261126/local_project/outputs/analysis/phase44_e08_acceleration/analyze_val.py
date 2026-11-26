"""CPU-only paired probability/AUC audit of independent ONNX candidate."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
KEY=['Subject_id','Study_id']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=('fp16','fp32'),default='fp16')
    precision=parser.parse_args().precision
    truth=pd.read_csv(ROOT/'batch_assets/val_labels.csv')
    old=pd.read_csv(ROOT/'outputs/submissions/E08_EMA_20261020/val_predictions.csv')
    new=pd.read_csv(HERE/('onnx_val_predictions.csv' if precision=='fp16' else 'onnx_fp32_val_predictions.csv'))
    pcols=[f'Prob_{j}' for j in range(1,11)]
    ocols=[f'probability_{j}' for j in range(1,11)]
    if list(new.columns)!=KEY+pcols or list(old.columns)!=KEY+ocols:raise ValueError('Category mapping differs')
    if len(truth)!=859 or len(old)!=859 or len(new)!=859 or any(x.duplicated(KEY).any() for x in (truth,old,new)):
        raise ValueError('Study coverage/duplicates')
    keys=set(map(tuple,truth[KEY].to_numpy()))
    if keys!=set(map(tuple,old[KEY].to_numpy())) or keys!=set(map(tuple,new[KEY].to_numpy())):
        raise ValueError('Study ID mismatch')
    data=truth.merge(old,on=KEY,validate='one_to_one').merge(new,on=KEY,validate='one_to_one')
    y=data[[f'Target_{j}' for j in range(1,11)]].to_numpy(dtype=int)
    a=data[ocols].to_numpy(dtype=float);b=data[pcols].to_numpy(dtype=float)
    if not np.isfinite(b).all() or (b<0).any() or (b>1).any():raise ValueError('Probability invalid')
    delta=np.abs(a-b)
    before=[];after=[];rows=[]
    for j in range(10):
        x=roc_auc_score(y[:,j],a[:,j]);z=roc_auc_score(y[:,j],b[:,j]);before.append(x);after.append(z)
        rows.append(dict(class_id=j+1,positive_studies=int(y[:,j].sum()),auc_frozen=x,auc_onnx=z,
                         auc_delta=z-x,max_abs_probability_diff=delta[:,j].max(),
                         mean_abs_probability_diff=delta[:,j].mean(),p99_abs_probability_diff=np.percentile(delta[:,j],99)))
    pd.DataFrame(rows).to_csv(HERE/f'probability_consistency_{precision}.csv',index=False)
    patients,inverse=np.unique(data.Subject_id.to_numpy(),return_inverse=True)
    groups=[[np.unique(p[:,j],return_inverse=True)[1] for j in range(10)] for p in (a,b)]
    def score(g,w):
        values=[]
        for j,bin_id in enumerate(g):
            pos=np.bincount(bin_id,weights=w*y[:,j]);neg=np.bincount(bin_id,weights=w*(1-y[:,j]))
            if not pos.sum() or not neg.sum():raise ValueError('Undefined AUC')
            values.append(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
        return np.mean(values)
    rng=np.random.default_rng(4408);draw=[];invalid=0
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))[inverse]
        try:draw.append(float(score(groups[1],w)-score(groups[0],w)))
        except ValueError:invalid+=1
    result=dict(studies=len(data),patients=len(patients),max_abs_diff=float(delta.max()),
                mean_abs_diff=float(delta.mean()),p99_abs_diff=float(np.percentile(delta,99)),
                frozen_macro_auc=float(np.mean(before)),onnx_macro_auc=float(np.mean(after)),
                macro_auc_delta=float(np.mean(after)-np.mean(before)),bootstrap_ci95=np.percentile(draw,[2.5,97.5]).tolist(),
                bootstrap_seed=4408,bootstrap_valid=len(draw),bootstrap_invalid=invalid,
                tolerance_1e_4_status='PASS' if delta.max()<=1e-4 else 'FAIL',
                protocol='859 same Val Studies; patient-cluster paired bootstrap, invalid whole draw discarded')
    (HERE/f'onnx_{precision}_val_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
