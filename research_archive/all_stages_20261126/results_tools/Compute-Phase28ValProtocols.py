"""Distinguish mean of five per-fold AUCs from AUC of averaged probabilities."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def helper():
    path=Path(__file__).with_name('Analyze-Phase27.py')
    spec=importlib.util.spec_from_file_location('phase27_val_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def local_probs(path,keys):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rr=list(csv.DictReader(f))
    mapping={(r['Subject_id'],r['Study_id']):[float(r[f'probability_{i}']) for i in range(1,11)] for r in rr}
    if len(rr)!=859 or set(mapping)!=set(keys):
        raise ValueError(f'Local Val Study mismatch: {path}')
    return np.asarray([mapping[k] for k in keys])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--friend-root',required=True,type=Path)
    ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    m=helper()
    keys,y=m.labels(a.friend_root/'manifests/val_labels.csv',859)
    local_manifest=json.loads((a.project/'outputs/analysis/phase17_final_handoff_20261023/freeze_manifest.json').read_text(encoding='utf-8'))
    local={(x['branch'],int(x['fold'])):x for x in local_manifest['models']}
    records=[]
    for source in ('FRIEND_E17','LOCAL_E17','N17_EMA'):
        fold_scores=[]
        averaged=np.zeros((859,10),dtype=float)
        for fold in range(1,6):
            if source=='N17_EMA':
                path=a.friend_root/'experiments'/f'N17_F{fold:02d}'/'ema_val_predictions.csv'
                values=m.probs(path,keys)
            else:
                values=np.zeros((859,10),dtype=float)
                for branch,weight in [('D17',.25),('C17',.25),('S17',.5)]:
                    if source=='FRIEND_E17':
                        path=a.friend_root/'experiments'/f'{branch}_F{fold:02d}'/'val_predictions.csv'
                        piece=m.probs(path,keys)
                    else:
                        info=local[(branch,fold)]
                        path=a.project/info['artifact_paths']['validation_analysis/study_predictions.csv']
                        if sha(path)!=info['val_prediction_sha256']:
                            raise ValueError(f'Local frozen Val SHA differs: {path}')
                        piece=local_probs(path,keys)
                    values+=weight*piece
            fold_scores.append(m.score(y,values)[0])
            averaged+=values/5
        recomputed_auc=m.score(y,averaged)[0]
        expected={'FRIEND_E17':a.friend_root/'validation/E17_val_probabilities.csv',
                  'LOCAL_E17':a.project/'outputs/analysis/phase17_five_fold_final_20261023/e17_val_fivefold.csv',
                  'N17_EMA':a.friend_root/'validation/N17_EMA_val_ensemble.csv'}[source]
        reference=m.probs(expected,keys) if source!='LOCAL_E17' else local_probs(expected,keys)
        maxdiff=float(np.max(np.abs(reference-averaged)))
        if maxdiff>1e-12:
            raise ValueError(f'Val fivefold probability formula differs: {source}: {maxdiff}')
        records.append({'model':source,'per_fold_val_auc':';'.join(str(v) for v in fold_scores),
                        'mean_of_fold_aucs':float(np.mean(fold_scores)),
                        'auc_of_frozen_fivefold_probabilities':m.score(y,reference)[0],
                        'recomputed_sum_order_auc':recomputed_auc,
                        'max_abs_probability_diff_from_frozen':maxdiff})
    with a.out.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
