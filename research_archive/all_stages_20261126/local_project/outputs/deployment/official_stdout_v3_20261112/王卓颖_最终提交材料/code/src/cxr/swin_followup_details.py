"""Verify follow-up sources and describe ranking changes without Test access."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from cxr.data.archive import read_split_rows
from cxr.fusion_opportunity_audit import auc, rows, savecsv
from cxr.swin_followup_audit import sha, load_prediction, bootstrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    cfg=json.loads(Path(ap.parse_args().config).read_text(encoding='utf-8'))
    root=Path(cfg['project']);out=Path(cfg['output']);friend=Path(cfg['friend17'])
    prior=root/'outputs/analysis/fusion_opportunities_20261110'
    z=np.load(prior/'screen_arrays.npz');keys=[tuple(k) for k in z['keys']];y=z['y']
    meta=read_split_rows(Path(json.loads((prior/'config.json').read_text(encoding='utf-8-sig'))['archive']),'val')
    counts=Counter((r['Subject_id'],r['Study_id']) for r in meta)
    truth={}
    for r in meta:
        k=r['Subject_id'],r['Study_id'];target=[int(str(j) in r['Predict_class'].split(';')) for j in range(1,11)]
        if k in truth:assert truth[k]==target
        truth[k]=target
    assert set(truth)==set(keys) and np.array_equal(y,np.array([truth[k] for k in keys]))
    inventory={r['model_id']:r for r in rows(prior/'model_inventory.csv')}
    def get(mid):return load_prediction(Path(inventory[mid]['path']),keys)
    e08=z['e08'];s=get('M027');n=get('M016');champ=.75*e08+.25*s
    h=get('M178');lr=get('M059')
    models={'CHAMPION_VS_E08':(e08,champ),'LRHIGH_VS_CHAMPION':(champ,champ-.375*n+.375*lr),
            'H17_SHARE_VS_CHAMPION':(champ,champ-.1875*n+.1875*h)}
    pair=[];strata=[]
    for name,(a,b) in models.items():
        for j in range(10):
            pos=y[:,j]==1;neg=~pos
            da=a[pos,j,None]-a[neg,j][None,:];db=b[pos,j,None]-b[neg,j][None,:]
            ca=(da>0)+.5*(da==0);cb=(db>0)+.5*(db==0)
            pair.append(dict(comparison=name,class_id=j+1,positive_studies=int(pos.sum()),negative_studies=int(neg.sum()),
                             improved_pair_credit=float(np.maximum(cb-ca,0).sum()),worsened_pair_credit=float(np.maximum(ca-cb,0).sum()),
                             auc_delta=float((cb-ca).mean())))
        for group,mask in [('SINGLE_IMAGE',np.array([counts[k]==1 for k in keys])),('MULTI_IMAGE',np.array([counts[k]>1 for k in keys]))]:
            support=y[mask].sum(0);valid=((support>0)&(support<mask.sum())).all()
            strata.append(dict(comparison=name,group=group,studies=int(mask.sum()),patients=len({k[0] for k,m in zip(keys,mask) if m}),
                baseline_auc=float(auc(y[mask],a[mask]).mean()) if valid else None,
                candidate_auc=float(auc(y[mask],b[mask]).mean()) if valid else None,
                delta=float((auc(y[mask],b[mask])-auc(y[mask],a[mask])).mean()) if valid else None))
    savecsv(out/'ranking_pair_changes.csv',pair);savecsv(out/'view_count_strata.csv',strata)
    # Directly reconstruct remote OOF from per-task predictions and prove equality.
    fm={(r['Subject_id'],r['Study_id']):int(r['fold']) for r in rows(root/'outputs/analysis/phase17_patient_folds/study_folds.csv')}
    oo_keys=sorted(fm);checks=[]
    for name,file in [('N17','N17_EMA_oof_all.csv'),('S17','S17_oof_all.csv')]:
        gathered={}
        for f in range(1,6):
            run=friend/f'experiments/{name}_F0{f}'
            p=run/('ema_oof_predictions.csv' if name=='N17' else 'oof_predictions.csv')
            provenance=json.loads(p.with_suffix('.provenance.json').read_text(encoding='utf-8'))
            assert sha(p)==provenance['csv_sha256']
            kk=sorted(k for k in oo_keys if fm[k]==f)
            gathered.update(zip(kk,load_prediction(p,kk)))
            checks.append(dict(file=str(p),sha256=sha(p),fold=f,status='PASS'))
        merged=np.array([gathered[k] for k in oo_keys]);published=load_prediction(friend/'oof'/file,oo_keys)
        assert np.array_equal(merged,published),name
    # Compare weighted/tie-aware bootstrap against explicit repeated-patient rows.
    ty=np.tile(np.array([[0],[1],[1],[0],[1],[0]]),(1,10));tp=np.tile(np.array([[.2],[.2],[.8],[.4],[.7],[.7]]),(1,10))
    ids=np.array(['a','a','b','c','c','d']);bb,valid=bootstrap(ty,[tp],ids,123,50)
    rng=np.random.default_rng(123);uq=np.unique(ids);manual=[]
    for _ in range(50):
        sample=rng.integers(len(uq),size=len(uq));ix=np.concatenate([np.where(ids==uq[i])[0] for i in sample])
        support=ty[ix].sum(0)
        if (support==0).any() or (support==len(ix)).any():continue
        manual.append(auc(ty[ix],tp[ix]).mean())
    error=float(np.max(abs(bb[0]-manual)));assert error<1e-12
    (out/'additional_integrity.json').write_text(json.dumps(dict(official_val_labels='MATCH',remote_per_fold_oof_reconstruction='EXACT_MATCH',
        bootstrap_weighted_vs_explicit_max_error=error,bootstrap_test_valid=valid,source_checks=checks),indent=2),encoding='utf-8')
    refresh_sizes(root,out,inventory)
    print('ADDITIONAL_INTEGRITY_PASS',flush=True)


def refresh_sizes(root,out,inventory):
    """Use raw inference checkpoint sizes, never compressed archives or resume states."""
    evidence=[]
    def remote_size(mid,kind='ema'):
        path=Path(inventory[mid]['path']).parent/'model_size.json'
        m=json.loads(path.read_text(encoding='utf-8-sig'))
        size=m[kind]['bytes'] if kind in m else m[kind+'_bytes']
        evidence.append(dict(model=mid,source=str(path),source_sha256=sha(path),bytes=size,status='REMOTE_MANIFEST_NOT_DOWNLOADED'))
        return size
    audit=json.loads((root/'outputs/analysis/e03_h17_fixed_reproduction_20261108/source_audit.json').read_text(encoding='utf-8'))
    h=next(r for r in audit['files'] if r.get('model')=='H17_BATCH1')
    hp=Path(h['path']);assert sha(hp)==h['sha256'] and hp.stat().st_size==h['bytes']
    evidence.append(dict(model='H17_BATCH1',source=str(hp),source_sha256=sha(hp),bytes=hp.stat().st_size,status='LOCAL_HASH_VERIFIED'))
    pos1=next(k for k,v in inventory.items() if 'P32_NS_POS' in v['path'] and 'ema_val' in v['path'])
    local_asl=Path(inventory['M013']['path']).parent.parent/'best_model.pt'
    full=388841336
    replacements={'H17_REPLACE_N01':full-81666223+hp.stat().st_size,'H17_SHARE_N01':full+hp.stat().st_size,
        'LRHIGH_REPLACE_N01':full-81666223+remote_size('M059'),
        'N17_FIVE_REPLACE_N01':full-81666223+sum(remote_size(mid) for mid in ('M116','M117','M118','M119','M120')),
        'POS_FIVE_REPLACE_N01':full-81666223+sum(remote_size(mid) for mid in (pos1,'M077','M079','M081','M083')),
        'MIXUP_REPLACE_N01':full-81666223+remote_size('M071'),
        'ASL_REPLACE_M02':full-111378631+local_asl.stat().st_size}
    table=rows(out/'fixed_candidates.csv')
    for r in table:
        if r['candidate'] in replacements:r['checkpoint_bytes']=replacements[r['candidate']]
        r['deployment_status']='OVER_500MB' if int(r['checkpoint_bytes'])>500000000 else 'T4_UNVERIFIED'
    savecsv(out/'fixed_candidates.csv',table)
    (out/'deployment_size_sources.json').write_text(json.dumps(dict(sources=evidence,
        caveat='Raw inference weight sizes only. Necessary final configs/code must be counted after implementation. New pipelines have no T4 result.'),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
