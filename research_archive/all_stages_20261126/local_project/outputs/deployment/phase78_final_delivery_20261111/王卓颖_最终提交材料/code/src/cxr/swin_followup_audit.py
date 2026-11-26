"""Fixed, CPU-only follow-up of the frozen E08/S17 candidate; no Test reads."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cxr.fusion_opportunity_audit import auc, rows, savecsv


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_prediction(path, keys):
    rr = rows(path)
    table = {(r['Subject_id'], r['Study_id']): r for r in rr}
    assert len(rr) == len(table) == len(keys) and set(table) == set(keys), str(path)
    prefix = next(p for p in ('probability_', 'Prob_', 'prob_') if p+'1' in rr[0])
    p = np.array([[float(table[k][prefix+str(j)]) for j in range(1, 11)] for k in keys])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def bootstrap(y, predictions, patients, seed, repeats):
    """Patient cluster bootstrap using integer Study weights and exact tie handling."""
    unique, pi = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    distributions = [[] for _ in predictions]
    valid = 0
    for offset in range(0, repeats, 100):
        weights = []
        for _ in range(min(100, repeats-offset)):
            w = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[pi].astype(float)
            support = (w[:, None]*y).sum(0)
            if (support == 0).any() or (support == w.sum()).any():
                continue
            weights.append(w)
        if not weights:
            continue
        w = np.stack(weights)
        valid += len(w)
        for m, p in enumerate(predictions):
            values = np.zeros(len(w))
            for j in range(10):
                ix = np.argsort(p[:, j], kind='stable')
                v = p[ix, j]
                starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1])+1]
                pos = np.add.reduceat(w[:, ix]*y[ix, j], starts, axis=1)
                neg = np.add.reduceat(w[:, ix]*(1-y[ix, j]), starts, axis=1)
                values += np.sum(pos*(np.cumsum(neg, axis=1)-neg+.5*neg), axis=1)/(pos.sum(1)*neg.sum(1))/10
            distributions[m].extend(values)
    return np.asarray(distributions), valid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text(encoding='utf-8'))
    root = Path(cfg['project']); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    prior = root/'outputs/analysis/fusion_opportunities_20261110'
    z = np.load(prior/'screen_arrays.npz')
    keys = [tuple(k) for k in z['keys']]; y = z['y']; patients = z['patients']
    inventory = {r['model_id']: r for r in rows(prior/'model_inventory.csv')}
    source_records = []
    def model(mid):
        info = inventory[mid]; path = Path(info['path'])
        assert sha(path) == info['sha256'], str(path)
        source_records.append(dict(model_id=mid, path=str(path), sha256=sha(path)))
        return load_prediction(path, keys)
    base = z['e08']; swin = model('M027'); champion = .75*base+.25*swin
    b = [model(mid) for mid in ('M000','M001','M002')]
    m02 = model('M015'); n01 = model('M016'); e03 = .5*np.mean(b, axis=0)+.5*m02
    assert abs(auc(y, champion).mean()-.8145689350883447) < 1e-12
    e08manifest = json.loads((root/'outputs/submissions/E08_EMA_20261020/manifest.json').read_text(encoding='utf-8'))
    weights_bytes = 388841336
    candidates = []
    def add(name, p, size, count, formula):
        candidates.append(dict(candidate=name, probability=p, checkpoint_bytes=size, model_count=count, formula=formula))
    # These mechanisms and coefficients are frozen in config before any new scores.
    h17 = model('M178')
    add('H17_REPLACE_N01', .375*e03+.375*h17+.25*swin, weights_bytes-81666223+81654329, 6,
        '.375 E03 + .375 H17_BATCH1 + .25 S17F3')
    add('H17_SHARE_N01', .375*e03+.1875*n01+.1875*h17+.25*swin, weights_bytes+81654329, 7,
        '.375 E03 + .1875 N01_EMA + .1875 H17_BATCH1 + .25 S17F3')
    add('LRHIGH_REPLACE_N01', .375*e03+.375*model('M059')+.25*swin, weights_bytes-81666223+81655151, 6,
        '.375 E03 + .375 ET_LR_HIGH_F01_EMA + .25 S17F3')
    add('N17_FIVE_REPLACE_N01', .375*e03+.375*model('M137')+.25*swin, None, 10,
        '.375 E03 + .375 mean(N17_EMA_F1_F5) + .25 S17F3')
    pos1 = next(k for k,v in inventory.items() if 'P32_NS_POS' in v['path'] and 'ema_val' in v['path'])
    pos = np.mean([model(mid) for mid in (pos1,'M077','M079','M081','M083')], axis=0)
    add('POS_FIVE_REPLACE_N01', .375*e03+.375*pos+.25*swin, None, 10,
        '.375 E03 + .375 mean(POS_EMA_F1_F5) + .25 S17F3')
    add('MIXUP_REPLACE_N01', .375*e03+.375*model('M071')+.25*swin, None, 6,
        '.375 E03 + .375 P32_MIXUP_EMA + .25 S17F3')
    add('NM17_REPLACE_N01', .375*e03+.375*model('M121')+.25*swin, weights_bytes-81666223+213046249, 6,
        '.375 E03 + .375 NM17_F01 + .25 S17F3')
    s5 = np.mean([model(mid) for mid in ('M025','M026','M027','M028','M029')], axis=0)
    add('SWIN_FIVE_FIXED_MEAN', .75*base+.25*s5, 278436014+5*110405322, 10,
        '.75 E08 + .25 mean(local_S17_F1_F5)')
    friend_s3 = next(k for k,v in inventory.items() if 'phase17_batch' in v['path'] and 'S17_F03' in v['path'] and v['path'].endswith('val_predictions.csv'))
    add('SWIN_LOCAL_REMOTE_F3_MEAN', .75*base+.125*swin+.125*model(friend_s3), weights_bytes+110402570, 7,
        '.75 E08 + .125 local_S17F3 + .125 friend_S17F3')
    for omit in range(3):
        dense = np.mean([p for i,p in enumerate(b) if i != omit], axis=0)
        add('REMOVE_B0'+str(omit+1), .1875*dense+.1875*m02+.375*n01+.25*swin,
            weights_bytes-28463720, 5, '.1875 mean(other_two_DenseNet) + .1875 M02 + .375 N01_EMA + .25 S17F3')
    add('ASL_REPLACE_M02', .1875*np.mean(b, axis=0)+.1875*model('M013')+.375*n01+.25*swin, None, 6,
        '.1875 mean(B01_B02_B03) + .1875 L01_ASL + .375 N01_EMA + .25 S17F3')
    assert len(candidates) == 13
    print('Frozen 13 candidate formulas loaded. Starting Val bootstrap.', flush=True)
    ps = [champion]+[c['probability'] for c in candidates]
    boot, valid = bootstrap(y, ps, patients, cfg['seed'], cfg['bootstrap'])
    point = np.array([auc(y,p).mean() for p in ps]); d = boot[1:]-boot[0]
    observed = point[1:]-point[0]; se = d.std(1,ddof=1)
    max_t = ((d-observed[:,None])/se[:,None]).max(0)
    output = []; perclass=[]; corr=[]
    for i,c in enumerate(candidates):
        ca=auc(y,c['probability']); ba=auc(y,champion); ci=np.quantile(d[i],[.025,.975])
        output.append({k:v for k,v in c.items() if k != 'probability'} | dict(
            val_auc=float(point[i+1]),delta_vs_7881=float(observed[i]),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
            maxT_p=float((1+(max_t>=observed[i]/se[i]).sum())/(len(max_t)+1)),
            positive_classes=int((ca>ba).sum()),worst_class_delta=float((ca-ba).min()),valid_bootstraps=valid,
            deployment_status='WEIGHT_SIZE_PENDING' if c['checkpoint_bytes'] is None else ('OVER_500MB' if c['checkpoint_bytes']>500000000 else 'T4_UNVERIFIED')))
        for j in range(10):
            perclass.append(dict(candidate=c['candidate'],class_id=j+1,auc=float(ca[j]),baseline_auc=float(ba[j]),
                                 delta=float(ca[j]-ba[j]),positive_studies=int(y[:,j].sum())))
    for mid in inventory:
        # Reuse earlier screen arrays to inspect residual complementarity without a new weight search.
        info=inventory[mid]
        order=json.loads((prior/'candidate_order.json').read_text(encoding='utf-8'))
        idx=next(i for i,c in enumerate(order) if c['model_id']==mid and c['formula']=='E08_075_MODEL_025')
        p=(z['predictions'][idx]-.75*base)/.25
        corr.append(dict(model_id=mid,path=info['path'],own_val_auc=float(auc(y,p).mean()),
            mean_correlation_champion=float(np.mean([np.corrcoef(p[:,j],champion[:,j])[0,1] for j in range(10)])),
            mean_residual_correlation_swin=float(np.mean([np.corrcoef((p-base)[:,j],(swin-base)[:,j])[0,1] for j in range(10)])) if np.max(abs(p-base))>1e-12 else None))
    savecsv(out/'fixed_candidates.csv',sorted(output,key=lambda r:r['val_auc'],reverse=True))
    savecsv(out/'per_class_candidates.csv',perclass); savecsv(out/'residual_complementarity.csv',corr)
    (out/'val_method.json').write_text(json.dumps(dict(baseline_auc=point[0],baseline_official_score=78.81,
        bootstrap_valid=valid,bootstrap_invalid=cfg['bootstrap']-valid,source_records=source_records,
        limitations='Exploratory post-leaderboard hypotheses; repeated Val, earlier 382-formula selection and historical selection are NOT corrected by the new 13-way maxT. No independent validation.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(sorted(output,key=lambda r:r['val_auc'],reverse=True)[:4],ensure_ascii=True),flush=True)
    oof_audit(root,out,cfg)


def oof_audit(root,out,cfg):
    friend=Path(cfg['friend17']); fold_dir=root/'outputs/analysis/phase17_patient_folds'
    fold_rows=rows(fold_dir/'study_folds.csv'); fm={(r['Subject_id'],r['Study_id']):int(r['fold']) for r in fold_rows}
    keys=sorted(fm); assert len(keys)==18513 and len({k[0] for k in keys})==11055
    patient_fold={r['Subject_id']:int(r['fold']) for r in rows(fold_dir/'patient_folds.csv')}
    assert all(patient_fold[k[0]]==fm[k] for k in keys)
    truth={ (r['Subject_id'],r['Study_id']):[int(str(j) in r['Predict_class'].split(';')) for j in range(1,11)] for r in fold_rows}
    y=np.array([truth[k] for k in keys]); patients=np.array([k[0] for k in keys]); folds=np.array([fm[k] for k in keys])
    nfile=friend/'oof/N17_EMA_oof_all.csv'; sfile=friend/'oof/S17_oof_all.csv'
    for path,source in ((nfile,'N17_EMA'),(sfile,'S17')):
        provenance=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        assert sha(path)==provenance['csv_sha256']
        source_rows=rows(friend/f'oof/{source}_source_index.csv')
        assert len(source_rows)==18513 and len({(r['Subject_id'],r['Study_id']) for r in source_rows})==18513
        assert all(fm[(r['Subject_id'],r['Study_id'])]==int(r['fold']) for r in source_rows)
    for f in range(1,6):
        for architecture in ('N17','S17'):
            rr=rows(friend/f'experiments/{architecture}_F0{f}/training_image_index.csv')
            assert all(patient_fold[r['Subject_id']] != f and fm[(r['Subject_id'],r['Study_id'])] != f for r in rr)
        local=next((root/'outputs/experiments').glob(f's17_f{f}_*'))
        frozen=json.loads((local/'frozen_manifest.json').read_text(encoding='utf-8'))
        assert all(sha(fold_dir/n)==h for n,h in frozen['fold_index_sha256'].items())
        source=local/'oof_analysis/study_predictions.csv'
        assert sha(source)==frozen['files']['oof_analysis/study_predictions.csv']['sha256']
        local_keys=sorted(k for k in keys if fm[k]==f)
        lp=load_prediction(source,local_keys)
        if f==1: local_table={}
        local_table.update(zip(local_keys,lp))
    n=load_prediction(nfile,keys); s=load_prediction(sfile,keys); local=np.array([local_table[k] for k in keys])
    models={'N17_EMA':n,'LOCAL_SWIN':local,'FRIEND_SWIN':s,'N75_LOCAL_S25':.75*n+.25*local,
            'N75_FRIEND_S25':.75*n+.25*s,'N75_LOCAL_S125_FRIEND_S125':.75*n+.125*local+.125*s}
    results=[]; per=[]
    for scope,mask in [('ALL_5_FOLDS',np.ones(len(keys),bool)),('EXCLUDE_SELECTED_FOLD3',folds!=3)]:
        yy=y[mask]; nn=n[mask]; selected=['N17_EMA','N75_LOCAL_S25','N75_FRIEND_S25','N75_LOCAL_S125_FRIEND_S125']
        print('OOF bootstrap '+scope,flush=True)
        boot,valid=bootstrap(yy,[models[name][mask] for name in selected],patients[mask],cfg['seed'],cfg['bootstrap'])
        for i,name in enumerate(selected):
            ca=auc(yy,models[name][mask]); ba=auc(yy,nn); ci=np.quantile(boot[i]-boot[0],[.025,.975])
            results.append(dict(scope=scope,model=name,studies=int(mask.sum()),patients=len(set(patients[mask])),
                auc=float(ca.mean()),delta_vs_n17=float((ca-ba).mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),valid_bootstraps=valid))
            for j in range(10):per.append(dict(scope=scope,model=name,class_id=j+1,auc=float(ca[j]),delta=float(ca[j]-ba[j]),positive_studies=int(yy[:,j].sum())))
    for f in range(1,6):
        mask=folds==f
        for name,p in models.items():
            results.append(dict(scope=f'FOLD{f}',model=name,studies=int(mask.sum()),patients=len(set(patients[mask])),
                auc=float(auc(y[mask],p[mask]).mean()),delta_vs_n17=float((auc(y[mask],p[mask])-auc(y[mask],n[mask])).mean()),
                ci95_low=None,ci95_high=None,valid_bootstraps=None))
    savecsv(out/'oof_mechanism_confirmation.csv',results);savecsv(out/'oof_per_class.csv',per)
    (out/'oof_integrity.json').write_text(json.dumps(dict(studies=18513,patients=11055,status='PASS',
        checks=['local five fold frozen index and OOF hashes','remote source index matches local patient fold','remote N17/S17 training patient isolation','remote OOF provenance hash'],
        limitation='OOF tests architecture-level complementarity, NOT the E08+S17F3 submitted ensemble. Local/remote training protocol differences remain. No E08 OOF fabricated.'),indent=2),encoding='utf-8')
    manifest={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest_sha256.json'}
    (out/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
