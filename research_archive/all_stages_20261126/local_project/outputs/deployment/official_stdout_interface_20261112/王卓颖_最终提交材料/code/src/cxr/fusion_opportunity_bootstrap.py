"""Joint patient bootstrap for all screened fixed fusion candidates (CPU only)."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from cxr.fusion_opportunity_audit import auc,savecsv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));out=Path(args.config).parent
    z=np.load(out/'screen_arrays.npz');y=z['y'];baseline=z['e08'];ps=z['predictions'];patients=z['patients']
    candidates=json.loads((out/'candidate_order.json').read_text(encoding='utf-8'))
    unique=sorted(set(patients));lookup={s:i for i,s in enumerate(unique)};pi=np.array([lookup[s] for s in patients])
    rng=np.random.default_rng(cfg['seed']);ww=[];invalid=0
    for _ in range(cfg['bootstrap']):
        count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));w=count[pi].astype(float)
        support=(w[:,None]*y).sum(0)
        if (support==0).any() or (support==w.sum()).any():invalid+=1;continue
        ww.append(w)
    w=np.stack(ww)
    def boot(p):
        result=np.zeros(len(w))
        for j in range(10):
            ix=np.argsort(p[:,j],kind='stable');v=p[ix,j];starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
            pos=np.add.reduceat(w[:,ix]*y[ix,j],starts,axis=1)
            neg=np.add.reduceat(w[:,ix]*(1-y[ix,j]),starts,axis=1)
            result+=np.sum(pos*(np.cumsum(neg,axis=1)-neg+.5*neg),axis=1)/(pos.sum(1)*neg.sum(1))/10
        return result
    bb=boot(baseline);differences=[];t=time.time()
    for i,p in enumerate(ps):
        differences.append(boot(p)-bb)
        if i%20==0:print('bootstrap',i+1,'/',len(ps),'seconds',round(time.time()-t),flush=True)
    d=np.stack(differences);observed=np.array([r['delta'] for r in candidates]);se=d.std(1,ddof=1)
    nonzero=se>1e-14
    max_t=np.max((d[nonzero]-observed[nonzero,None])/se[nonzero,None],axis=0)
    threshold=np.quantile(max_t,.95);results=[];per=[]
    ba=auc(y,baseline)
    for i,(r,p) in enumerate(zip(candidates,ps)):
        ci=np.quantile(d[i],[.025,.975]);stat=observed[i]/se[i] if nonzero[i] else 0
        results.append(dict(**r,ci95_low=float(ci[0]),ci95_high=float(ci[1]),maxT_adjusted_p=float((1+np.sum(max_t>=stat))/(len(max_t)+1)) if nonzero[i] else 1.,simultaneous_one_sided_lower=float(observed[i]-threshold*se[i]),valid_bootstraps=len(w)))
        ca=auc(y,p)
        for j in range(10):per.append(dict(candidate=r['candidate'],class_id=j+1,positive_studies=int(y[:,j].sum()),auc=ca[j],delta=ca[j]-ba[j]))
    results.sort(key=lambda r:r['val_auc'],reverse=True)
    savecsv(out/'bootstrap_all_candidates.csv',results);savecsv(out/'per_class_all_candidates.csv',per)
    (out/'bootstrap_method.json').write_text(json.dumps(dict(seed=cfg['seed'],valid=len(w),invalid=invalid,patients=len(unique),candidates=len(ps),method='Patient cluster percentile CI; centered standardized bootstrap maxT across ALL screened formulas; one-sided familywise 95% lower bound',maxT_threshold=float(threshold),limitations='Exploratory repeated Val and history-adaptive model choices remain; no independent confirmation; no training-seed uncertainty'),indent=2))
    print(json.dumps(results[:6],ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
