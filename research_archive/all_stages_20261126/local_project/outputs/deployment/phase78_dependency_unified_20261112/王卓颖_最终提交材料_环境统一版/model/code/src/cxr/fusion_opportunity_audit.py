"""CPU-only inventory and fixed-formula exploratory Val fusion audit."""
import argparse,csv,json,hashlib,os,io,zipfile
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(p,data):
    if not data:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def auc(y,p):
    a=[]
    for j in range(10):
        ix=np.argsort(p[:,j],kind='stable');v=p[ix,j];s=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];e=np.r_[s[1:],len(v)]
        ranks=np.repeat((s+e+1)/2,e-s);yy=y[ix,j];n=yy.sum()
        a.append((np.sum(ranks*yy)-n*(n+1)/2)/(n*(len(y)-n)))
    return np.array(a)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));p=Path(cfg['project']);out=Path(args.config).parent
    with zipfile.ZipFile(cfg['archive']) as z:
        rr=list(csv.DictReader(io.StringIO(z.read('cxr_first/valid.csv').decode('utf-8-sig'))))
    truth={}
    for r in rr:
        k=(r['Subject_id'],r['Study_id']);label=[int(str(j) in r['Predict_class'].split(';')) for j in range(1,11)]
        if k in truth:assert truth[k]==label
        truth[k]=label
    keys=sorted(truth);assert len(keys)==859;y=np.array([truth[k] for k in keys]);keyset=set(keys)
    def readpred(path):
        rr=rows(path)
        if len(rr)!=859:return None
        cols=next(([prefix+str(j) for j in range(1,11)] for prefix in ('Prob_','probability_','prob_') if prefix+'1' in rr[0]),None)
        if not cols:return None
        m={(r['Subject_id'],r['Study_id']):r for r in rr}
        if len(m)!=859 or set(m)!=keyset:return None
        a=np.array([[float(m[k][c]) for c in cols] for k in keys])
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Invalid probabilities '+str(path))
        for k in keys:
            if 'target_1' in m[k]:assert [int(float(m[k]['target_'+str(j)])) for j in range(1,11)]==truth[k]
        return a
    e08=readpred(p/'outputs/submissions/E08_EMA_20261020/val_predictions.csv');baseauc=auc(y,e08)
    prior=json.loads((p/'outputs/analysis/e08_fixed_weight_assessment_20261104/audit.json').read_text())
    def component(stem):return readpred(p/next(s for s in prior['source_sha256'] if stem in s))
    e03=.5*(component('b01_')+component('b02_')+component('b03_'))/3+.5*component('m02_')
    models={};aliases={};rejected=[];seen=0
    for root in cfg['roots']:
        for folder,dirs,files in os.walk(root):
            dirs[:]=sorted(d for d in dirs if d not in ('.git','.venv','__pycache__','isolated_deps','node_modules') and not d.startswith('pytest'))
            for name in sorted(files):
                low=(folder+'/'+name).lower()
                if not name.lower().endswith('.csv') or not ('val' in name.lower() or name.lower()=='study_predictions.csv'):continue
                if any(s in low for s in ('holdout','internal_val','phase15','test_prediction','timing','per_class','history','bootstrap','summary','comparison','latency','submission.csv')):continue
                path=Path(folder)/name
                try:
                    if path.stat().st_size>5_000_000:continue
                    seen+=1;a=readpred(path)
                    if a is None:continue
                    h=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
                    aliases.setdefault(h,[]).append(str(path))
                    if h not in models:models[h]=(a,path,hashlib.sha256(path.read_bytes()).hexdigest())
                except (ValueError,KeyError,UnicodeError,OSError) as exc:rejected.append(dict(path=str(path),reason=str(exc)))
    inventory=[];scores=[];allpred=[]
    for i,(h,(a,path,filehash)) in enumerate(models.items()):
        mid='M%03d'%i;own=auc(y,a)
        corr=np.mean([np.corrcoef(a[:,j],e08[:,j])[0,1] for j in range(10)])
        inventory.append(dict(model_id=mid,path=str(path),sha256=filehash,matrix_sha256=h,aliases=len(aliases[h]),val_auc=own.mean(),mean_correlation_e08=corr))
        for name,prob in [('E08_075_MODEL_025',.75*e08+.25*a),('E03_050_MODEL_050',.5*e03+.5*a)]:
            ca=auc(y,prob);cid=mid+'__'+name
            scores.append(dict(candidate=cid,model_id=mid,formula=name,val_auc=ca.mean(),delta=float((ca-baseauc).mean()),worst_class_delta=float((ca-baseauc).min()),positive_classes=int(((ca-baseauc)>0).sum()),source=str(path)))
            allpred.append(prob)
    savecsv(out/'model_inventory.csv',inventory);savecsv(out/'fixed_formula_screen.csv',sorted(scores,key=lambda r:r['val_auc'],reverse=True));savecsv(out/'rejected_sources.csv',rejected)
    (out/'source_aliases.json').write_text(json.dumps(aliases,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'screen_arrays.npz',y=y,e08=e08,predictions=np.stack(allpred),patients=np.array([k[0] for k in keys]),keys=np.array(keys))
    (out/'candidate_order.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(scanned=seen,unique_models=len(models),candidate_formulas=len(scores),e08_auc=float(baseauc.mean()),top=sorted(scores,key=lambda r:r['val_auc'],reverse=True)[:12]),ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
