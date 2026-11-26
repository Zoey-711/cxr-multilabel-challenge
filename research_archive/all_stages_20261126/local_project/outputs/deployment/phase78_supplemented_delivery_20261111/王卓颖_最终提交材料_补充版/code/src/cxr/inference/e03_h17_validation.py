"""Fixed E03/H17 batch-one inference and Val-only reproduction. Never trains."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, platform, time, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from cxr.data.archive import read_split_rows, split_archive_path, image_member, parse_positive_classes
from cxr.data.dataset import preprocess_image
from cxr.models.factory import create_model
from cxr.utils.checkpoint import load_model_state

NAMES = ['B01', 'B02', 'B03', 'M02', 'H17_BATCH1']

def roc_auc_score(y,p,average='macro',sample_weight=None):
    """Weighted pairwise AUROC with half credit for ties, no extra dependency."""
    if y.ndim==2:
        return float(np.mean([roc_auc_score(y[:,j],p[:,j],sample_weight=sample_weight) for j in range(y.shape[1])]))
    w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
    ix=np.argsort(p,kind='stable'); yy=y[ix]; pp=p[ix]; ww=w[ix]
    starts=np.r_[0,np.flatnonzero(np.diff(pp))+1]
    pos=np.add.reduceat(ww*yy,starts); neg=np.add.reduceat(ww*(1-yy),starts)
    assert pos.sum()>0 and neg.sum()>0
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def save(p, obj):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(t,p)

def fuse(p):
    return .5*(.5*(p['B01']+p['B02']+p['B03'])/3+.5*p['M02'])+.5*p['H17_BATCH1']

def load_models(source):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required, no CPU fallback')
    models={}
    for r in source['weights']:
        p=Path(r['path'])
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'], str(p)
        if r['model']=='H17_BATCH1':
            m=create_model('efficientnet_v2_s',num_classes=10,pretrained=False)
            m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
            m=m.cuda().eval()
        else: m=load_model_state(p,device='cuda')
        m.requires_grad_(False); models[r['model']]=m
    assert list(models)==NAMES and all(not m.training for m in models.values())
    return models

def predict_image(image, models):
    """Actual final entry: original black_pad_v1, batch=1, FP16 sigmoid."""
    if isinstance(image,Image.Image): x=preprocess_image(image,384,preprocessing='black_pad_v1')
    else:
        with Image.open(image) as im: x=preprocess_image(im,384,preprocessing='black_pad_v1')
    x=x.unsqueeze(0).cuda()
    p={}
    with torch.inference_mode():
        for n,m in models.items():
            with torch.autocast('cuda',dtype=torch.float16):
                p[n]=torch.sigmoid(m(x)).squeeze(0).float().cpu().numpy().astype(np.float64)
    p['E03_H17']=fuse(p)
    assert all(v.shape==(10,) and np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in p.values())
    return p

def read_probs(p, keys):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: rr=list(csv.DictReader(f))
    cols=[f'Prob_{j}' for j in range(1,11)] if 'Prob_1' in rr[0] else [f'probability_{j}' for j in range(1,11)]
    d={(r['Subject_id'],r['Study_id']):np.array([float(r[c]) for c in cols]) for r in rr}
    assert len(d)==len(rr) and set(d)==set(keys)
    a=np.array([d[k] for k in keys]); assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
    return a

def csv_write(p, fields, rr):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rr)

def compare(a,b,y,keys,name):
    d=np.abs(a-b); i,j=np.unravel_index(d.argmax(),d.shape)
    return dict(name=name,max_abs=float(d.max()),mean_abs=float(d.mean()),p99_abs=float(np.quantile(d,.99)),
        auc=float(roc_auc_score(y,a,average='macro')),reference_auc=float(roc_auc_score(y,b,average='macro')),
        strict_1e4_pass=bool(d.max()<=1e-4),worst_subject=keys[i][0],worst_study=keys[i][1],worst_class=int(j+1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True,type=Path); args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig')); out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists(): raise FileExistsError('Completed output exists; no overwrite')
    lock=out/'running.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try: run(cfg,out,args.config)
    finally: lock.unlink(missing_ok=True)

def run(cfg,out,config_path):
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    source=json.loads(Path(cfg['candidate_manifest']).read_text(encoding='utf-8'))
    checked=[]
    for r in source['weights']+source['source_configs']:
        assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
        checked.append(r)
    archive=Path(cfg['archive']); rows=read_split_rows(archive,'val'); truth={}; counts=defaultdict(int)
    assert len(rows)==1422 and len({r['Dicom_id'] for r in rows})==1422
    for r in rows:
        k=(r['Subject_id'],r['Study_id']); lab=np.array([int(j in parse_positive_classes(r['Predict_class'])) for j in range(1,11)])
        if k in truth: assert np.array_equal(truth[k],lab)
        truth[k]=lab; counts[k]+=1
    keys=sorted(truth); y=np.array([truth[k] for k in keys])
    assert len(keys)==859 and len({k[0] for k in keys})==228 and len({k[1] for k in keys})==859
    refs={n:read_probs(cfg['references'][n],keys) for n in NAMES}; refs['E03_H17']=fuse(refs)
    e08=read_probs(cfg['e08_reference'],keys)
    from cxr.data import dataset
    hsrc=Path(cfg['h17_package'])/'src/cxr/data/dataset.py'
    # Execute original frozen preprocessing module independently, compare every real image tensor.
    import importlib.util
    spec=importlib.util.spec_from_file_location('cxr.data.frozen_h17_dataset',hsrc)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    models=load_models(source); torch.cuda.reset_peak_memory_stats()
    save(out/'source_audit.json',dict(files=checked,reference_hashes={k:sha(v) for k,v in cfg['references'].items()},
        e08_reference_sha256=sha(cfg['e08_reference']),candidate_manifest_sha256=sha(cfg['candidate_manifest']),
        code_sha256=sha(__file__),config_sha256=sha(config_path),frozen_h17_preprocess_sha256=sha(hsrc),
        current_preprocess_sha256=sha(dataset.__file__),torch=torch.__version__,cuda=torch.version.cuda,
        python=platform.python_version(),gpu=torch.cuda.get_device_name(),protocol=cfg['protocol']))
    passes=[]; started=time.monotonic(); inputmax=0.
    with zipfile.ZipFile(split_archive_path(archive,'val')) as z:
        metadata=z.read('cxr_first/valid.csv'); save(out/'metadata_source.json',{'member':'cxr_first/valid.csv','sha256':hashlib.sha256(metadata).hexdigest()})
        for repeat in range(2):
            acc={n:defaultdict(list) for n in NAMES+['E03_H17']}; images=[]
            for i,r in enumerate(rows):
                member=image_member(r,'val'); raw=z.read(member)
                with Image.open(io.BytesIO(raw)) as im:
                    if repeat==0:
                        a=preprocess_image(im,384,preprocessing='black_pad_v1'); b=old.preprocess_image(im,384,preprocessing='black_pad_v1')
                        delta=float((a-b).abs().max()); inputmax=max(inputmax,delta); assert delta==0
                    p=predict_image(im,models)
                k=(r['Subject_id'],r['Study_id'])
                for n,v in p.items(): acc[n][k].append(v)
                images.append([*k,r['Dicom_id'],hashlib.sha256(raw).hexdigest(),*[float(t) for n in NAMES for t in p[n]]])
                if (i+1)%100==0 or i==len(rows)-1:
                    state=dict(pass_number=repeat+1,images_done=i+1,images_total=len(rows),elapsed_seconds=time.monotonic()-started)
                    save(out/'status.json',state); print(json.dumps(state),flush=True)
            arrays={n:np.array([np.mean(acc[n][k],axis=0) for k in keys]) for n in acc}
            assert np.max(np.abs(arrays['E03_H17']-fuse(arrays)))<1e-14
            for n,a in arrays.items(): csv_write(out/f'pass{repeat+1}_{n}_val.csv',['Subject_id','Study_id']+[f'Prob_{j}' for j in range(1,11)],[[*k,*v] for k,v in zip(keys,a)])
            csv_write(out/f'pass{repeat+1}_image_predictions.csv',['Subject_id','Study_id','Dicom_id','image_sha256']+[f'{n}_Prob_{j}' for n in NAMES for j in range(1,11)],images)
            passes.append(arrays)
    comparisons=[]; pc=[]
    for n in NAMES+['E03_H17']:
        a=passes[0][n]; comparisons.append(compare(a,refs[n],y,keys,n))
        d=np.abs(a-refs[n])
        for j in range(10): pc.append([n,j+1,float(d[:,j].max()),float(d[:,j].mean()),float(np.quantile(d[:,j],.99)),float(roc_auc_score(y[:,j],a[:,j])),float(roc_auc_score(y[:,j],refs[n][:,j])),int(y[:,j].sum())])
    repeat_diff={n:float(np.max(np.abs(passes[0][n]-passes[1][n]))) for n in passes[0]}
    csv_write(out/'per_class_consistency.csv',['model','class','max_abs','mean_abs','p99_abs','auc','reference_auc','positive_studies'],pc)
    csv_write(out/'val_comparison.csv',list(comparisons[0]),[list(r.values()) for r in comparisons])
    # Patient-cluster bootstrap; complete patient clusters, no classwise deletion.
    subjects=sorted({k[0] for k in keys}); ix={s:i for i,s in enumerate(subjects)}; si=np.array([ix[k[0]] for k in keys]); rng=np.random.default_rng(42); diffs=[]; invalid=0
    actual=passes[0]['E03_H17']
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(subjects),len(subjects)),minlength=len(subjects))[si]
        if np.any((y*w[:,None]).sum(0)==0) or np.any(((1-y)*w[:,None]).sum(0)==0): invalid+=1; continue
        diffs.append(float(roc_auc_score(y,actual,sample_weight=w)-roc_auc_score(y,e08,sample_weight=w)))
    boot=dict(draws=2000,valid=len(diffs),invalid=invalid,seed=42,ci95=np.quantile(diffs,[.025,.975]).tolist(),delta=float(roc_auc_score(y,actual)-roc_auc_score(y,e08)))
    report=dict(state='VAL_REPRODUCTION_COMPLETED',images=1422,studies=859,patients=228,input_tensor_max_abs=inputmax,
        historical_comparisons=comparisons,self_repeat_max_abs=repeat_diff,self_repeat_pass=all(v<=cfg['self_tolerance'] for v in repeat_diff.values()),
        self_tolerance=cfg['self_tolerance'],historical_tolerance=1e-4,bootstrap_vs_e08=boot,
        elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        weights_and_source_configs_bytes=source['checkpoint_and_source_config_bytes'],test_read=False,submission_generated=False,
        latency='NOT_BENCHMARKED; NEW_CANDIDATE_T4_UNVERIFIED')
    # Recheck immutable model/config inputs after inference.
    assert all(sha(r['path'])==r['sha256'] for r in checked)
    save(out/'report.json',report)
    save(out/'manifest_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest_sha256.json','running.lock','training.log']})
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
