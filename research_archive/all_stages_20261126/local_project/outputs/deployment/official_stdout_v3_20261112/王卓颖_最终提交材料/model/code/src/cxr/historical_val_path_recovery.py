"""Val-only historical numerical-path recovery. No Test-driven selection."""
from __future__ import annotations
import argparse,csv,gc,io,json,time,types,zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cxr.data.archive import read_split_rows,image_member
from cxr.data.dataset import preprocess_image
from cxr.e08_s17_acceleration import NAMES,load_models,fuse
from cxr.fusion_opportunity_audit import auc
from cxr.prediction_consistency import compare_studies,sha256,write_csv

POLICIES=('historical_batches','batch1_fp16','batch1_fp32_sigmoid','batch1_fp32_classifier','batch1_full_fp32')
BATCHES=dict(B01=16,B02=16,B03=16,M02=8,N01_EMA=8,S17_F3=8)


@contextmanager
def classifier_fp32(models,enabled):
    previous=[]
    def forward(module,x):
        with torch.autocast('cuda',enabled=False):
            return torch.nn.functional.linear(x.float(),module.weight,module.bias)
    try:
        if enabled:
            for name,model in models.items():
                layer=model.head if name=='S17_F3' else model.classifier
                if isinstance(layer,torch.nn.Sequential):layer=layer[-1]
                if not isinstance(layer,torch.nn.Linear):raise TypeError(name)
                previous.append((layer,layer.forward))
                layer.forward=types.MethodType(forward,layer)
        yield
    finally:
        for layer,old in previous:layer.forward=old


class NumericalGraph:
    def __init__(self,models,policy):
        if policy not in POLICIES[1:]:raise ValueError(policy)
        self.models=models;self.policy=policy
        self.x=torch.zeros((1,3,384,384),device='cuda')
        full=policy=='batch1_full_fp32'
        old_tf32=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
        try:
            if full:
                torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            with classifier_fp32(models,policy=='batch1_fp32_classifier'):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream),torch.inference_mode():
                    for _ in range(3):self.forward()
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                self.graph=torch.cuda.CUDAGraph()
                with torch.inference_mode(),torch.cuda.graph(self.graph):self.output=self.forward()
                torch.cuda.synchronize()
        finally:
            torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=old_tf32

    def forward(self):
        outputs=[]
        for name in NAMES:
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.policy!='batch1_full_fp32',cache_enabled=False):
                logits=self.models[name](self.x)
                if self.policy in ('batch1_fp32_sigmoid','batch1_fp32_classifier','batch1_full_fp32'):
                    logits=logits.float()
                outputs.append(torch.sigmoid(logits).squeeze(0).float())
        return torch.stack(outputs)

    def predict(self,x):
        with torch.inference_mode():
            self.x.copy_(x.unsqueeze(0));self.graph.replay()
            return self.output.cpu().numpy().astype(np.float64)


def read_matrix(path,keys):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    table={(r['Subject_id'],r['Study_id']):r for r in rows}
    if len(table)!=len(rows) or set(table)!=set(keys):raise ValueError(f'Bad Study alignment {path}')
    return np.array([[float(table[k][f'probability_{j}']) for j in range(1,11)] for k in keys])


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for item in cfg['locked_sources']:
        if sha256(item['path'])!=item['sha256']:raise ValueError(item['path'])
    if tuple(cfg['policies'])!=POLICIES:raise ValueError('Unfrozen precision choices')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    rows=read_split_rows(cfg['archive'],'val');keys=sorted({(r['Subject_id'],r['Study_id']) for r in rows})
    if len(rows)!=1422 or len(keys)!=859:raise ValueError('Val shape mismatch')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['Subject_id'],r['Study_id'])].append(i)
    labels={k:[int(str(j) in rows[ii[0]]['Predict_class'].split(';')) for j in range(1,11)] for k,ii in groups.items()}
    y=np.array([labels[k] for k in keys])
    refs={n:read_matrix(cfg['references'][n],keys) for n in NAMES}
    historical=np.array([fuse([refs[n][i] for n in NAMES]) for i in range(859)])
    frozen=read_matrix(cfg['historical_e08_val'],keys)*.75+refs['S17_F3']*.25
    if np.max(abs(frozen-historical))>1e-14:raise ValueError('Historical composition mismatch')
    reference=dict(zip(keys,frozen));models=load_models(cfg)
    reports=[];components=[]
    # Process one graph at a time to avoid holding multiple graph workspaces on 8GB.
    for policy in POLICIES:
        start=time.perf_counter();predictions=[]
        graph=None if policy=='historical_batches' else NumericalGraph(models,policy)
        with zipfile.ZipFile(cfg['archive']) as z:
            for offset in range(0,len(rows),16):
                selected=rows[offset:offset+16];tensors=[]
                for r in selected:
                    with Image.open(io.BytesIO(z.read(image_member(r,'val')))) as image:
                        tensors.append(preprocess_image(image,384,preprocessing='black_pad_v1'))
                if policy=='historical_batches':
                    parts=[]
                    for name in NAMES:
                        values=[];batch=BATCHES[name]
                        for j in range(0,len(tensors),batch):
                            x=torch.stack(tensors[j:j+batch]).cuda()
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                                v=torch.sigmoid(models[name](x)).float().cpu().numpy().astype(np.float64)
                            values.extend(v)
                        parts.append(np.array(values))
                    predictions.extend(np.stack(parts,axis=1))
                else:
                    predictions.extend(graph.predict(t) for t in tensors)
                if offset%320==0:print(f'{policy}: {min(offset+16,len(rows))}/1422',flush=True)
        values=np.array(predictions)
        np.savez_compressed(out/f'{policy}_image_components.npz',probabilities=values,
                            image_ids=np.array([r['Dicom_id'] for r in rows]),models=np.array(NAMES))
        study_parts={n:np.array([values[groups[k],j].mean(0) for k in keys]) for j,n in enumerate(NAMES)}
        for n in NAMES:
            rr,_=compare_studies(dict(zip(keys,study_parts[n])),dict(zip(keys,refs[n])))
            components.append(dict(policy=policy,model=n,**rr))
        fused=np.array([fuse([study_parts[n][i] for n in NAMES]) for i in range(859)])
        report,detail=compare_studies(dict(zip(keys,fused)),reference)
        report.update(policy=policy,macro_auc=float(auc(y,fused).mean()),historical_macro_auc=float(auc(y,frozen).mean()),
                      seconds=time.perf_counter()-start,batch1_eligible=policy!='historical_batches',
                      exact_historical_match=bool(np.array_equal(fused,frozen)))
        reports.append(report)
        write_csv(out/f'{policy}_study_predictions.csv',[dict(Subject_id=k[0],Study_id=k[1],**{f'probability_{j}':float(v) for j,v in enumerate(p,1)}) for k,p in zip(keys,fused)])
        write_csv(out/f'{policy}_differences.csv',detail)
        print(json.dumps(report),flush=True)
        (out/'progress.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        del graph;gc.collect();torch.cuda.empty_cache()
    eligible=[r for r in reports if r['batch1_eligible'] and r['status']=='PASS']
    selected=min(eligible,key=lambda r:(r['max_abs'],r['mean_abs']))['policy'] if eligible else None
    result=dict(status='VAL_PASS_PENDING_TEST' if selected else 'BLOCKED_NO_BATCH1_VAL_EQUIVALENT',selected_policy=selected,
                selection='Val only; zero ranking/gate changes and max error <=1e-3, then minimum max/mean error; no AUC maximization',
                reports=reports,components=components,t4_status='T4_UNVERIFIED',test_evaluated=False,
                environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'manifest_sha256.json').write_text(json.dumps({p.name:sha256(p) for p in out.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print('VAL_RECOVERY_COMPLETE '+str(selected),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    run(json.loads(args.config.read_text(encoding='utf-8')))


if __name__=='__main__':main()
