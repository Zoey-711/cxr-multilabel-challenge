"""Phase65-inspired six-model inference; independent from frozen submissions."""
import argparse
import csv
import gc
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from cxr.benchmark_e08_s17 import CombinedEntry, load_module
from cxr.data.dataset import preprocess_image, IMAGENET_MEAN, IMAGENET_STD
from cxr.data.archive import read_split_rows, image_member
from cxr.fusion_opportunity_audit import auc, savecsv

NAMES = ('B01', 'B02', 'B03', 'M02', 'N01_EMA', 'S17_F3')
PRECISION = 'PyTorch FP16 autocast; FP32 stored weights; batch=1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def gray_first(image):
    if image.mode != 'L':
        return preprocess_image(image,384,preprocessing='black_pad_v1')
    padded=ImageOps.pad(image,(384,384),method=Image.Resampling.BILINEAR,color=0)
    return TF.normalize(TF.to_tensor(padded.convert('RGB')),IMAGENET_MEAN,IMAGENET_STD)


def fuse(values):
    p=dict(zip(NAMES,values))
    e01=(p['B01']+p['B02']+p['B03'])/3.0
    e03=.5*e01+.5*p['M02']
    e08=.5*e03+.5*p['N01_EMA']
    result=.75*e08+.25*p['S17_F3']
    if result.shape!=(10,) or not np.isfinite(result).all() or ((result<0)|(result>1)).any():
        raise ValueError('Invalid probability output')
    return result


class Entry:
    PRECISION=PRECISION

    def __init__(self,models,mode):
        self.models=models; self.mode=mode; self.graph=None
        if mode=='cuda_graph':
            self.static_x=torch.zeros((1,3,384,384),device='cuda',dtype=torch.float32)
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                for _ in range(5):
                    self.forward_stack(self.static_x,cache=False)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(self.graph):
                self.static_output=self.forward_stack(self.static_x,cache=False)
            torch.cuda.synchronize()

    def load_models(self):
        return self

    def forward_stack(self,x,cache=True):
        # Leave sigmoid in the original autocast scope and retain FP16 rounding.
        pp=[]
        for name in NAMES:
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=cache):
                pp.append(torch.sigmoid(self.models[name](x)).squeeze(0).float())
        return torch.stack(pp)

    def from_tensor(self,tensor):
        with torch.inference_mode():
            if self.mode=='cuda_graph':
                self.static_x.copy_(tensor.unsqueeze(0),non_blocking=False)
                self.graph.replay()
                values=self.static_output.cpu().numpy().astype(np.float64)
            else:
                x=tensor.unsqueeze(0).to('cuda')
                if self.mode=='single_copy':
                    values=self.forward_stack(x).cpu().numpy().astype(np.float64)
                else:
                    values=[]
                    for name in NAMES:
                        with torch.autocast('cuda',dtype=torch.float16):
                            p=torch.sigmoid(self.models[name](x)).squeeze(0).float().cpu().numpy().astype(np.float64)
                        values.append(p)
            return fuse(values)

    def predict_image(self,path,_state=None):
        with Image.open(path) as image:
            tensor=preprocess_image(image,384,preprocessing='black_pad_v1') if self.mode=='baseline' else gray_first(image)
        return self.from_tensor(tensor)


def load_models(config):
    package=Path(config['e08_package'])
    sys.path.insert(0,str(package))
    frozen=load_module(package/'predict.py','e08_accel_frozen_entry')
    entry=CombinedEntry(frozen,Path(config['s17_checkpoint']),config['s17_sha256'])
    return entry.load_models()


def predict_image(path,state):
    return state.predict_image(path)


def stage_benchmark(cfg,out,core,entries):
    reports=[]
    samples=Path(cfg['val_samples'])
    for name,entry in entries.items():
        print('BENCHMARK_START '+name,flush=True)
        sys.argv=['cost','--pictures-100',str(samples/'pictures_100'),'--pictures-5',str(samples/'pictures_5'),
                  '--output',str(out/name)]
        core.run(entry,name)
        report=json.loads((out/name/'report.json').read_text())
        reports.append(dict(candidate=name,mean_ms=report['per_image_then_overall_mean_ms'],p50_ms=report['p50_ms'],
                            p95_ms=report['p95_ms'],max_ms=report['max_ms'],records=report['records']))
        savecsv(out/'benchmark_comparison.csv',reports)
    return reports


def validate_full(cfg,out,baseline,winner):
    archive=Path(cfg['archive']); metadata=read_split_rows(archive,'val')
    assert len(metadata)==1422
    ids=[(r['Subject_id'],r['Study_id'],r['Dicom_id']) for r in metadata]
    assert len(set(ids))==1422 and len({k[:2] for k in ids})==859
    ref=[];opt=[];max_tensor=0.;tensor_sum=0.;tensor_count=0;changed_inputs=0;changed_images=0
    with zipfile.ZipFile(archive) as z:
        for i,r in enumerate(metadata):
            with Image.open(io.BytesIO(z.read(image_member(r,'val')))) as image:
                t0=preprocess_image(image,384,preprocessing='black_pad_v1')
                t1=gray_first(image)
            diff=(t0-t1).abs(); mm=float(diff.max()); max_tensor=max(max_tensor,mm)
            tensor_sum+=float(diff.double().sum());tensor_count+=diff.numel();changed_inputs+=int(mm!=0)
            p0=baseline.from_tensor(t0);p1=winner.from_tensor(t1)
            ref.append(p0);opt.append(p1);changed_images+=int(not np.array_equal(p0,p1))
            if (i+1)%100==0:print('FULL_VAL',i+1,'/1422',flush=True)
    ref=np.array(ref);opt=np.array(opt)
    with (out/'val_image_probabilities.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['Subject_id','Study_id','Image_id',*[f'baseline_{j}' for j in range(1,11)],*[f'optimized_{j}' for j in range(1,11)]])
        for k,a,b in zip(ids,ref,opt):w.writerow([*k,*a,*b])
    keys=sorted({k[:2] for k in ids});groups=defaultdict(list);truth={}
    for i,r in enumerate(metadata):
        k=ids[i][:2];groups[k].append(i);y=[int(str(j) in r['Predict_class'].split(';')) for j in range(1,11)]
        if k in truth:assert truth[k]==y
        truth[k]=y
    a=np.array([ref[groups[k]].mean(0) for k in keys]);b=np.array([opt[groups[k]].mean(0) for k in keys]);y=np.array([truth[k] for k in keys])
    def matrix(path):
        with path.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        t={(r['Subject_id'],r['Study_id']):r for r in rr};assert set(t)==set(keys) and len(rr)==len(keys)
        return np.array([[float(t[k][f'probability_{j}']) for j in range(1,11)] for k in keys])
    root=Path(cfg['project']); historical=.75*matrix(root/'outputs/submissions/E08_EMA_20261020/val_predictions.csv')+.25*matrix(root/'outputs/experiments/s17_f3_20261021/validation_analysis/study_predictions.csv')
    assert abs(auc(y,historical).mean()-.8145689350883447)<1e-12
    def comparison(x,reference):
        d=abs(x-reference)
        return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),p99_abs=float(np.quantile(d,.99)),
                    threshold_changed_studies=int(np.any((x>=.5)!=(reference>=.5),axis=1).sum()),
                    ranking_changed_studies=int(np.any(np.argsort(x,axis=1,kind='stable')!=np.argsort(reference,axis=1,kind='stable'),axis=1).sum()))
    new_vs_baseline=comparison(b,a);new_vs_old=comparison(b,historical)
    exact=bool(np.array_equal(a,b));status='EXACT_MATCH' if exact else ('PASS_1E4' if new_vs_baseline['max_abs']<=cfg['probability_tolerance'] else 'FAIL')
    result=dict(studies=859,images=1422,baseline_auc=float(auc(y,a).mean()),optimized_auc=float(auc(y,b).mean()),
        historical_auc=float(auc(y,historical).mean()),optimized_vs_batch1_baseline=new_vs_baseline,
        optimized_vs_historical_submission_val=new_vs_old,input_max_abs=max_tensor,input_mean_abs=tensor_sum/tensor_count,
        changed_input_images=changed_inputs,changed_output_images=changed_images,status=status,
        tolerance_fixed_before_run=cfg['probability_tolerance'],t4_status='T4_UNVERIFIED')
    if exact:result['paired_auc_delta_ci95']=[0.,0.]
    for tag,p in [('baseline',a),('optimized',b)]:
        savecsv(out/f'{tag}_val_predictions.csv',[dict(Subject_id=k[0],Study_id=k[1],**{f'probability_{j}':float(v) for j,v in enumerate(pp,1)}) for k,pp in zip(keys,p)])
    savecsv(out/'per_class_auc.csv',[dict(class_id=j+1,baseline=float(auc(y,a)[j]),optimized=float(auc(y,b)[j]),historical=float(auc(y,historical)[j]),positive_studies=int(y[:,j].sum())) for j in range(10)])
    (out/'full_val_consistency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    models=load_models(cfg)
    entries={mode:Entry(models,mode) for mode in ('baseline','gray_first','single_copy')}
    package=Path(cfg['e08_package']);core=load_module(package.parents[1]/'benchmark_core.py','six_model_benchmark_core')
    # Fixed sample preflight before timing; no tuning based on results.
    sample=next(iter(sorted((Path(cfg['val_samples'])/'pictures_100').glob('*.jpg'))))
    p0=entries['baseline'].predict_image(sample)
    for name,entry in entries.items():
        delta=float(abs(entry.predict_image(sample)-p0).max())
        if delta>cfg['probability_tolerance']:raise ValueError(f'{name} smoke probability mismatch {delta}')
    graph_status={}
    try:
        entries['cuda_graph']=Entry(models,'cuda_graph')
        delta=float(abs(entries['cuda_graph'].predict_image(sample)-p0).max())
        graph_status=dict(status='CAPTURED',sample_max_abs=delta)
        if delta>cfg['probability_tolerance']:
            del entries['cuda_graph'];graph_status['status']='REJECTED_NUMERIC_MISMATCH'
    except Exception as exc:
        graph_status=dict(status='BLOCKED',error=repr(exc));print('GRAPH_BLOCKED '+repr(exc),flush=True)
        if 'cuda_graph' in entries:del entries['cuda_graph']
    (out/'cuda_graph_preflight.json').write_text(json.dumps(graph_status,indent=2),encoding='utf-8')
    reports=stage_benchmark(cfg,out,core,entries)
    best=min(reports,key=lambda r:r['mean_ms'])['candidate']
    print('SELECTED_FOR_FULL_VAL '+best,flush=True)
    if best=='baseline':
        best='gray_first' # Check equivalent preprocessing without claiming a speed gain.
    validation=validate_full(cfg,out,entries['baseline'],entries[best])
    # Repeat unchanged baseline and best to check that the gain is reproducible.
    repeated=stage_benchmark(cfg,out,core,{'baseline_repeat':entries['baseline'],'winner_repeat':entries[best]})
    savecsv(out/'benchmark_comparison.csv',reports+repeated)
    result=dict(winner=best,validation=validation,benchmarks=reports+repeated,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                graph=graph_status,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                tensorrt_status='BLOCKED_NOT_INSTALLED; supplied engines target T4/Linux, not this 4060/Windows',
                source_package_sha256=cfg['phase65_predict_sha256'],t4_status='T4_UNVERIFIED')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    manifest['source_code']=sha(Path(__file__))
    (out/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('OPTIMIZATION_COMPLETE',flush=True)


if __name__=='__main__':main()
