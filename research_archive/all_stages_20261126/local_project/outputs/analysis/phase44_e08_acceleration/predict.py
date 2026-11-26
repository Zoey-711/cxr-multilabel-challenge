"""Independent E08 ONNX candidate. Fixed five models, FP16 and fusion formula."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE/'isolated_deps'),str(ROOT/'src')]
import numpy as np
import onnxruntime as ort
from PIL import Image
from cxr.data.archive import image_member, read_split_rows
from cxr.data.dataset import preprocess_image

NAMES = ('B01','B02','B03','M02','N01_EMA')
FROZEN = ROOT/'outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_models(provider='CUDAExecutionProvider',precision='fp16'):
    if precision not in ('fp16','fp32'):raise ValueError('precision must be fp16 or fp32')
    source=json.loads(FROZEN.read_text(encoding='utf-8'))
    exported=json.loads((HERE/'onnx_export_manifest.json').read_text(encoding='utf-8'))
    if provider=='CUDAExecutionProvider':ort.preload_dlls()
    if provider not in ort.get_available_providers():raise RuntimeError(f'Provider unavailable: {provider}')
    sessions={}
    for name in NAMES:
        original=ROOT/'outputs/experiments'/source['models'][name]['run_id']/source['models'][name]['checkpoint']
        if sha(original)!=source['models'][name]['checkpoint_sha256'] or sha(original)!=exported[name]['source_sha256']:
            raise ValueError(f'Frozen model hash changed: {name}')
        graph=HERE/'onnx_models'/f'{name}_{precision}.onnx'
        if sha(graph)!=exported[name][f'{precision}_sha256']:raise ValueError(f'Exported graph hash changed: {name}')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        session=ort.InferenceSession(str(graph),sess_options=options,providers=[provider])
        if session.get_providers()[0]!=provider:raise RuntimeError(f'Unexpected provider fallback: {name}')
        sessions[name]=session
    return sessions

def from_image(image,models,stage=None):
    start=time.perf_counter()
    tensor=preprocess_image(image,384,preprocessing='black_pad_v1').unsqueeze(0).numpy()
    if stage is not None:stage['preprocess']=(time.perf_counter()-start)*1000
    parts={}
    for name,session in models.items():
        start=time.perf_counter()
        logits=session.run(None,{'image':tensor})[0][0].astype(np.float64)
        parts[name]=1/(1+np.exp(-logits))
        if stage is not None:stage[f'forward_{name}']=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    e01=(parts['B01']+parts['B02']+parts['B03'])/3
    fused=.5*(.5*e01+.5*parts['M02'])+.5*parts['N01_EMA']
    if fused.shape!=(10,) or not np.isfinite(fused).all() or (fused<0).any() or (fused>1).any():
        raise ValueError('Invalid fused probability')
    if stage is not None:stage['fusion']=(time.perf_counter()-start)*1000
    return fused

def predict_image(path,models,stage=None):
    if stage is None:
        with Image.open(path) as image:return from_image(image,models)
    start=time.perf_counter()
    raw=Path(path).read_bytes()
    stage['file_read']=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        stage['decode']=(time.perf_counter()-start)*1000
        return from_image(image,models,stage)

def validation(archive, output, models):
    rows=read_split_rows(archive,'val');grouped={}
    if len(rows)!=1422:raise ValueError('Unexpected Val image count')
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            with Image.open(io.BytesIO(source.read(image_member(row,'val')))) as image:
                p=from_image(image,models)
            grouped.setdefault((row['Subject_id'],row['Study_id']),[]).append(p)
    if len(grouped)!=859:raise ValueError('Unexpected Val Study count')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    with output.open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['Subject_id','Study_id']+[f'Prob_{i}' for i in range(1,11)])
        for key,values in sorted(grouped.items()):writer.writerow([*key,*[format(float(x),'.17g') for x in np.mean(values,axis=0)]])
    return dict(images=len(rows),studies=len(grouped),sha256=sha(output))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',default='CUDAExecutionProvider',choices=('CUDAExecutionProvider','CPUExecutionProvider'))
    p.add_argument('--precision',default='fp16',choices=('fp16','fp32'))
    p.add_argument('--val-archive',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    sessions=load_models(args.provider,args.precision)
    if args.val_archive:
        if args.output is None:p.error('--output required with --val-archive')
        print(json.dumps(validation(args.val_archive,args.output,sessions)))
    else:
        example=ROOT/'outputs/analysis/phase29_deployment/benchmark_images/pictures_100/pic_033077.jpg'
        print(json.dumps(dict(provider=args.provider,example_probability=from_image(Image.open(example),sessions).tolist())))
