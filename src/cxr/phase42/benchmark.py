"""Analytic Conv/Linear/attention MACs and non-official end-to-end reference."""
import csv
import io
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from cxr.data.archive import image_member
from cxr.data.dataset import preprocess_image
from cxr.phase39_support import write_json
from cxr.phase42.models import Candidate, build_model


def profile(model):
    counts={'conv_linear_macs':0,'attention_macs':0}
    hooks=[]
    def hook(module, inputs, output):
        if isinstance(module,nn.Conv2d):
            counts['conv_linear_macs']+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,nn.Conv1d):
            counts['conv_linear_macs']+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module,nn.Linear): counts['conv_linear_macs']+=output.numel()*module.in_features
        elif isinstance(module,nn.MultiheadAttention):
            b,q,d=inputs[0].shape;k=inputs[1].shape[1]
            counts['attention_macs']+=b*((2*q+2*k)*d*d+2*q*k*d)
    # MHA uses functional projections; do not double count its children.
    mha_children={id(x) for m in model.modules() if isinstance(m,nn.MultiheadAttention) for x in m.modules() if x is not m}
    for m in model.modules():
        if id(m) not in mha_children and isinstance(m,(nn.Conv2d,nn.Conv1d,nn.Linear,nn.MultiheadAttention)):
            hooks.append(m.register_forward_hook(hook))
    was=model.training;model.eval()
    try:
        with torch.no_grad(): model(torch.zeros(1,3,384,384,device=next(model.parameters()).device))
    finally:
        for h in hooks:h.remove()
        model.train(was)
    if isinstance(model,Candidate) and model.variant in ('B04','C01'):
        counts['attention_macs']+=(1 if model.variant=='B04' else 10)*1280*144
    counts.update(parameters=sum(p.numel() for p in model.parameters()),input_size=384,
        macs=sum(counts.values()),scope='Conv/Linear + attention matmuls; excludes pooling, normalization, activation, decode; not full FLOPs')
    return counts


def benchmark(model,dataset,run,cfg):
    """Same fixed Val images for all candidates; does not select architecture."""
    model.eval(); rows=dataset.rows[:105]
    if len(rows)<105: raise ValueError('Need fixed 105 Val images for reference timing')
    durations=[]
    # ZIP opening is outside the timer; each read/decode/resize/transfer/forward/
    # probability copy/CSV close is inside. This is a declared research protocol.
    with zipfile.ZipFile(dataset.archive) as archive:
        def predict(row):
            with Image.open(io.BytesIO(archive.read(image_member(row,'val')))) as im:
                x=preprocess_image(im,384,preprocessing='black_pad_v1').unsqueeze(0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                p=torch.sigmoid(model(x.cuda())).float().cpu().tolist()[0]
            with (run/'benchmark_last_prediction.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f);writer.writerow(['Subject_id','Study_id',*[f'Prob_{i}' for i in range(1,11)]])
                writer.writerow([row['Subject_id'],row['Study_id'],*p])
        for row in rows[:5]:predict(row)
        for repeat in range(3):
            for row in rows[5:]:
                torch.cuda.synchronize();start=time.perf_counter();predict(row);torch.cuda.synchronize()
                durations.append((time.perf_counter()-start)*1000)
    write_json(run/'inference_benchmark.json',dict(gpu=torch.cuda.get_device_name(),batch_size=1,precision='FP16 autocast',
        warmup=5,images=100,repeats=3,mean_ms=float(np.mean(durations)),p50_ms=float(np.percentile(durations,50)),
        p95_ms=float(np.percentile(durations,95)),max_ms=max(durations),status='T4_UNVERIFIED',
        timing_scope='Open archive excluded; ZIP member read through CSV close included; OS cache may apply',
        contention='Other authorized training workers may run; reference only, not a selection input'))
    write_json(run/'compute_cost.json',profile(model))


def inventory(output):
    from cxr.phase42.models import IDS
    torch.set_num_threads(2)
    base=profile(build_model('BASE'));rows=[]
    for variant in IDS:
        model=build_model(variant);cost=profile(model)
        rows.append(dict(candidate=variant,**cost,added_parameters=cost['parameters']-base['parameters'],
            mac_ratio=cost['macs']/base['macs'],weight_fp32_tensor_bytes=4*cost['parameters'],
            persistent_training_tensor_lower_bound_bytes=20*cost['parameters'],
            lower_bound_scope='FP32 model+gradient+Adam first/second moments+EMA; excludes activations/buffers/workspace, not peak VRAM',
            expected_vram='PENDING_REMOTE_MEASUREMENT',remote_vram_cap_gib=20))
        del model
    write_json(Path(output),dict(baseline=base,candidates=rows,training_gpu_used=False))
    return rows
