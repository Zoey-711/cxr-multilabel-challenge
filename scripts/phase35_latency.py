"""Reference batch-one end-to-end latency and Conv/Linear MAC count; T4_UNVERIFIED."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from cxr.data.archive import archive_path,read_split_rows
from cxr.phase32.dataset import CXRArchiveDataset
from cxr.phase32.config import load_config
from cxr.utils.checkpoint import load_model_state

def convolution_linear_macs(model,x):
    total=[0];hooks=[]
    def count(m,inputs,out):
        if isinstance(m,torch.nn.Conv2d):total[0]+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,torch.nn.Linear):total[0]+=out.numel()*m.in_features
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(m.register_forward_hook(count))
    try:
        with torch.inference_mode():model(x)
    finally:
        for h in hooks:h.remove()
    return total[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--checkpoint-kind',choices=['raw','ema'],default='raw');a=p.parse_args();cfg=load_config(a.config)
    model=load_model_state(a.run_dir/('best_model.pt' if a.checkpoint_kind=='raw' else 'best_ema_model.pt'),device='cuda').eval()
    archive=archive_path(cfg);rows=read_split_rows(archive,'val')[:25]
    ds=CXRArchiveDataset(archive,'val',rows=rows,preprocessing=cfg['data'].get('preprocessing','black_pad_v1'),input_size=cfg['data']['input_size'])
    times=[];prefix='' if a.checkpoint_kind=='raw' else 'ema_'
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            macs=convolution_linear_macs(model,ds[0]['image'][None].cuda())
            for round_id in range(3):
                for i in range(25):
                    torch.cuda.synchronize();start=time.perf_counter();sample=ds[i]
                    probs=torch.sigmoid(model(sample['image'][None].cuda())).float().cpu().numpy()[0]
                    with (a.run_dir/(prefix+'latency_last.csv')).open('w',newline='') as f:
                        w=csv.writer(f);w.writerow([f'Prob_{j}' for j in range(1,11)]);w.writerow(probs)
                    torch.cuda.synchronize()
                    if i>=5:times.append((time.perf_counter()-start)*1000)
    finally:ds.close()
    data={'device':torch.cuda.get_device_name(0),'precision':'FP16 autocast','batch':1,'input_size':cfg['data']['input_size'],
          'conv_linear_macs':macs,'mac_scope':'Conv2d and Linear only; BN/activation/pooling/IO excluded from operation count',
          'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'max_ms':float(np.max(times)),
          'warmup_per_round':5,'timed_images_per_round':20,'rounds':3,'image_ids':[r['Dicom_id'] for r in rows],
          'scope':'ZIP read/decode, resize/pad/normalize, transfer, forward, sigmoid, CPU transfer, CSV write/close; synchronized wall time',
          'status':'T4_UNVERIFIED','official_protocol_pass':False}
    (a.run_dir/(prefix+'latency.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
if __name__=='__main__':main()
