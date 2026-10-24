"""Batch-one real Val image decode-to-probability latency; no training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cxr.data.archive import archive_path,read_split_rows
from cxr.phase32.dataset import CXRArchiveDataset
from cxr.utils.checkpoint import load_model_state
from cxr.phase32.config import load_config

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--checkpoint-kind',choices=('raw','ema'),default='raw')
    a=p.parse_args();c=load_config(a.config)
    model=load_model_state(a.run_dir/('best_model.pt' if a.checkpoint_kind=='raw' else 'best_ema_model.pt'),device='cuda').eval()
    data=archive_path(c);rows=read_split_rows(data,'val')[:23]
    ds=CXRArchiveDataset(data,'val',rows=rows,preprocessing=c['data'].get('preprocessing','black_pad_v1'),input_size=c['data']['input_size'])
    times=[]
    with torch.inference_mode():
        for i in range(len(rows)):
            torch.cuda.synchronize();start=time.perf_counter();sample=ds[i]
            with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(model(sample['image'][None].cuda())).float().cpu()
            torch.cuda.synchronize()
            if i>=3:times.append((time.perf_counter()-start)*1000)
    ds.close()
    (a.run_dir/('latency.json' if a.checkpoint_kind=='raw' else 'ema_latency.json')).write_text(json.dumps({'device':torch.cuda.get_device_name(0),'batch':1,'samples':len(times),
      'warmup':3,'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
      'scope':'ZIP read/decode + deterministic preprocessing + transfer + raw model + sigmoid + CPU transfer; no T4 claim; not whole ensemble latency'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
