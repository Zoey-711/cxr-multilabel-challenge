"""Private one-device worker; use phase42_run.py for budget/preflight enforcement."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=('smoke','train','baseline'));p.add_argument('--candidate',required=True)
    p.add_argument('--fold',required=True,type=int);p.add_argument('--gpu',required=True)
    p.add_argument('--probe',type=Path);p.add_argument('--identity')
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import torch
    torch.set_num_threads(2)
    from cxr.phase42.protocol import load_plan,server_config,task,fingerprint
    from cxr.phase42.train import real_smoke,run_task
    plan=load_plan();cfg=server_config(args.config);item=task(plan,args.candidate,args.fold)
    if not torch.cuda.is_available():raise RuntimeError('Remote worker requires an authorized CUDA device')
    torch.cuda.set_per_process_memory_fraction(min(1.,cfg['max_vram_gib']*2**30/torch.cuda.get_device_properties(0).total_memory),0)
    if args.mode=='baseline':
        from cxr.phase42.protocol import verify_baseline,digest
        from cxr.phase42.models import build_model
        from cxr.phase42.benchmark import benchmark
        from cxr.data.dataset import make_dataloader
        from cxr.phase39_support import write_json
        folder,_=verify_baseline(plan,cfg,2)
        model=build_model('BASE').cuda()
        model.load_state_dict(torch.load(folder/'ema_best.pth',map_location='cpu',weights_only=True))
        args.probe.mkdir(parents=True,exist_ok=True)
        loader=make_dataloader(Path(cfg['data_root']),'val',batch_size=4,num_workers=0,input_size=384,preprocessing='black_pad_v1')
        benchmark(model,loader.dataset,args.probe,{})
        write_json(args.probe/'receipt.json',dict(state='PASS',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            source_sha256=digest(folder/'ema_best.pth'),sha256={p.name:digest(p) for p in args.probe.iterdir() if p.name in ('compute_cost.json','inference_benchmark.json')}))
    elif args.mode=='smoke':print(json.dumps(real_smoke(plan,item,cfg,args.probe)))
    else:
        receipt=Path(cfg['output_root'])/'preflight/REMOTE_PREFLIGHT.json'
        state=json.loads(receipt.read_text())
        if state['state']!='PASS' or state['identity_sha256']!=args.identity:raise ValueError('Valid remote preflight required')
        if args.fold!=2:
            selection=json.loads((Path(cfg['output_root'])/'reports/selection.json').read_text())
            if not cfg['confirmation']['enabled'] or args.candidate not in selection['selected']:raise ValueError('Confirmation not authorized/selected')
        print(json.dumps(run_task(plan,item,cfg,args.identity)))


if __name__=='__main__':main()
