"""Friend-cluster preflight. Static mode never uses GPU; GPU mode explicitly requested."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cxr.batch.plan import ROOT, load_plan, digest, write_json, gpu_tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpus',default='0')
    p.add_argument('--max-concurrent',type=int,default=1)
    p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--budget-hours',type=float,help='Explicit GPU-hour cap for preflight subprocesses')
    a=p.parse_args(); plan=load_plan(a.config)
    if a.output.exists():
        raise FileExistsError(a.output)
    from cxr.data.archive import archive_path, read_split_rows
    from cxr.data.patient_folds import select_fold_rows
    from cxr.utils.config import load_config
    from cxr.utils.cloud_paths import pretrained_home
    from cxr.models.factory import imagenet_weight_info
    for key in ('CXR_DATA_ROOT','CXR_OUTPUT_ROOT','CXR_PRETRAINED_ROOT','CXR_BACKUP_ROOT'):
        if not os.environ.get(key) or not Path(os.environ[key]).is_absolute():
            raise ValueError(f'Explicit absolute {key} required')
    base=load_config(ROOT/plan['tasks'][0]['config'])
    data=archive_path(base)
    if not data.is_file() or digest(data)!=plan['data_sha256']:
        raise ValueError('Original official combined ZIP SHA-256 mismatch; do not repack silently')
    train,val=read_split_rows(data,'train'),read_split_rows(data,'val')
    try:
        test=read_split_rows(data,'test')
        test_state={'state':'available','images':len(test),'studies':len({r['Study_id'] for r in test})}
        if test_state['images']!=2218 or test_state['studies']!=1414:
            raise ValueError('Official Test images/Studies differ from 2218/1414')
    except (FileNotFoundError,KeyError):
        test_state={'state':'PENDING','reason':'Test images unavailable; training may continue, submission disabled'}
    if (len(train),len(val))!=(30344,1422):
        raise ValueError('Train/Val image count mismatch')
    if {r['Subject_id'] for r in train}&{r['Subject_id'] for r in val}:
        raise ValueError('Official Train/Val patient leakage')
    reports={}; selected={}; verified_weights={}
    for task in plan['tasks']:
        cfg=load_config(ROOT/task['config'])
        if task['fold'] is not None:
            tr=select_fold_rows(train,cfg['data']['train_subset'],role='train')
            oo=select_fold_rows(train,cfg['data']['train_subset'],role='oof')
            key=task['fold']; signature=sorted((r['Subject_id'],r['Study_id'],r['Dicom_id']) for r in oo)
            if key in selected and signature!=selected[key]:
                raise ValueError('Backbones have different OOF rows')
            selected[key]=signature
            reports[task['id']]={'train_images':len(tr),'oof_images':len(oo)}
        _,url,name=imagenet_weight_info(cfg['model']['name'])
        weights=pretrained_home(ROOT)/'hub'/'checkpoints'/name
        if not cfg['model'].get('pretrained_sha256'):
            raise ValueError(f'Unsealed ImageNet SHA-256 for {task["id"]}; obtain official weights, audit and seal BEFORE launch')
        if str(weights) not in verified_weights:
            verified_weights[str(weights)]=digest(weights)
        measured_sha=verified_weights[str(weights)]
        if measured_sha!=cfg['model']['pretrained_sha256']:
            raise ValueError(f'ImageNet hash mismatch: {weights}')
    output_root=Path(os.environ['CXR_OUTPUT_ROOT']); output_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output_root).free<plan['minimum_free_output_bytes']:
        raise OSError('Need at least 50GB free output space, separate from dataset and backup')
    backup_root=Path(os.environ['CXR_BACKUP_ROOT']);backup_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(backup_root).free<50_000_000_000:
        raise OSError('Need at least 50GB free backup space')
    if output_root.stat().st_dev==backup_root.stat().st_dev and shutil.disk_usage(output_root).free<110_000_000_000:
        raise OSError('Output and backup share a filesystem: require 110GB free including packing reserve')
    devices=gpu_tokens(a.gpus)
    if not 1<=a.max_concurrent<=min(len(devices),8):
        raise ValueError('Single-host concurrency must be 1..min(authorized GPUs,8)')
    from cxr.batch.host_resources import check_resources
    host_resources=check_resources(data,a.max_concurrent,plan)
    a.output.mkdir(parents=True)
    report={'state':'static_passed','plan_sha256':digest(a.config),'gpus':devices,'folds':reports,
            'max_concurrent':a.max_concurrent,'host_resources':host_resources,'test':test_state,
            'host':__import__('socket').gethostname(),
            'environment':{k:os.environ[k] for k in ('CXR_DATA_ROOT','CXR_OUTPUT_ROOT','CXR_PRETRAINED_ROOT','CXR_BACKUP_ROOT')}}
    if a.gpu_smoke:
        if a.budget_hours is None or a.budget_hours<=0:
            raise ValueError('Explicit --budget-hours required for GPU preflight')
        smoke_start=time.monotonic()
        from cxr.batch.runner import run_command
        def checked(cmd,env,name):
            remaining=a.budget_hours*3600-(time.monotonic()-smoke_start)
            if remaining<=0:raise TimeoutError('Preflight GPU-hour budget exhausted')
            rc=run_command(cmd[1:],env,a.output/(name+'.stdout.log'),a.output/(name+'.stderr.log'),remaining)
            if rc:raise RuntimeError(f'Preflight {name} failed ({rc}); see logs')
        import torch,torchvision
        if sys.version_info[:2]!=(3,12) or torch.__version__.split('+')[0]!='2.10.0' or torchvision.__version__.split('+')[0]!='0.25.0' or torch.version.cuda!='12.8':
            raise RuntimeError('Require Python3.12 / torch2.10.0 / torchvision0.25.0 / CUDA12.8')
        for device in devices:
            probe_env=os.environ.copy();probe_env['CUDA_VISIBLE_DEVICES']=device
            check=subprocess.run(['nvidia-smi','-i',device,'--query-gpu=name,memory.total',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            line=check.stdout.strip().splitlines()[0]
            name,memory=line.rsplit(',',1)
            if '4090' not in name or int(memory.strip())<23000:
                raise RuntimeError(f'Expected authorized RTX 4090 24GB, got {line}')
        for device in devices:
            # One allocation smoke per GPU; remaining architectures on the first GPU only.
            smoke_tasks=[t for t in plan['tasks'] if t['fold']==1] if device==devices[0] else [plan['tasks'][0]]
            for task in smoke_tasks:
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=device,CXR_OUTPUT_ROOT=str(a.output.resolve()))
                cmd=[sys.executable,'scripts/batch_train.py','--config',task['config'],
                     '--run-id',f'preflight_{task["id"]}_{devices.index(device)}','--preflight-steps','2']
                checked(cmd,env,f'smoke_{task["id"]}_{devices.index(device)}')
                status=json.loads((a.output/'experiments'/f'preflight_{task["id"]}_{devices.index(device)}'/'status.json').read_text(encoding='utf-8'))
                if status.get('state')!='preflight_passed' or status.get('optimizer_steps')!=2 or status.get('amp_skipped_steps')!=0:
                    raise RuntimeError(f'Unstable FP16 or missing optimizer updates: {task["id"]}')
                if a.benchmark:
                    # Preserve scheduler allocation UUIDs; subprocess GPU index is relative zero.
                    cmd=[sys.executable,'scripts/batch_benchmark.py','--config',task['config'],
                         '--output',str(a.output.resolve()/f'benchmark_{task["id"]}_{devices.index(device)}.json')]
                    checked(cmd,env,f'benchmark_{task["id"]}_{devices.index(device)}')
            checked([sys.executable,'-m','pytest','-q','tests/test_training_resume.py'],env,f'resume_{devices.index(device)}')
        report['state']='passed'
        report['gpu_hours']=(time.monotonic()-smoke_start)/3600
        report['framework']={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
                             'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
    write_json(a.output/'receipt.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
