"""Research-only full-candidate timing, including ZIP decode and every model.

This is not an official compliance measurement until the organiser confirms its
hardware, precision, batch and timing statistic. The script never marks PASS.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cxr.batch.plan import digest, write_json
from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.models.factory import create_model
from cxr.utils.config import load_config

APPROVED_RESEARCH_CANDIDATES = ('E17_F01', 'D5_C1_S1', 'N17_F01',
                                'CS17_F01', 'NM17_F01')


def model_weights(name, plan):
    formula = plan.get('deployment_candidates', {}).get(name)
    if formula is None:
        formula = plan.get('fixed_fusions', {}).get(name)
    if formula is None and name in ('D5','C5','S5','N5'):
        branch={'D5':'D17','C5':'C17','S5':'S17','N5':'N17'}[name]
        formula={f'{branch}_F{i:02d}':.2 for i in range(1,6)}
    if formula is None:return {name:1.0}
    result={}
    for source,weight in formula.items():
        for model,part in model_weights(source,plan).items():
            result[model]=result.get(model,0)+weight*part
    return result


def timed_candidate(root, candidate, plan, *, warmup=3, measured=20):
    if not torch.cuda.is_available():
        raise RuntimeError('An authorized CUDA GPU is required')
    if candidate['bytes'] > 500_000_000:
        raise ValueError('Do not spend GPU time on a confirmed size failure')
    dependencies = candidate['dependencies']
    if not dependencies or any(not d['path'].startswith('experiments/') for d in dependencies):
        raise ValueError('Original frozen reference weights are unavailable in this package')
    weights=model_weights(candidate['candidate'],plan)
    models_in_files={Path(d['path']).parts[1] for d in dependencies}
    if set(weights)!=models_in_files or abs(sum(weights.values())-1)>1e-8:
        raise ValueError('Candidate inference weights do not match frozen fusion formula')
    entries = []
    datasets = {}
    archive = None
    for dependency in dependencies:
        path = root/dependency['path']
        if path.stat().st_size != dependency['bytes'] or digest(path) != dependency['sha256']:
            raise ValueError('Candidate checkpoint hash/size mismatch')
        cfg = load_config(path.parent/'resolved_config.yaml')
        if archive is None:
            archive = archive_path(cfg)
            rows = read_split_rows(archive, 'val')[:warmup+measured]
            if len(rows) < warmup+measured:
                raise ValueError('Insufficient official Val images')
        elif archive_path(cfg) != archive:
            raise ValueError('Models use different data archives')
        key = (cfg['data'].get('preprocessing', 'black_pad_v1'),cfg['data']['input_size'])
        if key not in datasets:
            datasets[key] = CXRArchiveDataset(archive,'val',rows=rows,
                preprocessing=key[0],input_size=key[1])
        model = create_model(cfg['model']['name'],pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        entries.append((model,key,weights[path.parent.name]))
    times = []
    try:
        with torch.inference_mode():
            for i in range(warmup+measured):
                torch.cuda.synchronize()
                start=time.perf_counter()
                inputs={key:ds[i]['image'][None].cuda() for key,ds in datasets.items()}
                probabilities=[]
                with torch.autocast('cuda',dtype=torch.float16):
                    for model,key,weight in entries:
                        probabilities.append(weight*torch.sigmoid(model(inputs[key])).float().cpu().numpy()[0])
                # Include 10-class multi-model probability reduction in elapsed time.
                fused=np.sum(np.stack(probabilities,axis=0),axis=0)
                if fused.shape!=(10,) or not np.isfinite(fused).all():
                    raise ValueError('Non-finite candidate probability')
                torch.cuda.synchronize()
                if i>=warmup:times.append((time.perf_counter()-start)*1000)
    finally:
        for ds in datasets.values():ds.close()
    return {'candidate':candidate['candidate'],'device':torch.cuda.get_device_name(0),
            'cuda_runtime':torch.version.cuda,'precision':'fp16_amp','batch_size':1,
            'warmup':warmup,'measured_samples':measured,'model_count':len(entries),
            'checkpoint_sha256':sorted(d['sha256'] for d in dependencies),
            'fusion_model_weights':weights,
            'scope':'ZIP read/decode, deterministic preprocessing, host-to-device, ALL sequential model forwards, sigmoid, CPU transfer, probability reduction, synchronize',
            'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),
            'p95_ms':float(np.quantile(times,.95)),'max_ms':float(np.max(times)),
            'official_compliance':False,
            'reason':'Official target hardware and timing scope/statistic have not been fully specified'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--candidates',nargs='+',choices=APPROVED_RESEARCH_CANDIDATES,
                   default=list(APPROVED_RESEARCH_CANDIDATES))
    args=p.parse_args()
    plan=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    root=args.root.resolve()
    records=json.loads((root/'deployment/deployment_candidates.json').read_text(encoding='utf-8'))
    lookup={r['candidate']:r for r in records}
    output=root/'deployment/research_benchmarks';output.mkdir(exist_ok=True)
    for name in args.candidates:
        if name not in lookup or lookup[name]['bytes']>plan['deployment_protocol']['weight_limit_bytes']:
            continue
        target=output/(name+'.json')
        if target.exists():continue
        result=timed_candidate(root,lookup[name],plan)
        write_json(target,result)
        print(f'{name}: {result["mean_ms"]:.3f} ms research full pipeline, {result["device"]}',flush=True)

if __name__=='__main__':main()
