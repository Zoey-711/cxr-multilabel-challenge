"""Measure an exact deployment candidate only under a confirmed official protocol.

If hardware, timing scope or statistic differ, evidence is saved as research only.
No candidate is marked PASS by this script alone; CPU report verifies accuracy too.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torchvision

from cxr.batch.plan import digest, write_json
from batch_ensemble_latency import timed_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--official-protocol',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    plan=__import__('yaml').safe_load(args.config.read_text(encoding='utf-8'))
    protocol=json.loads(args.official_protocol.read_text(encoding='utf-8'))
    if protocol.get('confirmed') is not True or not protocol.get('official_source'):
        raise ValueError('Official protocol has not been confirmed')
    canonical=json.dumps({k:v for k,v in protocol.items() if k!='sha256'},sort_keys=True,
                         ensure_ascii=False,separators=(',',':')).encode('utf-8')
    if protocol.get('sha256') != hashlib.sha256(canonical).hexdigest():
        raise ValueError('Official protocol canonical SHA-256 mismatch')
    records=json.loads((root/'deployment/deployment_candidates.json').read_text(encoding='utf-8'))
    candidate=next((r for r in records if r['candidate']==args.candidate),None)
    if candidate is None:
        raise ValueError('Unknown reviewed deployment candidate')
    result=timed_candidate(root,candidate,plan,
        warmup=int(protocol['warmup_samples']),measured=int(protocol['measured_samples']))
    statistic=protocol['statistic']
    if statistic not in ('mean_ms','p50_ms','p95_ms','max_ms'):
        raise ValueError('Unsupported official statistic')
    measured={'candidate':args.candidate,'hardware':torch.cuda.get_device_name(0),
              'precision':'fp16_amp','batch_size':1,'timing_scope':result['scope'],
              'statistic':statistic,'warmup_samples':result['warmup'],
              'measured_samples':result['measured_samples'],
              'software_environment':{'python':platform.python_version(),'torch':torch.__version__,
                                      'torchvision':torchvision.__version__,'cuda':torch.version.cuda},
              'checkpoint_sha256':result['checkpoint_sha256'],
              'inference_recipe_sha256':candidate['inference_recipe_sha256'],
              'full_pipeline':True,'latency_ms':result[statistic],
              'raw_research_result':result}
    from cxr.batch.deployment import PROTOCOL_FIELDS
    if any(measured[key]!=protocol[key] for key in PROTOCOL_FIELDS):
        measured['official_match']=False
        folder=root/'deployment/research_benchmarks'
    else:
        measured['official_match']=True
        measured['protocol_sha256']=protocol['sha256']
        folder=root/'deployment/official_benchmarks'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(args.candidate+'.json')
    if path.exists():
        raise FileExistsError(path)
    write_json(path,measured)
    print(json.dumps({'candidate':args.candidate,'official_match':measured['official_match'],
                      'latency_ms':measured['latency_ms'],'output':str(path)}))


if __name__=='__main__':
    main()
