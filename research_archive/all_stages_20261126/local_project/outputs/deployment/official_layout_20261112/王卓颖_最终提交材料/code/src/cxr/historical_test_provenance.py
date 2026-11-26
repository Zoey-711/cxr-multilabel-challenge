"""Read-only replay of documented historical batched inference on official Test.

Provenance only: batch 8/16 MUST NOT be presented as official batch=1 timing.
"""
from __future__ import annotations
import argparse,csv,io,json,time,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cxr.data.archive import read_split_rows,image_member
from cxr.data.dataset import preprocess_image
from cxr.e08_s17_acceleration import NAMES,load_models,fuse
from cxr.historical_val_path_recovery import BATCHES
from cxr.prediction_consistency import compare_studies,read_submission,sha256,write_csv


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    result=json.loads((Path(cfg['output'])/'result.json').read_text(encoding='utf-8'))
    restored=result['reports'][0]
    if restored['policy']!='historical_batches' or not restored['exact_historical_match']:
        raise ValueError('Historical Val path did not exactly restore; Test provenance prohibited')
    if cfg['historical_batches']!=BATCHES:
        raise ValueError('Unexpected documented batch settings')
    for source in cfg['locked_sources']:
        if sha256(source['path'])!=source['sha256']:raise ValueError(f"Changed source: {source['path']}")
    output=Path(cfg['output']).parent/'historical_test_provenance';output.mkdir(exist_ok=False)
    lock=dict(purpose='provenance only; not official batch1 implementation',
              val_result_sha256=sha256(Path(cfg['output'])/'result.json'),
              config_sha256=sha256(args.config),code_sha256=sha256(__file__),
              batches=BATCHES,policy='historical_batches')
    (output/'protocol_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    reference_file=Path(cfg['project'])/'outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv'
    expected='8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c'
    if sha256(reference_file)!=expected:raise ValueError('Historical submission changed')
    reference=read_submission(reference_file)
    rows=read_split_rows(cfg['archive'],'test');groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['Subject_id'],r['Study_id'])].append(i)
    if len(rows)!=2218 or set(groups)!=set(reference):raise ValueError('Incorrect Test IDs')
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    models=load_models(cfg);allvalues=[];start=time.perf_counter()
    with zipfile.ZipFile(cfg['archive']) as archive:
        for offset in range(0,len(rows),16):
            selected=rows[offset:offset+16];tensors=[]
            for r in selected:
                with Image.open(io.BytesIO(archive.read(image_member(r,'test')))) as im:
                    tensors.append(preprocess_image(im,384,preprocessing='black_pad_v1'))
            parts=[]
            for name in NAMES:
                values=[]
                for j in range(0,len(tensors),BATCHES[name]):
                    x=torch.stack(tensors[j:j+BATCHES[name]]).cuda()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        v=torch.sigmoid(models[name](x)).float().cpu().numpy().astype(np.float64)
                    values.extend(v)
                parts.append(np.array(values))
            allvalues.extend(np.stack(parts,axis=1))
            if offset%320==0:print(f'HISTORICAL_TEST {min(offset+16,2218)}/2218',flush=True)
    values=np.array(allvalues)
    parts={n:{k:values[ii,j].mean(0) for k,ii in groups.items()} for j,n in enumerate(NAMES)}
    actual={k:fuse([parts[n][k] for n in NAMES]) for k in groups}
    comparison,details=compare_studies(actual,reference)
    # Separately compare with the previously frozen component Study outputs,
    # making it clear which historical source cannot be re-created locally.
    refs={
        'E03':Path(cfg['project'])/'outputs/submissions/E03_20261017/probability_matrix.csv',
        'N01_EMA':Path(cfg['project'])/'outputs/submissions/E08_EMA_20261020/n01_ema_test_predictions.csv',
        'S17_F3':Path(cfg['project'])/'outputs/submissions/E08_075_S17_F3_025_20261110/s17_f3_test_predictions.csv',
    }
    source_comparison={}
    for name,path in refs.items():
        if not path.is_file():
            source_comparison[name]={'status':'UNAVAILABLE'};continue
        with path.open(encoding='utf-8-sig',newline='') as stream:
            records=list(csv.DictReader(stream))
        if name=='E03':
            present={k:.5*((parts['B01'][k]+parts['B02'][k]+parts['B03'][k])/3)+.5*parts['M02'][k] for k in groups}
        else:present=parts[name]
        first=records[0]
        fields=[f'probability_{j}' if f'probability_{j}' in first else f'Prob_{j}' for j in range(1,11)]
        table={(r['Subject_id'],r['Study_id']):np.array([float(r[f]) for f in fields]) for r in records}
        if len(table)!=1414 or set(table)!=set(groups):
            source_comparison[name]={'status':'INVALID_KEYS','path':str(path)};continue
        report,_=compare_studies(present,table)
        source_comparison[name]=dict(report,path=str(path),sha256=sha256(path))
    write_csv(output/'study_predictions.csv',[dict(Subject_id=k[0],Study_id=k[1],**{f'probability_{j}':float(v) for j,v in enumerate(p,1)}) for k,p in sorted(actual.items())])
    write_csv(output/'comparison.csv',details)
    report=dict(purpose='historical provenance only; batch sizes 8/16 are not official batch1',
                reference_sha256=expected,images=2218,studies=1414,comparison=comparison,
                historical_components=source_comparison,seconds=time.perf_counter()-start,
                no_test_parameter_fit=True,t4_status='T4_UNVERIFIED',
                official_batch1_eligibility=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'manifest_sha256.json').write_text(json.dumps({p.name:sha256(p) for p in output.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
