"""Six-model, one-image-at-a-time T4 benchmark. Imported by package cost.py."""
from __future__ import annotations
import argparse,csv,hashlib,json,statistics,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'phase65_source'))
sys.path.insert(0,str(ROOT/'phase65_source/src'))
sys.path.insert(0, str(ROOT))
import predict
from cxr.prediction_consistency import compare_studies,read_submission,probabilities,sha256


def images(folder,count):
    paths=sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if len(paths)!=count or len({p.name for p in paths})!=count:raise ValueError(f'Expected {count} distinct image names')
    return paths


def sample_index(path,measured):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['image','Subject_id','Study_id','study_image_count']:raise ValueError('Bad index schema')
        rows=list(reader)
    names={p.name for p in measured}
    if len(rows)!=100 or {r['image'] for r in rows}!=names:raise ValueError('Index must map exactly the timed 100 images')
    by_image={r['image']:r for r in rows}
    if len(by_image)!=100:raise ValueError('Duplicate image mapping')
    grouped={};total={}
    for r in rows:
        key=r['Subject_id'],r['Study_id'];count=int(r['study_image_count'])
        if not all(key) or count<1:raise ValueError('Invalid index identifiers/count')
        if key in total and total[key]!=count:raise ValueError('Inconsistent Study image count')
        total[key]=count;grouped.setdefault(key,[]).append(r['image'])
    incomplete={k:(len(v),total[k]) for k,v in grouped.items() if len(v)!=total[k]}
    return grouped,incomplete


def verify_manifest(root):
    manifest=json.loads((root/'manifest_sha256.json').read_text(encoding='utf-8'))
    for relative,expected in manifest.items():
        path=root/relative
        if not path.is_file() or sha256(path)!=expected:raise ValueError(f'Package hash mismatch: {relative}')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('submission_pos',type=Path,nargs='?')
    ap.add_argument('pictures_100_pos',type=Path,nargs='?')
    ap.add_argument('pictures_5_pos',type=Path,nargs='?')
    ap.add_argument('--config',type=Path,default=ROOT/'config.json')
    ap.add_argument('--pictures-100',type=Path)
    ap.add_argument('--pictures-5',type=Path)
    ap.add_argument('--image-index',type=Path)
    ap.add_argument('--submission',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT.parent/('cost_'+str(time.time_ns())))
    ap.add_argument('--official-sample',action='store_true')
    args=ap.parse_args()
    for key in ('submission','pictures_100','pictures_5'):
        flag=getattr(args,key); pos=getattr(args,key+'_pos')
        if flag and pos: ap.error('Use positional or named argument, not both: '+key)
        setattr(args,key,flag or pos)
        if not getattr(args,key): ap.error('Missing '+key)
    if args.config.resolve() != (ROOT/'config.json').resolve(): raise ValueError('Use frozen config')
    if args.output.resolve().is_relative_to(ROOT): raise ValueError('Output must be outside package')
    if args.output.exists():raise FileExistsError(args.output)
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['formula']!='0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3':raise ValueError('Formula changed')
    verify_manifest(ROOT)
    if sha256(args.submission)!=cfg['historical_submission_sha256']:raise ValueError('Wrong reference submission')
    reference=read_submission(args.submission)
    measured,warm=images(args.pictures_100,100),images(args.pictures_5,5)
    if args.image_index:
        grouped,incomplete=sample_index(args.image_index,measured)
    else:
        from cxr.prediction_consistency import read_index
        full=read_index(ROOT/'test_image_index.csv')
        wanted={p.name for p in measured}
        rows=[r for r in full if r['image'] in wanted]
        if len(rows)!=100: raise ValueError('Timed images absent from frozen Test mapping')
        grouped={}; totals={}
        for row in rows:
            key=row['Subject_id'],row['Study_id']
            grouped.setdefault(key,[]).append(row['image'])
            totals[key]=int(row['study_image_count'])
        incomplete={k:(len(v),totals[k]) for k,v in grouped.items() if len(v)!=totals[k]}
    measured_hash={sha256(p) for p in measured}
    if measured_hash & {sha256(p) for p in warm}:raise ValueError('Warmup overlaps timed images')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('This package must be measured on NVIDIA T4')
    args.output.mkdir(parents=True)
    models=predict.load_models()
    for p in warm:
        predict.predict_image(p,models);torch.cuda.synchronize()
    times={p.name:[] for p in measured};rounds=[]
    with (args.output/'timings_300.csv').open('x',encoding='utf-8',newline='') as tf, \
         (args.output/'predictions_300.csv').open('x',encoding='utf-8',newline='') as pf, \
         (args.output/'study_predictions.csv').open('x',encoding='utf-8',newline='') as sf:
        tw,pw,sw=csv.writer(tf),csv.writer(pf),csv.writer(sf)
        tw.writerow(['round','image','elapsed_ms'])
        pw.writerow(['round','image',*[f'Prob_{i}' for i in range(1,11)]])
        sw.writerow(['round','Subject_id','Study_id',*[f'Prob_{i}' for i in range(1,11)]])
        for r in range(1,4):
            values={};pending={}
            for path in measured:
                torch.cuda.synchronize();start=time.perf_counter()
                v=probabilities(predict.predict_image(path,models))
                values[path.name]=v
                pw.writerow([r,path.name,*[format(float(x),'.17g') for x in v]]);pf.flush()
                for key,names in grouped.items():
                    if path.name in names:
                        pending.setdefault(key,{})[path.name]=v
                        if len(pending[key])==len(names) and key not in incomplete:
                            mean=np.mean([pending[key][n] for n in names],axis=0)
                            sw.writerow([r,*key,*[format(float(x),'.17g') for x in mean]]);sf.flush()
                        break
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                tw.writerow([r,path.name,format(elapsed,'.9f')]);tf.flush()
                times[path.name].append(elapsed)
            rounds.append(values)
    per_image=[statistics.mean(times[p.name]) for p in measured]
    checks=[];all_keys=set(grouped)
    for round_number,values in enumerate(rounds,1):
        complete={k:np.mean([values[n] for n in names],axis=0) for k,names in grouped.items() if k not in incomplete}
        if complete:
            report,_=compare_studies(complete,{k:reference[k] for k in complete})
        else:report={'status':'UNVERIFIED','studies':0}
        report['round']=round_number;checks.append(report)
    repeat_max=max(float(np.max(np.abs(rounds[0][name]-rounds[r][name]))) for name in rounds[0] for r in (1,2))
    consistency='UNVERIFIED_PARTIAL_STUDY' if incomplete else ('PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL')
    report=dict(status='T4_REFERENCE_PASS' if consistency=='PASS' and statistics.mean(per_image)<=100 else 'NOT_ACCEPTED',
                consistency_status=consistency,round_checks=checks,
                incomplete_studies=[{'Subject_id':k[0],'Study_id':k[1],'selected_images':v[0],'full_count':v[1]} for k,v in sorted(incomplete.items())],
                max_internal_repeat_diff=repeat_max,mean_ms=statistics.mean(per_image),
                p50_ms=float(np.percentile(per_image,50)),p95_ms=float(np.percentile(per_image,95)),
                records=300,batch=1,warmup=5,rounds=3,gpu=torch.cuda.get_device_name(),
                timer_scope='file read through image prediction write/flush, complete Study Mean/write when possible, and CUDA sync',
                official_sample_attested=bool(args.official_sample),
                official_compliance='PENDING_ORGANIZER_RETEST',
                note='Random/local sample PASS does not guarantee official fixed 100 images; partial Studies cannot be asserted consistent')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0 if report['status']=='T4_REFERENCE_PASS' else 2)


if __name__=='__main__':main()
