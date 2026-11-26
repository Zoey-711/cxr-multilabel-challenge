"""Official Study-based timing. Three positional arguments; stdout only."""
import os, sys
sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_CACHE_DISABLE', '1')
import argparse, json, time
from pathlib import Path

def select_studies(folder, expected, mapping, all_groups):
    paths = sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
    by_name = {p.name:p for p in paths}
    if len(by_name) != len(paths) or not paths:
        raise ValueError('Empty input or duplicate image filenames')
    if not set(by_name) <= set(mapping):
        raise ValueError('Images absent from packaged Test metadata')
    keys = {(mapping[n]['Subject_id'],mapping[n]['Study_id']) for n in by_name}
    if len(keys) != expected:
        raise ValueError(f'Expected {expected} Studies, found {len(keys)}')
    groups = {}
    for key in sorted(keys):
        missing = set(all_groups[key])-set(by_name)
        if missing:
            raise ValueError(f'Incomplete Study {key}; missing images: {sorted(missing)}')
        groups[key] = [by_name[n] for n in all_groups[key]]
    return groups

def predict_study(paths, predict_one):
    import numpy as np
    values = []
    for p in paths:
        v = np.asarray(predict_one(p), dtype=np.float64)
        if v.shape != (10,) or not np.isfinite(v).all() or np.any((v<0)|(v>1)):
            raise ValueError('Invalid image probabilities')
        values.append(v)
    return np.mean(values,axis=0)

def run_rounds(groups, predict_one, synchronize, emit):
    rounds, timings = [], {key:[] for key in groups}
    for number in range(1,4):
        values = {}
        for key, paths in groups.items():
            synchronize()
            start = time.perf_counter()
            v = predict_study(paths,predict_one)
            emit('[PREDICTION] '+json.dumps({'round':number,'Subject_id':key[0],'Study_id':key[1],'probabilities':v.tolist()}))
            synchronize()
            elapsed = (time.perf_counter()-start)*1000
            values[key]=v
            timings[key].append(elapsed)
            emit('[TIMING] '+json.dumps({'round':number,'Study_id':key[1],'images':len(paths),'elapsed_ms':elapsed}))
        rounds.append(values)
    return rounds,timings

def evaluate_consistency(rounds,reference):
    from cxr.prediction_consistency import compare_studies
    failed,reports=set(),[]
    for number,values in enumerate(rounds,1):
        report,details=compare_studies(values,{k:reference[k] for k in values})
        failed.update((d['Subject_id'],d['Study_id']) for d in details if d['failed'])
        reports.append(dict(report,round=number))
    return {'status':'FAIL' if failed else 'PASS','inconsistent_studies':len(failed),'round_checks':reports}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission',type=Path)
    parser.add_argument('pictures_100',type=Path)
    parser.add_argument('pictures_5',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    runtime=root/'model'
    sys.path.insert(0,str(runtime/'phase65_source/src'))
    sys.path.insert(0,str(root))
    from cxr.official_deployment_tools import verify_package,require_environment,require_engines
    from cxr.prediction_consistency import read_index,validate_index,read_submission
    verify_package(runtime)
    mapping,all_groups=validate_index(read_index(runtime/'test_image_index.csv'))
    measured=select_studies(args.pictures_100,100,mapping,all_groups)
    warm=select_studies(args.pictures_5,5,mapping,all_groups)
    if set(measured)&set(warm):
        raise ValueError('Warmup Studies overlap timed Studies')
    reference=read_submission(args.submission)
    if not set(measured)<=set(reference):
        raise ValueError('Timed Study missing from submission')
    environment=require_environment()
    require_engines(runtime)
    import torch, predict
    models=predict.load_models()
    predict_one=lambda p:predict.predict_image(p,models)
    for paths in warm.values():
        predict_study(paths,predict_one)
        torch.cuda.synchronize()
    print('[ENVIRONMENT] '+json.dumps(environment,ensure_ascii=False),flush=True)
    rounds,timings=run_rounds(measured,predict_one,torch.cuda.synchronize,lambda line:print(line,flush=True))
    check=evaluate_consistency(rounds,reference)
    mean_ms=sum(sum(v)/3 for v in timings.values())/100
    print('[CONSISTENCY] '+json.dumps(check,ensure_ascii=False),flush=True)
    print(f"[CONCLUSION] 结论：100 个样本、batch=1 串行、连续3轮；单样本端到端墙钟时间先按每个样本3轮取均值，再对100个取平均，最终平均耗时{mean_ms:.6f}ms；预测结果与 submission.csv 比对 100 个样本，不一致{check['inconsistent_studies']}个样本；推理精度{predict.PRECISION}。",flush=True)
    return 0 if mean_ms<=100 and check['status']=='PASS' else 2

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
