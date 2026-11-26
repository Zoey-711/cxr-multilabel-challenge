"""Official three-argument, stdout-only timing adapter; no result files."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_CACHE_DISABLE', '1')
import argparse
import json
import time
from pathlib import Path


def run_rounds(paths, groups, predict_one, synchronize, emit):
    import numpy as np
    lookup = {name: key for key, names in groups.items() for name in names}
    rounds, timings = [], {p.name: [] for p in paths}
    for number in range(1, 4):
        values, pending = {}, {}
        for path in paths:
            synchronize()
            start = time.perf_counter()
            v = np.asarray(predict_one(path), dtype=np.float64)
            if v.shape != (10,) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
                raise ValueError('Invalid prediction')
            values[path.name] = v
            key = lookup[path.name]
            pending.setdefault(key, {})[path.name] = v
            emit('[PREDICTION] ' + json.dumps({'round': number, 'image': path.name, 'probabilities': v.tolist()}))
            if len(pending[key]) == len(groups[key]):
                mean = np.mean([pending[key][n] for n in groups[key]], axis=0)
                emit('[STUDY] ' + json.dumps({'round': number, 'Subject_id': key[0], 'Study_id': key[1], 'probabilities': mean.tolist()}))
            synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            timings[path.name].append(elapsed)
            emit('[TIMING] ' + json.dumps({'round': number, 'image': path.name, 'elapsed_ms': elapsed}))
        rounds.append(values)
    return rounds, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('pictures_100', type=Path)
    parser.add_argument('pictures_5', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runtime = root / 'model'
    sys.path.insert(0, str(runtime / 'phase65_source/src'))
    sys.path.insert(0, str(root))
    from cxr.official_deployment_tools import verify_package, require_environment, require_engines
    from cxr.prediction_consistency import read_index, validate_index, read_submission, compare_rounds, sha256
    verify_package(runtime)
    cfg = json.loads((runtime/'config.json').read_text(encoding='utf-8'))
    if sha256(args.submission) != cfg['historical_submission_sha256']:
        raise ValueError('Reference must be the frozen historical submission')
    def images(folder, count):
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if len(paths) != count or len({p.name for p in paths}) != count:
            raise ValueError(f'Expected exactly {count} distinct image names: {folder}')
        return paths
    measured, warm = images(args.pictures_100,100), images(args.pictures_5,5)
    if {sha256(p) for p in measured} & {sha256(p) for p in warm}:
        raise ValueError('Warmup images overlap timed images')
    rows = read_index(runtime/'test_image_index.csv')
    mapping, all_groups = validate_index(rows)
    names = {p.name for p in measured}
    if not names <= set(mapping):
        raise ValueError('Timed images absent from frozen official Test metadata')
    selected = {(mapping[n]['Subject_id'], mapping[n]['Study_id']) for n in names}
    groups = {k: all_groups[k] for k in selected}
    if {n for group in groups.values() for n in group} != names:
        raise ValueError('UNVERIFIED_PARTIAL_STUDY: samples omit images needed for historical Study Mean; organizer clarification required')
    reference = read_submission(args.submission)
    environment = require_environment()
    require_engines(runtime)
    import torch
    import predict
    models = predict.load_models()
    for p in warm:
        predict.predict_image(p, models)
        torch.cuda.synchronize()
    print('[ENVIRONMENT] '+json.dumps(environment,ensure_ascii=False),flush=True)
    rounds, timings = run_rounds(measured, groups, lambda p:predict.predict_image(p,models),
                                torch.cuda.synchronize, lambda line:print(line,flush=True))
    check = compare_rounds(rows,rounds,reference,[p.name for p in measured])
    # Count a timed image once if any round fails its complete Study comparison.
    from cxr.prediction_consistency import aggregate_complete, compare_studies
    failed = set()
    for values in rounds:
        actual = aggregate_complete(rows, values, selected)
        _, details = compare_studies(actual, {k:reference[k] for k in selected})
        failed.update((d['Subject_id'],d['Study_id']) for d in details if d['failed'])
    inconsistent = sum((mapping[n]['Subject_id'],mapping[n]['Study_id']) in failed for n in names)
    mean_ms = sum(sum(v)/3 for v in timings.values())/100
    print('[CONSISTENCY] '+json.dumps(check,ensure_ascii=False),flush=True)
    limit_text = '未超过上限' if mean_ms <= 100 else '超过100ms上限'
    print(f'[CONCLUSION] 结论：100 张样本、batch=1 串行、连续3轮；单张端到端墙钟时间先按每张3轮取均值，再对100张取平均，最终平均耗时 {mean_ms:.6f} ms，{limit_text}；预测结果与 submission.csv 比对 100 张，不一致 {inconsistent} 张；推理精度 {predict.PRECISION}。',flush=True)
    return 0 if mean_ms<=100 and check['status']=='PASS' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        print('[ERROR] 若为驱动/CUDA初始化错误，请核对官方镜像和实际Python解释器；本程序不升级框架、不回退CPU、不输出虚构测速结论。',file=sys.stderr,flush=True)
        raise SystemExit(2)
