"""Benchmark the exact Phase44 predict.py model loader and image entry."""
import csv
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from predict import load_models,predict_image,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def benchmark(precision='fp16'):
    folder=ROOT/'outputs/analysis/phase29_deployment/benchmark_images'
    warm=sorted((folder/'warmup_5').glob('*.jpg'))
    measured=sorted((folder/'pictures_100').glob('*.jpg'))
    if len(warm)!=5 or len(measured)!=100:raise ValueError('Fixed Val timing sample incomplete')
    models=load_models(precision=precision)
    for path in warm:predict_image(path,models)
    times=[]
    output=HERE/f'onnx_{precision}_cost_predictions.csv'
    if output.exists():raise FileExistsError(output)
    with output.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['round','image']+[f'Prob_{i}' for i in range(1,11)])
        for round_id in range(3):
            for path in measured:
                start=time.perf_counter()
                prob=predict_image(path,models)
                writer.writerow([round_id,path.name,*[format(float(x),'.17g') for x in prob]])
                f.flush()
                times.append((time.perf_counter()-start)*1000)
    report=dict(device='NVIDIA RTX 4060 Laptop',provider='ONNX Runtime CUDA',precision=precision.upper(),
                batch=1,warmup_images=5,measured_images=100,rounds=3,
                scope='file open/decode/preprocess, H2D, five model forwards, fusion and CSV write; model load excluded',
                mean_ms=statistics.mean(times),p50_ms=float(np.percentile(times,50)),
                p95_ms=float(np.percentile(times,95)),max_ms=max(times),
                official_status='T4_UNVERIFIED',cost_output_sha256=sha(output))
    (HERE/f'onnx_{precision}_cost_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=('fp16','fp32'),default='fp16')
    benchmark(parser.parse_args().precision)
