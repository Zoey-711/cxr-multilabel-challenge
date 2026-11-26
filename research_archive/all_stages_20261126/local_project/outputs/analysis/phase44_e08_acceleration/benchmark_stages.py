"""Single-round explanatory stage timing on the fixed 100 Val images."""
import csv
import argparse
from pathlib import Path
import statistics
import time
import numpy as np
from predict import load_models,predict_image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=('fp16','fp32'),default='fp16')
    precision=parser.parse_args().precision
    sample=ROOT/'outputs/analysis/phase29_deployment/benchmark_images'
    warm=sorted((sample/'warmup_5').glob('*.jpg'))
    images=sorted((sample/'pictures_100').glob('*.jpg'))
    if len(warm)!=5 or len(images)!=100:raise ValueError('Fixed sample incomplete')
    models=load_models(precision=precision)
    for p in warm:predict_image(p,models)
    observations={}
    for path in images:
        stage={};start=time.perf_counter()
        predict_image(path,models,stage)
        stage['end_to_end_without_csv']=(time.perf_counter()-start)*1000
        for name,value in stage.items():observations.setdefault(name,[]).append(value)
    output=HERE/f'latency_breakdown_{precision}.csv'
    with output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','mean_ms','p50_ms','p95_ms','scope'])
        for name,values in observations.items():
            writer.writerow([name,statistics.mean(values),np.percentile(values,50),np.percentile(values,95),
                             f'RTX4060 ONNX Runtime {precision.upper()}; fixed 100 Val images; one instrumented round; not official T4'])
    print(output)

if __name__=='__main__':main()
