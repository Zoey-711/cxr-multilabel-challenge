"""Benchmark the same predict.py entry; fixed original Val sample configuration."""
import argparse
import json
import sys
from pathlib import Path
import predict
from cxr.benchmark_e08_s17 import load_module
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    samples=Path(cfg['val_samples']);package=Path(cfg['e08_package'])
    core=load_module(package.parents[1]/'benchmark_core.py','e08_s17_optimized_cost_core')
    entry=SimpleNamespace(PRECISION=predict.PRECISION,
                          load_models=lambda:predict.load_models(args.config),predict_image=predict.predict_image)
    sys.argv=['cost','--pictures-100',str(samples/'pictures_100'),'--pictures-5',str(samples/'pictures_5'),
              '--output',str(args.output)]
    core.run(entry,'E08_S17_CUDA_GRAPH_FP16')


if __name__=='__main__':main()
