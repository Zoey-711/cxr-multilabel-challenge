"""Phase49 cost runner. Uses the same load_backend/predict_image as predict.py."""
from __future__ import annotations

import argparse
from pathlib import Path

from predict import BACKENDS, benchmark, load_backend
from run_phase49 import samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    measured, warmup = samples()
    print(benchmark(load_backend(args.backend), measured, warmup, args.output_dir))
