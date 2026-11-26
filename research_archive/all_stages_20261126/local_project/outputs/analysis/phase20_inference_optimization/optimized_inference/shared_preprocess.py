"""Validated Phase 20 shared-preprocessing Val inference entrypoint.

This delegates to the exact implementation whose 859-Study comparison is saved
in ../prediction_equivalence.json. It does not change the production predictor.
"""
from pathlib import Path
import argparse
import os
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_phase20 import benchmark, load_sources, validate  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "benchmark"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    archive = Path(os.environ["CXR_ARCHIVE"])
    if not archive.is_file():
        raise FileNotFoundError(archive)
    manifest, sources = load_sources()
    if args.mode == "validate":
        validate(archive, manifest, sources)
    else:
        benchmark(archive, sources)
