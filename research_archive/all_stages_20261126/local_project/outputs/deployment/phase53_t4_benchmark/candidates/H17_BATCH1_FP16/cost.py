"""Official-style cost check for H17_BATCH1_FP16; uses predict.py unchanged."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import predict
from benchmark_core import run
if __name__ == "__main__":
    run(predict, 'H17_BATCH1_FP16')
