"""Independent consistency-audited six-model entry; does not replace frozen code."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from cxr.e08_s17_consistent_entry import (
    PRECISION, load_models, predict_image, aggregate_studies, main,
)

if __name__=='__main__':main()
