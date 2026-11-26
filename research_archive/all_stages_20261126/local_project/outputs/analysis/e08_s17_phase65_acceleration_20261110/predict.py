"""Independent local six-model CUDA Graph entry. Initialization is not timed."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from cxr.e08_s17_acceleration import Entry, load_models as load_frozen_models, PRECISION


def load_models(config_path=Path(__file__).with_name('config.json')):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    return Entry(load_frozen_models(config),'cuda_graph')


def predict_image(path,state):
    return state.predict_image(Path(path))
