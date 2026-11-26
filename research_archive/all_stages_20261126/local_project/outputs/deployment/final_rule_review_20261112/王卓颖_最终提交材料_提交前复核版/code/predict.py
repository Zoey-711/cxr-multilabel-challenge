"""Delegate to the unchanged T4 benchmark runtime."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "model"
sys.path.insert(0, str(RUNTIME))
_spec = importlib.util.spec_from_file_location("_frozen_t4_predict", RUNTIME / "predict.py")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)
load_models = _entry.load_models
predict_image = _entry.predict_image
PRECISION = _entry.PRECISION
