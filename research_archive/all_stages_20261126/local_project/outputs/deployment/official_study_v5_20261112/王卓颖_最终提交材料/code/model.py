"""Architecture factory for reviewing/reproducing the archived training models.
Deployment must use predict.load_models(), including the frozen T4 engines.
Importing this module does not download weights or instantiate models.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cxr.models.factory import create_model, imagenet_weight_info
__all__ = ["create_model", "imagenet_weight_info"]
