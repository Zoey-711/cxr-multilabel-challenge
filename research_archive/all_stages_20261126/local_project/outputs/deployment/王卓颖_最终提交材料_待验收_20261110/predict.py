from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source"))
sys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source/src"))
from phase65_source.predict import load_models,predict_image,PRECISION
