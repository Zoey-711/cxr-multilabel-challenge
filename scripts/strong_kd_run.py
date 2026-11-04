"""CPU teacher audit by default; explicitly authorized remote student training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr.strong_kd.runner import main
if __name__=='__main__':main()
