"""Read docs/confirmation/README.md before using the Linux GPU queue."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr.regularization_confirmation.runner import main
if __name__=='__main__':main()
