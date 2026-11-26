from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source/src"))
from cxr.e08_s17_t4_cost import main
if __name__ == "__main__": main()
