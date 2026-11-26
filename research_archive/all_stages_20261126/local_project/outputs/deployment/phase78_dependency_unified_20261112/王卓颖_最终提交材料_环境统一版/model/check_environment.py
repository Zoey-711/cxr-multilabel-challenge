from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase65_source/src"))
from cxr.official_deployment_tools import main
if __name__ == "__main__": main("check", ROOT)
