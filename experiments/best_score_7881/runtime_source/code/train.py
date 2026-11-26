"""Official-layout training entry; delegates to the archived implementation.
Requires original authorized training data and an explicit --config.
This file is not used during inference.
"""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
