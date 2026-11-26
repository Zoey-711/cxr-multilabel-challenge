"""Run the unchanged cost implementation beside the frozen runtime predict.py."""
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True
RUNTIME = Path(__file__).resolve().parent / "model"
sys.path.insert(0, str(RUNTIME))
if __name__ == "__main__":
    runpy.run_path(str(RUNTIME / "cost.py"), run_name="__main__")
