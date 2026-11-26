"""Local paired confirmation; all settings are frozen by --config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cxr.mixup_confirmation import main

if __name__ == '__main__':
    main()
