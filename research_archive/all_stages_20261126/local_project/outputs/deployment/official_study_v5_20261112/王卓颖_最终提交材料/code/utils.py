"""Public utility facade; preserves the original checkpoint implementation."""
from pathlib import Path
import hashlib
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cxr.utils.checkpoint import load_model_state, save_model_state, check_model_file_size

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def load_config(path):
    # PyYAML is required only for reproducing training, not deployment.
    from cxr.utils.config import load_config as original_load_config
    return original_load_config(path)

__all__ = ["sha256", "load_config", "load_model_state", "save_model_state", "check_model_file_size"]
