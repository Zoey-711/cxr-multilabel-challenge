"""Run the pending E17-F1 Test inference only after local Fold 2 fully completes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    status_path = root / "outputs/analysis/phase17_fold2_pipeline_20261020/status.json"
    target = root / "outputs/submissions/E17_F1_20261020"
    while True:
        if (target / "manifest.json").exists():
            print("E17-F1 candidate already finalized; no repeat inference", flush=True)
            return
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if state["state"] == "complete":
            break
        if state["state"] == "failed":
            raise RuntimeError(f"Fold 2 failed; GPU inference remains pending: {state.get('error')}")
        time.sleep(30)
    subprocess.run([sys.executable, "-u", "scripts/generate_e17_f1_candidate.py",
                    "--config", str(args.config), "--stage", "finish",
                    "--output-dir", str(target)], cwd=root, check=True)


if __name__ == "__main__":
    main()
