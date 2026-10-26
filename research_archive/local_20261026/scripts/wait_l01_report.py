"""Write the final L01 summary after the validation pipeline completes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("Output exists or L01 config mismatch")
    while True:
        state = json.loads((args.pipeline_dir / "status.json").read_text(encoding="utf-8"))
        if state["state"] == "complete":
            break
        if state["state"] in ("training_failed", "postprocessing_failed"):
            raise RuntimeError(f"L01 pipeline failed: {state}")
        time.sleep(30)
    subprocess.run([sys.executable, "scripts/report_l01.py", "--config", str(args.config),
                    "--run-dir", str(args.run_dir), "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
