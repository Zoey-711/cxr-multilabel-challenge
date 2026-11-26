"""Wait for B02 completion, then independently validate and compare saved best checkpoints."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--b01-run-dir", required=True, type=Path)
    parser.add_argument("--b02-run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    b02 = args.b02_run_dir.resolve()
    if load_config(args.config) != load_config(b02 / "config.yaml"):
        parser.error("B02 config changed after launch")
    if args.poll_seconds < 10:
        parser.error("Poll interval must be at least 10 seconds")
    while True:
        status = json.loads((b02 / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"B02 failed: {status.get('error')}")
        time.sleep(args.poll_seconds)
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(scripts / "analyze_validation.py"), "--config", str(args.config.resolve()),
                    "--run-dir", str(b02)], check=True)
    subprocess.run([sys.executable, str(scripts / "compare_b01_b02.py"), "--config", str(args.config.resolve()),
                    "--b01-run-dir", str(args.b01_run_dir.resolve()), "--b02-run-dir", str(b02),
                    "--output-dir", str(b02 / "comparison"), "--bootstrap-replicates", "2000", "--seed", "42"],
                   check=True)
    print("B02 final validation and paired comparison complete", flush=True)


if __name__ == "__main__":
    main()
