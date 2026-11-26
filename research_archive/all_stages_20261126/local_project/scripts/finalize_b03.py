"""Wait for B03 completion, then independently validate and compare saved best checkpoints."""

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
    parser.add_argument("--b03-run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    b03 = args.b03_run_dir.resolve()
    if load_config(args.config) != load_config(b03 / "config.yaml"):
        parser.error("B03 config changed after launch")
    if args.poll_seconds < 10:
        parser.error("Poll interval must be at least 10 seconds")
    while True:
        status = json.loads((b03 / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"B03 failed: {status.get('error')}")
        time.sleep(args.poll_seconds)
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(scripts / "analyze_validation.py"), "--config", str(args.config.resolve()),
                    "--run-dir", str(b03)], check=True)
    subprocess.run([sys.executable, str(scripts / "compare_b01_b03.py"), "--config", str(args.config.resolve()),
                    "--b01-run-dir", str(args.b01_run_dir.resolve()), "--b03-run-dir", str(b03),
                    "--output-dir", str(b03 / "comparison"), "--bootstrap-replicates", "2000", "--seed", "42"],
                   check=True)
    print("B03 final validation and paired comparison complete", flush=True)


if __name__ == "__main__":
    main()
