"""Wait for the existing M02 finalizer, then compare frozen M02 with E01."""

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
    parser.add_argument("--m02-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("Poll interval must be at least ten seconds")
    if load_config(args.config) != load_config(args.m02_run_dir / "config.yaml"):
        parser.error("M02 runtime config differs")
    while True:
        status = json.loads((args.m02_run_dir / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "failed":
            raise RuntimeError(f"M02 training failed: {status.get('error')}")
        if status["state"] == "complete" and (args.m02_run_dir / "frozen_manifest.json").is_file():
            break
        time.sleep(args.poll_seconds)
    command = [sys.executable, str(Path(__file__).with_name("compare_e01_m02.py")),
               "--config", str(args.config), "--m02-run-dir", str(args.m02_run_dir),
               "--output-dir", str(args.output_dir), "--bootstrap-replicates", "2000"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
