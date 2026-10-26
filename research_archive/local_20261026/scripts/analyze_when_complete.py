"""Wait for an existing run to finish, then launch validation-only analysis."""

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
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    run_dir = args.run_dir.resolve()
    if load_config(args.config) != load_config(run_dir / "config.yaml"):
        parser.error("Watcher config differs from the training config copy")
    while True:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            break
        if status.get("state") == "failed":
            raise RuntimeError(f"Training failed; analysis not started: {status.get('error')}")
        time.sleep(args.poll_seconds)
    command = [sys.executable, str(Path(__file__).with_name("analyze_validation.py")),
               "--config", str(args.config.resolve()), "--run-dir", str(run_dir)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
