"""Wait for M02 to finish, then independently validate, compare, and freeze it."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config


def invoke(script: Path, arguments: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, str(script), *arguments], stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--m02-run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    m01, m02 = args.m01_run_dir.resolve(), args.m02_run_dir.resolve()
    if load_config(args.config) != load_config(m02 / "config.yaml"):
        parser.error("M02 config differs from its runtime copy")
    if args.poll_seconds < 10:
        parser.error("poll-seconds must be at least 10")
    if json.loads((m01 / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        parser.error("Frozen M01 must be complete")
    while True:
        status = json.loads((m02 / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"M02 training failed: {status.get('error')}")
        time.sleep(args.poll_seconds)
    # The passive GPU monitor writes a final complete-state row before stopping.
    monitor = m02 / "gpu_monitor.csv"
    for _ in range(12):
        if monitor.exists() and ",complete," in monitor.read_text(encoding="utf-8-sig"):
            break
        time.sleep(5)
    else:
        raise RuntimeError("GPU monitor did not save its final complete-state sample")
    scripts = Path(__file__).resolve().parent
    invoke(scripts / "analyze_validation.py",
           ["--config", str(args.config), "--run-dir", str(m02)],
           m02 / "validation_analysis_stdout.log")
    invoke(scripts / "compare_m01_m02.py",
           ["--config", str(args.config), "--m01-run-dir", str(m01),
            "--m02-run-dir", str(m02), "--output-dir", str(m02 / "comparison"),
            "--bootstrap-replicates", "2000", "--seed", "42"],
           m02 / "comparison_stdout.log")
    invoke(scripts / "freeze_m02.py",
           ["--config", str(args.config), "--run-dir", str(m02)],
           m02 / "freeze_stdout.log")
    print((m02 / "freeze_stdout.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
