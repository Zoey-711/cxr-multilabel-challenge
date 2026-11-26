"""Wait for the authorized L01 run, then validate, compare, report, and freeze it."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config


def write_status(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if load_config(args.config) != load_config(args.run_dir / "config.yaml"):
        raise ValueError("L01 config mismatch")
    args.output_dir.mkdir(parents=True)
    state_file = args.output_dir / "status.json"
    write_status(state_file, {"state": "waiting_for_training", "run_dir": str(args.run_dir)})
    while True:
        state = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
        if state["state"] == "complete":
            break
        if state["state"] == "failed":
            write_status(state_file, {"state": "training_failed", "training_status": state})
            raise RuntimeError("L01 training failed; no postprocessing started")
        time.sleep(30)
    commands = [
        ("independent_validation", ["scripts/analyze_validation.py", "--config", str(args.config),
                                    "--run-dir", str(args.run_dir)]),
        ("fixed_comparison", ["scripts/compare_l01.py", "--config", str(args.config),
                              "--run-dir", str(args.run_dir), "--output-dir", str(args.run_dir / "comparison"),
                              "--bootstrap-replicates", "2000", "--seed", "42"]),
        ("report", ["scripts/report_l01.py", "--config", str(args.config), "--run-dir", str(args.run_dir)]),
        ("freeze", ["scripts/freeze_l01.py", "--config", str(args.config), "--run-dir", str(args.run_dir)]),
    ]
    for label, command in commands:
        write_status(state_file, {"state": label, "run_dir": str(args.run_dir)})
        with (args.output_dir / f"{label}_stdout.log").open("w", encoding="utf-8") as out, \
                (args.output_dir / f"{label}_stderr.log").open("w", encoding="utf-8") as err:
            outcome = subprocess.run([sys.executable, *command], stdout=out, stderr=err, check=False)
        if outcome.returncode:
            write_status(state_file, {"state": "postprocessing_failed", "failed_step": label,
                                      "returncode": outcome.returncode, "run_dir": str(args.run_dir)})
            raise RuntimeError(f"L01 {label} failed: {outcome.returncode}")
    write_status(state_file, {"state": "complete", "run_dir": str(args.run_dir)})


if __name__ == "__main__":
    main()
