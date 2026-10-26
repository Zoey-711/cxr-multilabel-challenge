"""Wait for R01, then independently validate, compare, benchmark, report, freeze."""

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
        raise ValueError("R01 runtime config mismatch")
    args.output_dir.mkdir(parents=True)
    state_path = args.output_dir / "status.json"
    write_status(state_path, {"state": "waiting_for_training", "run_dir": str(args.run_dir)})
    while True:
        training = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
        if training["state"] == "complete":
            break
        if training["state"] == "failed":
            write_status(state_path, {"state": "training_failed", "training_status": training})
            raise RuntimeError("R01 training failed; no postprocessing started")
        stdout = args.run_dir / "train_stdout.log"
        if stdout.exists() and time.time() - stdout.stat().st_mtime > 7200:
            write_status(state_path, {"state": "training_stalled", "run_dir": str(args.run_dir)})
            raise RuntimeError("R01 training log did not advance for two hours")
        time.sleep(30)
    commands = [
        ("independent_validation", ["scripts/analyze_validation.py", "--config", str(args.config),
                                    "--run-dir", str(args.run_dir)]),
        ("fixed_comparison", ["scripts/compare_r01.py", "--config", str(args.config),
                              "--run-dir", str(args.run_dir), "--output-dir", str(args.run_dir / "comparison"),
                              "--bootstrap-replicates", "2000", "--seed", "42"]),
        ("inference_benchmark", ["scripts/benchmark_r01.py", "--config", str(args.config),
                                 "--run-dir", str(args.run_dir), "--output", str(args.run_dir / "inference_benchmark.json")]),
        ("report", ["scripts/report_r01.py", "--config", str(args.config), "--run-dir", str(args.run_dir)]),
        ("freeze", ["scripts/freeze_r01.py", "--config", str(args.config), "--run-dir", str(args.run_dir)]),
    ]
    for label, command in commands:
        write_status(state_path, {"state": label, "run_dir": str(args.run_dir)})
        with (args.output_dir / f"{label}_stdout.log").open("w", encoding="utf-8") as out, \
                (args.output_dir / f"{label}_stderr.log").open("w", encoding="utf-8") as err:
            outcome = subprocess.run([sys.executable, *command], stdout=out, stderr=err, check=False)
        if outcome.returncode:
            write_status(state_path, {"state": "postprocessing_failed", "step": label,
                                      "returncode": outcome.returncode, "run_dir": str(args.run_dir)})
            raise RuntimeError(f"R01 {label} failed with code {outcome.returncode}")
    write_status(state_path, {"state": "complete", "run_dir": str(args.run_dir)})


if __name__ == "__main__":
    main()
