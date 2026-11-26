"""Wait for N01, then independently validate raw/EMA, compare, benchmark and freeze."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cxr.utils.config import load_config


def status(path: Path, value: dict) -> None:
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
        raise ValueError("N01 runtime config mismatch")
    args.output_dir.mkdir(parents=True)
    state = args.output_dir / "status.json"
    status(state, {"state": "waiting_for_training", "run_dir": str(args.run_dir)})
    while True:
        training = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
        if training["state"] == "complete":
            break
        if training["state"] == "failed":
            status(state, {"state": "training_failed", "detail": training})
            raise RuntimeError("N01 training failed; no postprocessing started")
        stdout = args.run_dir / "train_stdout.log"
        if stdout.exists() and time.time() - stdout.stat().st_mtime > 7200:
            status(state, {"state": "training_stalled"})
            raise RuntimeError("N01 training log did not advance for two hours")
        time.sleep(30)
    commands = [
        ("raw_validation", ["scripts/analyze_validation.py", "--config", str(args.config),
                            "--run-dir", str(args.run_dir), "--checkpoint-kind", "raw"]),
        ("ema_validation", ["scripts/analyze_validation.py", "--config", str(args.config),
                            "--run-dir", str(args.run_dir), "--checkpoint-kind", "ema"]),
        ("fixed_comparison", ["scripts/compare_n01.py", "--config", str(args.config),
                              "--run-dir", str(args.run_dir), "--output-dir", str(args.run_dir / "comparison"),
                              "--bootstrap-replicates", "2000", "--seed", "42"]),
        ("inference_benchmark", ["scripts/benchmark_n01.py", "--config", str(args.config),
                                 "--run-dir", str(args.run_dir),
                                 "--output", str(args.run_dir / "inference_benchmark.json")]),
        ("freeze", ["scripts/freeze_n01.py", "--config", str(args.config), "--run-dir", str(args.run_dir)]),
    ]
    for label, command in commands:
        status(state, {"state": label})
        with (args.output_dir / f"{label}_stdout.log").open("x", encoding="utf-8") as out, \
                (args.output_dir / f"{label}_stderr.log").open("x", encoding="utf-8") as err:
            done = subprocess.run([sys.executable, *command], stdout=out, stderr=err, cwd=Path(__file__).resolve().parents[1])
        if done.returncode:
            status(state, {"state": "postprocessing_failed", "step": label,
                           "returncode": done.returncode})
            raise RuntimeError(f"N01 {label} failed with code {done.returncode}")
    status(state, {"state": "complete", "run_dir": str(args.run_dir)})


if __name__ == "__main__":
    main()
