"""Wait for S01, then independently validate, compare, and freeze its run."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.monitor_dir.exists():
        raise FileExistsError(args.monitor_dir)
    args.monitor_dir.mkdir(parents=True)
    state_path = args.monitor_dir / "status.json"

    def record(state, **details):
        state_path.write_text(json.dumps({"state": state,
                                          "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                                          **details}, ensure_ascii=False, indent=2), encoding="utf-8")

    record("waiting_for_training")
    try:
        while True:
            train_status = args.run_dir / "status.json"
            if train_status.is_file():
                current = json.loads(train_status.read_text(encoding="utf-8"))
                if current.get("state") == "failed":
                    raise RuntimeError(f"S01 training failed: {current.get('error')}")
                if current.get("state") == "complete":
                    break
            time.sleep(20)
        # The training process writes status last; allow stdout redirection to close.
        time.sleep(15)
        jobs = [
            ("independent_validation", ["scripts/analyze_validation.py", "--config", str(args.config),
                                        "--run-dir", str(args.run_dir)]),
            ("comparison", ["scripts/compare_s01.py", "--config", str(args.config),
                            "--run-dir", str(args.run_dir), "--output-dir", str(args.run_dir / "comparison"),
                            "--bootstrap-replicates", "2000", "--seed", "42"]),
            ("freeze", ["scripts/freeze_s01.py", "--config", str(args.config),
                        "--run-dir", str(args.run_dir)]),
        ]
        for name, arguments in jobs:
            record("running", stage=name)
            with (args.monitor_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, \
                    (args.monitor_dir / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                subprocess.run([sys.executable, *arguments], check=True, stdout=stdout, stderr=stderr)
        record("complete", run_dir=str(args.run_dir))
    except Exception as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
