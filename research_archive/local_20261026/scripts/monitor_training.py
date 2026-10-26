"""Passively sample GPU board memory and existing B01 logs until training ends."""

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("--interval must be at least 10 seconds")
    run_dir = args.run_dir.resolve()
    if load_config(args.config) != load_config(run_dir / "config.yaml"):
        parser.error("Monitor config differs from the training config copy")
    path = run_dir / "gpu_monitor.csv"
    fields = ("local_time", "status", "last_completed_epoch", "active_epoch", "active_step",
              "gpu_board_used_mib", "gpu_board_free_mib", "gpu_utilization_percent")
    with path.open("a", encoding="utf-8-sig" if not path.exists() else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        while True:
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            log = (run_dir / "train_stdout.log").read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"epoch=(\d+) step=(\d+)/(\d+)", log)
            active_epoch, active_step = (int(matches[-1][0]), int(matches[-1][1])) if matches else (None, None)
            sample = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            memory = [part.strip() for part in sample.stdout.strip().splitlines()[0].split(",")] if sample.returncode == 0 and sample.stdout.strip() else ["", "", ""]
            writer.writerow({
                "local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": status.get("state"), "last_completed_epoch": status.get("epoch"),
                "active_epoch": active_epoch, "active_step": active_step,
                "gpu_board_used_mib": memory[0], "gpu_board_free_mib": memory[1],
                "gpu_utilization_percent": memory[2],
            })
            handle.flush()
            if status.get("state") in ("complete", "failed"):
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
