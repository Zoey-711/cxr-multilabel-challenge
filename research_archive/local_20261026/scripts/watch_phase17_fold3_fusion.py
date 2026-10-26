"""Wait for the Fold 3 sequence, then compute its fixed CPU-only fusion once."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve() != (root / "configs/f17_densenet121_fold3_384.yaml").resolve():
        parser.error("Expected the frozen Fold 3 config")
    pipeline = root / "outputs/analysis/phase17_fold3_pipeline_20261021"
    state_path = pipeline / "status.json"
    result_path = root / "outputs/analysis/phase17_fold3_fixed_fusion_20261021/summary.json"
    while True:
        if result_path.exists():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["state"] == "failed":
            raise RuntimeError(f"Fold 3 pipeline failed: {state['error']}")
        if state["state"] == "complete":
            with (pipeline / "fusion_stdout.log").open("w", encoding="utf-8") as out, (
                    pipeline / "fusion_stderr.log").open("w", encoding="utf-8") as err:
                result = subprocess.run([sys.executable, "-u", "scripts/analyze_phase17_fold3_fusion.py",
                                         "--config", str(args.config)], cwd=root, stdout=out, stderr=err)
            if result.returncode:
                raise RuntimeError(f"Fold 3 fusion failed; see {pipeline / 'fusion_stderr.log'}")
            with (pipeline / "phase22_report_stdout.log").open("w", encoding="utf-8") as out, (
                    pipeline / "phase22_report_stderr.log").open("w", encoding="utf-8") as err:
                result = subprocess.run([sys.executable, "-u", "scripts/report_phase22_status.py",
                                         "--config", str(args.config)], cwd=root, stdout=out, stderr=err)
            if result.returncode:
                raise RuntimeError(f"Phase22 status report failed; see {pipeline / 'phase22_report_stderr.log'}")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
