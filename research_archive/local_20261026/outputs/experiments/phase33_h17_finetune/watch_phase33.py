"""CPU-only watcher: finalize completed Pilot, preserve failures for review."""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "P33_H17_HEAD_01"


def main():
    while True:
        status = RUN / "status.json"
        if status.exists():
            state = json.loads(status.read_text(encoding="utf-8"))
            if state.get("state") == "TRAINING_COMPLETE":
                if not (RUN / "final_decision.md").exists():
                    result = subprocess.run([sys.executable, str(HERE / "finalize_phase33.py")],
                                            capture_output=True, text=True)
                    if result.returncode:
                        (HERE / "finalize_error.log").write_text(result.stderr, encoding="utf-8")
                    else:
                        (HERE / "finalize_stdout.log").write_text(result.stdout, encoding="utf-8")
                return
            if state.get("state") in ("FAILED", "failed"):
                (HERE / "watch_status.json").write_text(json.dumps({"state":"TRAINING_FAILED", "details":state},
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
                return
        time.sleep(30)


if __name__ == "__main__":
    main()
