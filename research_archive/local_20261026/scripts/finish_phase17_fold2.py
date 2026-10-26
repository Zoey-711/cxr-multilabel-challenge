"""Wait for the approved Fold 2 pilot and calculate its fixed fusion once."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve() != (root / "configs/f17_densenet121_fold2_384.yaml").resolve():
        parser.error("Expected the approved Fold 2 configuration")
    pipeline = root / "outputs/analysis/phase17_fold2_pipeline_20261020/status.json"
    output = root / "outputs/analysis/phase17_fold2_fixed_fusion_20261020"
    while True:
        state = json.loads(pipeline.read_text(encoding="utf-8"))
        if state["state"] == "complete":
            break
        if state["state"] == "failed":
            raise RuntimeError(f"Fold 2 stopped: {state.get('error')}")
        time.sleep(30)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fusion output: {output}")
    subprocess.run([sys.executable, "-u", "scripts/analyze_phase17_fold2_fusion.py",
                    "--config", str(args.config), "--output-dir", str(output)],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()
