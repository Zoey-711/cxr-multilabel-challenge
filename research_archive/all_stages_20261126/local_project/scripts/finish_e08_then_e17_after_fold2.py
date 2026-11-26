"""After Fold 2, finish E08-EMA first, then the previously prepared E17-F1."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_finish(root: Path, script: str, config: str, target: Path) -> None:
    if (target / "manifest.json").exists():
        print(f"ALREADY_COMPLETE {target}", flush=True)
        return
    if not (target / "preparation_manifest.json").exists():
        raise FileNotFoundError(f"Candidate was not prepared: {target}")
    with (target / "priority_finish_stdout.log").open("x", encoding="utf-8") as out, \
            (target / "priority_finish_stderr.log").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", script, "--config", config,
                                 "--stage", "finish", "--output-dir", str(target)],
                                cwd=root, stdout=out, stderr=err, check=False)
    if result.returncode:
        raise RuntimeError(f"Candidate finish failed ({result.returncode}); see {target / 'priority_finish_stderr.log'}")
    if not (target / "manifest.json").exists():
        raise RuntimeError(f"Candidate did not produce a manifest: {target}")
    print(f"FINISHED {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = root / "outputs/experiments/n01_efficientnet_v2_s_384_amp_retry_20261018/config.yaml"
    if args.config.resolve() != expected.resolve():
        parser.error("Expected frozen N01 runtime config")
    status_path = root / "outputs/analysis/phase17_fold2_pipeline_20261020/status.json"
    while True:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"Fold 2 failed; deferring both Test inference jobs: {status.get('error')}")
        time.sleep(30)
    run_finish(root, "scripts/generate_e08_ema_candidate.py", str(args.config),
               root / "outputs/submissions/E08_EMA_20261020")
    run_finish(root, "scripts/generate_e17_f1_candidate.py",
               "configs/f17_densenet121_fold1_384.yaml",
               root / "outputs/submissions/E17_F1_20261020")


if __name__ == "__main__":
    main()
