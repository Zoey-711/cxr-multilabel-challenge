"""Run exactly one approved Phase17 backbone/fold after a passing cloud preflight."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cxr.utils.cloud_paths import output_root
from cxr.utils.config import load_config


MODELS = {"D17": ("f17_densenet121", "densenet121"),
          "C17": ("c17", "convnext_tiny"), "S17": ("s17", "swin_t")}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_file(root: Path, model: str, fold: int) -> Path:
    if model not in MODELS or fold not in range(1, 6):
        raise ValueError("Choose D17/C17/S17 and fold 1..5")
    return root / "configs" / f"{MODELS[model][0]}_fold{fold}_384.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = config_file(root, args.model, args.fold)
    if args.config.resolve() != expected.resolve():
        parser.error(f"Expected frozen cloud config: {expected}")
    config = load_config(expected)
    if config["model"]["name"] != MODELS[args.model][1] or \
            config["data"]["train_subset"]["fold"] != args.fold:
        parser.error("Model/fold mismatch")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("state") != "passed" or preflight.get("config_sha256") != sha256(expected) or \
            preflight.get("gpu_index_requested") != args.gpu:
        parser.error("Passing preflight for this exact config/GPU is required")
    run_id = f"{args.model.lower()}_f{args.fold}_cloud"
    run = output_root() / "experiments" / run_id
    if args.resume and not (run / "last_training.pt").is_file():
        parser.error("Cannot resume without last_training.pt")
    if not args.resume and run.exists():
        parser.error(f"Run directory already exists: {run}")
    run.mkdir(parents=True, exist_ok=args.resume)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    commands = [(["scripts/train.py", "--config", str(expected), "--run-id", run_id] +
                 (["--resume"] if args.resume else []), "train")]
    # train.py stores exact config and full resume state; postprocessing starts only after completion.
    for command, name in commands:
        with (run / f"{name}_stdout.log").open("a" if args.resume else "w", encoding="utf-8") as stdout, \
             (run / f"{name}_stderr.log").open("a" if args.resume else "w", encoding="utf-8") as stderr:
            result = subprocess.run([sys.executable, "-u", *command], cwd=root, env=env,
                                    stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {run}/{name}_stderr.log")
    for name, command in (("validation", ["scripts/analyze_validation.py", "--config", str(expected),
                                           "--run-dir", str(run)]),
                          ("oof", ["scripts/analyze_phase17_oof.py", "--config", str(expected),
                                    "--run-dir", str(run)]),
                          ("freeze", ["scripts/freeze_phase17_fold.py", "--config", str(expected),
                                       "--run-dir", str(run)])):
        with (run / f"{name}_stdout.log").open("w", encoding="utf-8") as stdout, \
             (run / f"{name}_stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run([sys.executable, "-u", *command], cwd=root, env=env,
                                    stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {run}/{name}_stderr.log")
    # The freezer hash-locks all result files and makes the run read-only.
    print(json.dumps({"state": "complete", "model": args.model, "fold": args.fold,
                      "run": str(run), "finished_utc": datetime.now(timezone.utc).isoformat()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
