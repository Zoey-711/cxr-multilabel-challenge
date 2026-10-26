"""Run only the approved local Fold 4 models in sequence, with epoch resume."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cxr.utils.config import load_config


TASKS = (
    ("D17-F4", "configs/f17_densenet121_fold4_384.yaml", "d17_f4_20261022"),
    ("C17-F4", "configs/c17_fold4_384.yaml", "c17_f4_20261022"),
    ("S17-F4", "configs/s17_fold4_384.yaml", "s17_f4_20261022"),
)


def write_status(path: Path, **values) -> None:
    path.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), **values},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(root: Path, command: list[str], stdout: Path, stderr: Path) -> None:
    if stdout.exists() or stderr.exists():
        raise FileExistsError(f"Log already exists: {stdout} or {stderr}")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", *command], cwd=root,
                                stdout=out, stderr=err, check=False)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(command)}; see {stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Must be the frozen D17 Fold 4 configuration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve() != (root / TASKS[0][1]).resolve():
        parser.error("Only the approved D17 Fold 4 configuration is accepted")
    fold_hashes = load_config(args.config)["data"]["train_subset"]["sha256"]
    for model_id, path, _ in TASKS:
        subset = load_config(root / path)["data"]["train_subset"]
        if subset["fold"] != 4 or subset["sha256"] != fold_hashes:
            raise ValueError(f"{model_id} has a different frozen fold")
    output = root / "outputs/analysis/phase17_fold4_pipeline_20261022"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    completed = []
    model_id = "preflight"
    try:
        for model_id, config_path, run_id in TASKS:
            run = root / "outputs/experiments" / run_id
            run.mkdir(parents=True, exist_ok=True)
            config = load_config(root / config_path)
            if (run / "config.yaml").exists() and load_config(run / "config.yaml") != config:
                raise ValueError(f"{model_id} runtime config differs; refusing resume")
            if (run / "frozen_manifest.json").exists():
                completed.append(model_id)
                continue
            status = json.loads((run / "status.json").read_text(encoding="utf-8")) if (run / "status.json").exists() else {}
            if status.get("state") != "complete":
                resume = (run / "last_training.pt").exists()
                if status and not resume:
                    raise RuntimeError(f"{model_id} has no valid epoch checkpoint; cannot resume safely")
                tag = "train_resume" if resume else "train"
                if resume:
                    tag += f"_after_epoch_{status.get('epoch', 'unknown')}"
                write_status(status_path, state="running", step=f"{model_id}:training", completed=completed)
                run_command(root, ["scripts/train.py", "--config", config_path, "--run-id", run_id,
                                   *(["--resume"] if resume else [])],
                            run / f"{tag}_stdout.log", run / f"{tag}_stderr.log")
            if not (run / "validation_analysis/analysis.json").exists():
                write_status(status_path, state="running", step=f"{model_id}:validation", completed=completed)
                run_command(root, ["scripts/analyze_validation.py", "--config", config_path,
                                   "--run-dir", str(run)], run / "val_stdout.log", run / "val_stderr.log")
            if not (run / "oof_analysis/analysis.json").exists():
                write_status(status_path, state="running", step=f"{model_id}:oof", completed=completed)
                run_command(root, ["scripts/analyze_phase17_oof.py", "--config", config_path,
                                   "--run-dir", str(run)], run / "oof_stdout.log", run / "oof_stderr.log")
            write_status(status_path, state="running", step=f"{model_id}:freeze", completed=completed)
            run_command(root, ["scripts/freeze_phase17_fold.py", "--config", config_path,
                               "--run-dir", str(run)],
                        output / f"{model_id}_freeze_stdout.log", output / f"{model_id}_freeze_stderr.log")
            completed.append(model_id)
        fusion_dir = root / "outputs/analysis/phase17_fold4_fixed_fusion_20261022"
        if not (fusion_dir / "summary.json").exists():
            write_status(status_path, state="running", step="fold4_fixed_fusion", completed=completed)
            run_command(root, ["scripts/analyze_phase17_fold4_fusion.py", "--config", TASKS[0][1]],
                        output / "fold4_fusion_stdout.log", output / "fold4_fusion_stderr.log")
        write_status(status_path, state="complete", step="fold4_fixed_fusion_complete", completed=completed)
    except Exception as exc:
        write_status(status_path, state="failed", step=model_id, completed=completed,
                     error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
