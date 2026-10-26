"""Run only the approved D17/C17/S17 Fold 1 pilot, sequentially."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cxr.data.patient_folds import sha256
from cxr.utils.config import load_config


PILOT = (
    ("D17-F1", "configs/f17_densenet121_fold1_384.yaml", "d17_f1_20261019", "d17_f1_preflight_20261019", "d17_checkpoint.json"),
    ("C17-F1", "configs/c17_fold1_384.yaml", "c17_f1_20261019", "c17_f1_preflight_20261019", "c17_checkpoint.json"),
    ("S17-F1", "configs/s17_fold1_384.yaml", "s17_f1_20261019", "s17_f1_preflight_20261019", "s17_checkpoint.json"),
)


def write_status(path: Path, **values) -> None:
    path.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), **values},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(root: Path, args: list[str], stdout: Path, stderr: Path) -> None:
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", *args], cwd=root, stdout=out, stderr=err,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}; see {stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="D17 Fold 1 config for protocol assertion")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config.resolve() != (root / PILOT[0][1]).resolve():
        parser.error("Pilot must use the predeclared D17 Fold 1 config")
    first = load_config(args.config)
    fold_hashes = first["data"]["train_subset"]["sha256"]
    preflight_dir = root / "outputs/analysis/phase17_preflight"
    for name, config_path, run_id, preflight_id, checkpoint_report in PILOT:
        config_file = root / config_path
        config = load_config(config_file)
        if config["data"]["train_subset"]["fold"] != 1 or \
                config["data"]["train_subset"]["sha256"] != fold_hashes:
            raise ValueError(f"{name} does not use identical Fold 1 indices")
        preflight = json.loads((root / "outputs/experiments" / preflight_id / "status.json").read_text(encoding="utf-8"))
        checkpoint = json.loads((preflight_dir / checkpoint_report).read_text(encoding="utf-8"))
        if preflight["state"] != "preflight_passed" or preflight["amp_skipped_steps"] != 0 or \
                preflight["optimizer_steps"] != 2 or not checkpoint["logits_identical"] or \
                checkpoint["config_sha256"] != sha256(config_file):
            raise ValueError(f"{name} GPU/checkpoint preflight did not pass")
        if (root / "outputs/experiments" / run_id).exists():
            raise FileExistsError(f"Formal experiment directory already exists: {run_id}")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Pilot pipeline directory already exists: {output}")
    output.mkdir(parents=True)
    status_path = output / "status.json"
    write_status(status_path, state="running", step="preflight_passed", completed=[])
    completed = []
    try:
        for name, config_path, run_id, _, _ in PILOT:
            run = root / "outputs/experiments" / run_id
            run.mkdir()
            write_status(status_path, state="running", step=f"{name}:training", completed=completed)
            run_command(root, ["scripts/train.py", "--config", config_path, "--run-id", run_id],
                        run / "train_stdout.log", run / "train_stderr.log")
            write_status(status_path, state="running", step=f"{name}:official_val_recompute", completed=completed)
            run_command(root, ["scripts/analyze_validation.py", "--config", config_path, "--run-dir", str(run)],
                        output / f"{name}_val_stdout.log", output / f"{name}_val_stderr.log")
            write_status(status_path, state="running", step=f"{name}:oof_recompute", completed=completed)
            run_command(root, ["scripts/analyze_phase17_oof.py", "--config", config_path, "--run-dir", str(run)],
                        output / f"{name}_oof_stdout.log", output / f"{name}_oof_stderr.log")
            write_status(status_path, state="running", step=f"{name}:freeze", completed=completed)
            run_command(root, ["scripts/freeze_phase17_fold.py", "--config", config_path, "--run-dir", str(run)],
                        output / f"{name}_freeze_stdout.log", output / f"{name}_freeze_stderr.log")
            completed.append(name)
        write_status(status_path, state="running", step="fixed_fusion_analysis", completed=completed)
        run_command(root, ["scripts/analyze_phase17_pilot.py", "--config", PILOT[0][1],
                           "--d-run-dir", "outputs/experiments/d17_f1_20261019",
                           "--c-run-dir", "outputs/experiments/c17_f1_20261019",
                           "--s-run-dir", "outputs/experiments/s17_f1_20261019",
                           "--output-dir", "outputs/analysis/phase17_fold1_fixed_fusion_20261019"],
                    output / "fusion_stdout.log", output / "fusion_stderr.log")
        write_status(status_path, state="complete", step="done", completed=completed)
    except Exception as exc:
        write_status(status_path, state="failed", step="error", completed=completed, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
