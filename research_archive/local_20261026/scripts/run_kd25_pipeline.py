"""Run the single approved KD25-D01 pilot, then independently seal its results."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / 'outputs/experiments/phase26_local_distillation/KD25_D01'
    run.mkdir(parents=True, exist_ok=True)
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
    if (run / 'manifest.json').exists():
        raise SystemExit('KD25-D01 is already frozen; refusing to rerun')
    resume = (run / 'last_training.pt').is_file() and status.get('state') != 'complete'
    if status and status.get('state') != 'complete' and not resume:
        raise SystemExit('Existing run has no full resume checkpoint; refusing to restart')
    with (run / 'training.log').open('a', encoding='utf-8', buffering=1) as stdout, \
            (run / 'training_error.log').open('a', encoding='utf-8', buffering=1) as stderr:
        if status.get('state') != 'complete':
            cmd = [sys.executable, '-u', str(root / 'scripts/train_kd25.py'),
                   '--config', str(args.config.resolve()), '--run-id', 'KD25_D01']
            if resume:
                cmd.append('--resume')
            result = subprocess.run(cmd, cwd=root, stdout=stdout, stderr=stderr, check=False)
            if result.returncode:
                raise SystemExit(f'Training failed with exit code {result.returncode}; see {run / "training_error.log"}')
        if json.loads(status_file.read_text(encoding='utf-8')).get('state') != 'complete':
            raise SystemExit('Training did not reach complete state; finalizer withheld')
        cmd = [sys.executable, '-u', str(root / 'scripts/finalize_kd25.py'),
               '--config', str(args.config.resolve())]
        result = subprocess.run(cmd, cwd=root, stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            raise SystemExit(f'Finalization failed with exit code {result.returncode}; see {run / "training_error.log"}')


if __name__ == '__main__':
    main()
