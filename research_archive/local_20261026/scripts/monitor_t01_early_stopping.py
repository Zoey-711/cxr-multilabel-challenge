"""Read-only, live T01 early-stopping audit with an explicit x/6 epoch log."""

import argparse
import json
import time
from pathlib import Path

from cxr.engine.early_stopping import EarlyStopTracker
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    if config != load_config(args.run_dir / "config.yaml"):
        raise ValueError("T01 runtime config differs")
    training = config["training"]
    if training["early_stopping_patience"] != 6 or training.get("early_stopping_min_delta", 0) != 0 or training["epochs"] != 30:
        raise ValueError("T01 early-stopping settings differ from approval")
    tracker = EarlyStopTracker(patience=6, min_delta=0.0)
    observed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", buffering=1) as handle:
        handle.write("monitor=val_study_macro_auc (implementation: study_level_macro_auc), mode=max, min_delta=0, patience=6, max_epochs=30\n")
        while True:
            metrics_path = args.run_dir / "metrics.jsonl"
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()] if metrics_path.exists() else []
            for row in records[observed:]:
                epoch = int(row["epoch"])
                decision = tracker.observe(float(row["val"]["macro_auc"]), epoch)
                if (row["best_epoch"] != tracker.best_epoch or row["best_auc"] != tracker.best_auc or
                        row["stale"] != tracker.stale or row["best_checkpoint_saved"] != decision.new_best_checkpoint or
                        row["patience_reset"] != decision.patience_reset or
                        (row["stop_reason"] == "patience_exhausted") != decision.should_stop):
                    raise ValueError(f"Early-stopping replay mismatch at Epoch {epoch}")
                line = (f"Epoch={epoch} Train Loss={row['train_loss']:.5f} "
                        f"Val Macro-AUC={row['val']['macro_auc']:.5f} Best Epoch={tracker.best_epoch} "
                        f"Best AUC={tracker.best_auc:.5f} Early Stop Counter={tracker.stale}/6 "
                        f"Checkpoint Saved={decision.new_best_checkpoint} Stop Reason={row['stop_reason']}\n")
                handle.write(line)
                print(line, end="", flush=True)
                observed += 1
            status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
            if status["state"] in {"complete", "failed"}:
                if status["state"] == "complete" and observed != status["stop_epoch"]:
                    raise ValueError("Completed T01 has missing epoch records")
                handle.write(f"final_state={status['state']} stop_epoch={status.get('stop_epoch')} best_epoch={tracker.best_epoch}\n")
                return
            time.sleep(20)


if __name__ == "__main__":
    main()
