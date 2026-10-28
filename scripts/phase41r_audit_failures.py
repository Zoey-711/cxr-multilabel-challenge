"""Read-only audit of historical FAILED tasks; writes report only to repair root."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from cxr.phase39_support import write_json
from cxr.phase41.protocol import digest, server_config


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def audit(source: Path) -> list[dict]:
    status = source / "reports/experiment_status.csv"
    with status.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    result = []
    for row in records:
        task = row["task_id"]
        raw_reason = row.get("reason", "")
        reason = "Nonfinite gradients" if "Nonfinite gradients" in raw_reason else (
                 "OTHER_FAILURE" if raw_reason else "")
        item = dict(task=task, state=row["state"], gpu=row.get("gpu"),
                    reported_epoch=row.get("epoch"), reason=reason)
        folder = source / "experiments" / task
        checkpoint = folder / "last_training.pt"
        if checkpoint.exists():
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                required = {"model", "ema", "optimizer", "scheduler", "scaler", "history", "tracker",
                            "epoch", "task", "identity_sha256", "config_sha256"}
                item.update(checkpoint_state="FINITE" if required.issubset(state) and
                            state["task"] == task and finite_tree(state) else "INVALID",
                            checkpoint_epoch=state.get("epoch"), checkpoint_sha256=digest(checkpoint))
            except Exception as exc:
                item.update(checkpoint_state="UNREADABLE", checkpoint_error=f"{type(exc).__name__}: {exc}")
        else:
            item["checkpoint_state"] = "MISSING"
        logs = sorted((source / "queue_logs").glob(task + "_attempt_*.log"))
        item["log_count"] = len(logs)
        item["log_sha256"] = {p.name: digest(p) for p in logs}
        item["last_epoch_line"] = next((line[-500:] for p in reversed(logs) for line in
                        reversed(p.read_text(encoding="utf-8", errors="replace").splitlines())
                        if "EPOCH_DONE" in line), None)
        item["recovery_action"] = "REUSE_FROZEN_DONE" if row["state"] == "DONE" and task == "P41_BASE_F02" else "RESTART_IN_NEW_ROOT"
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = server_config(args.config)
    if "recovery_source_root" not in cfg:
        raise ValueError("Use a Phase41-R config with a distinct historical source root")
    items = audit(Path(cfg["recovery_source_root"]))
    out = Path(cfg["output_root"]) / "reports/failure_audit.json"
    write_json(out, dict(source_done_sha256=digest(Path(cfg["recovery_source_root"]) /
                    "experiments/P41_BASE_F02/DONE.json"), tasks=items))
    print(json.dumps({"report": str(out), "done": sum(x["state"] == "DONE" for x in items),
                      "failed": sum(x["state"] == "FAILED" for x in items)}))


if __name__ == "__main__":
    main()
