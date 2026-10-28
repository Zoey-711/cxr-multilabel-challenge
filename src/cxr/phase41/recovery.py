"""Import the one completed historical fold into an independent repair run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from cxr.batch.plan import write_json
from cxr.phase41.protocol import digest

FROZEN_TASK = "P41_BASE_F02"


def import_completed_base(source_root: Path, output_root: Path, identity_hash: str) -> dict:
    source = Path(source_root).resolve() / "experiments" / FROZEN_TASK
    destination = Path(output_root).resolve() / "experiments" / FROZEN_TASK
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Recovery destination must be independent of historical results")
    source_done = source / "DONE.json"
    receipt = json.loads(source_done.read_text(encoding="utf-8"))
    files = receipt.get("sha256", {})
    required = {"ema_best.pth", "oof_predictions.csv", "val_predictions.csv", "test_predictions.csv",
                "resolved_config.yaml", "run_manifest.json", "metrics.json"}
    if receipt.get("state") != "DONE" or not required.issubset(files) or any(
            Path(name).name != name or digest(source / name) != sha for name, sha in files.items()):
        raise ValueError("Historical BASE Fold2 receipt or artifact hashes invalid")
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != FROZEN_TASK or manifest.get("identity_sha256") != receipt.get("identity_sha256"):
        raise ValueError("Historical BASE Fold2 provenance mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "DONE.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (existing.get("identity_sha256") != identity_hash or
                existing.get("reused_source_sha256") != digest(source_done) or
                any(digest(destination / name) != sha for name, sha in files.items())):
            raise ValueError("Previously imported Fold2 changed")
        return dict(state="REUSED", task=FROZEN_TASK, source=str(source), source_sha256=digest(source_done))
    for name, sha in files.items():
        target = destination / name
        if target.exists() and digest(target) != sha:
            raise ValueError(f"Recovery target conflict: {name}")
        if not target.exists():
            shutil.copy2(source / name, target)
        if digest(target) != sha:
            raise ValueError(f"Recovery copy failed: {name}")
    write_json(marker, dict(state="DONE", identity_sha256=identity_hash, sha256=files,
                            reused_source_sha256=digest(source_done),
                            historical_identity_sha256=receipt["identity_sha256"],
                            training_reused=True))
    return dict(state="REUSED", task=FROZEN_TASK, source=str(source), source_sha256=digest(source_done))
