"""One Phase41 command: preflight, eight paired folds, analysis, optional upload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.batch.host_gpu_lock import host_gpu_lock
from cxr.batch.plan import write_json
from cxr.phase41.protocol import load_plan, server_config, fingerprint, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--action", choices=("run", "validate", "preflight", "status", "stop", "resume", "analyze", "upload"), default="run")
    args = parser.parse_args()
    plan = load_plan()
    cfg = server_config(args.config)
    root = Path(cfg["output_root"])
    if args.action == "validate":
        print("PASS: 8 frozen Fold2-5 tasks; 2 Fold1 assets reused; GPU work is locked behind REMOTE_PREFLIGHT")
        return
    if args.action == "status":
        for name in ("reports/queue_summary.json", "reports/experiment_status.csv", "reports/fivefold_summary.json"):
            path = root / name
            print(f"{name}:\n{path.read_text(encoding='utf-8') if path.exists() else 'NOT_AVAILABLE'}")
        return
    if args.action == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        print("Drain stop requested: current GPU tasks finish; no new task will launch")
        return
    if args.action in ("analyze", "upload"):
        with host_gpu_lock():
            if args.action == "analyze":
                from cxr.phase41.analysis import analyze
                print(json.dumps(analyze(root, plan, cfg)))
            else:
                from cxr.phase41.upload import upload
                upload(root, cfg["upload"])
        return
    with host_gpu_lock():
        root.mkdir(parents=True, exist_ok=True)
        if args.action == "resume":
            (root / "STOP").unlink(missing_ok=True)
        elif (root / "STOP").exists():
            raise RuntimeError("STOP present; use --action resume after confirming children are done")
        from cxr.phase41.preflight import prepare, preflight
        from cxr.phase41.queue import run
        from cxr.phase41.analysis import analyze
        try:
            identity = prepare(plan, cfg)
            receipt = preflight(plan, cfg, args.config.resolve(), identity)
            if receipt["state"] != "PASS" or receipt["identity_sha256"] != fingerprint(identity):
                raise ValueError("REMOTE_PREFLIGHT receipt invalid")
        except Exception as exc:
            write_json(root / "reports/PREFLIGHT_FAILED.json", dict(type=type(exc).__name__, reason=str(exc)))
            raise
        if args.action == "preflight":
            print("REMOTE_PREFLIGHT PASS; formal training not started")
            return
        if cfg.get("recovery_source_root"):
            diagnosis = root / "diagnostic/P41_FUSION_F02/result.json"
            if not diagnosis.exists():
                raise RuntimeError("Phase41-R requires bounded real-data diagnostic before recovery")
            check = json.loads(diagnosis.read_text(encoding="utf-8"))
            if (check.get("state") != "PASS" or check.get("steps", 0) < 32 or
                    check.get("optimizer_updates", 0) < 1 or
                    check.get("config_sha256") != digest(args.config) or
                    check.get("training_code_sha256") != digest(ROOT / "src/cxr/phase39_run.py") or
                    check.get("historical_done_sha256") != identity["historical_base_f02_done_sha256"]):
                raise ValueError("Phase41-R real-data diagnostic missing or changed")
            from cxr.phase41.recovery import import_completed_base
            imported = import_completed_base(Path(cfg["recovery_source_root"]), root, fingerprint(identity))
            write_json(root / "reports/RECOVERY_IMPORT.json", imported)
        state = run(plan, cfg, args.config.resolve(), fingerprint(identity), receipt["gpu_hours"])
        result = analyze(root, plan, cfg)
        if cfg["upload"].get("enabled"):
            try:
                from cxr.phase41.upload import upload
                upload(root, cfg["upload"])
            except Exception as exc:
                write_json(root / "reports/UPLOAD_FAILED.json", dict(type=type(exc).__name__, reason=str(exc)))
        print(json.dumps(dict(tasks={k: v["state"] for k, v in state["tasks"].items()},
                              analysis=result["state"], gpu_hours_used=state["gpu_hours_used"])))


if __name__ == "__main__":
    main()
