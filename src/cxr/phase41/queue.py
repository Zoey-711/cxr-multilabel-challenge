"""One-host GPU scheduler with persistent state, safe resume, and budget cap."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from cxr.batch.plan import write_json
from cxr.phase35.queue import lock
from cxr.phase41.protocol import ROOT, digest, fingerprint


def verify_done(folder: Path, identity: str) -> dict:
    receipt = json.loads((folder / "DONE.json").read_text(encoding="utf-8"))
    if receipt["identity_sha256"] != identity or any(digest(folder / p) != sha for p, sha in receipt["sha256"].items()):
        raise ValueError(f"Completed task was modified: {folder.name}")
    return receipt


def report(root: Path, state: dict, plan: dict) -> None:
    folder = root / "reports"; folder.mkdir(parents=True, exist_ok=True)
    path = folder / "experiment_status.csv"; temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        fields = ("task_id", "variant", "fold", "state", "gpu", "attempts", "epoch", "best_epoch", "best_auc", "reason")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for task in plan["tasks"]:
            tid = task["id"]; rec = state["tasks"][tid]; progress = root / "experiments" / tid / "status.json"
            metrics = json.loads(progress.read_text()) if progress.exists() else {}
            writer.writerow(dict(task_id=tid, variant=task["variant"], fold=task["fold"], state=rec["state"],
                                 gpu=rec.get("gpu", ""), attempts=rec["attempts"], epoch=metrics.get("epoch", ""),
                                 best_epoch=metrics.get("best_epoch", ""), best_auc=metrics.get("best_auc", ""),
                                 reason=rec.get("reason", "")))
    os.replace(temp, path)
    write_json(folder / "queue_summary.json", dict(gpu_hours_used=state["gpu_hours_used"],
               budget_hours=state["budget_hours"], counts={kind: sum(r["state"] == kind for r in state["tasks"].values())
               for kind in ("PENDING", "RUNNING", "DONE", "FAILED", "BLOCKED")}))


def worker(task: dict, root: Path, server_file: Path, gpu: int, identity: str, attempt: int) -> dict:
    logs = root / "queue_logs"; logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / f"{task['id']}_attempt_{attempt}.log"
    sidecar = logs / f"{task['id']}.pid.json"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts/phase41_worker.py"), "--config", str(server_file),
               "--mode", "run", "--task", task["id"], "--gpu", str(gpu), "--identity-hash", identity]
    started = time.monotonic()
    with stdout.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        write_json(sidecar, dict(host=socket.gethostname(), pid=process.pid, task=task["id"], gpu=gpu))
        try:
            code = process.wait(timeout=task["timeout_hours"] * 3600)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=60)
            code = -1
            with stdout.open("a", encoding="utf-8") as extra:
                extra.write("\nTIMEOUT: training stopped; last complete Epoch checkpoint retained\n")
        finally:
            sidecar.unlink(missing_ok=True)
    hours = (time.monotonic() - started) / 3600
    if code == 0:
        verify_done(root / "experiments" / task["id"], identity)
        return dict(state="DONE", gpu_hours=hours)
    tail = stdout.read_text(encoding="utf-8", errors="replace")[-4000:]
    retryable = any(marker in tail for marker in ("Temporary failure", "Connection reset", "Resource temporarily unavailable"))
    return dict(state="FAILED", gpu_hours=hours, reason=f"exit={code}: {tail[-900:]}", retryable=retryable)


def run(plan: dict, cfg: dict, server_file: Path, identity: str, preflight_hours: float = 0., poll=3.) -> dict:
    root = Path(cfg["output_root"]); root.mkdir(parents=True, exist_ok=True)
    guard = lock(root)
    try:
        path = root / "queue_state.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if (state["plan_fingerprint"] != fingerprint(plan) or state["identity_sha256"] != identity or
                    state["budget_hours"] != cfg["gpu_hours"]):
                raise ValueError("Queue identity or authorized budget changed; inspect before continuing")
        else:
            state = dict(plan_fingerprint=fingerprint(plan), identity_sha256=identity,
                         gpu_hours_used=preflight_hours, budget_hours=cfg["gpu_hours"],
                         tasks={t["id"]: dict(state="PENDING", attempts=0) for t in plan["tasks"]})
        for task in plan["tasks"]:
            rec = state["tasks"][task["id"]]
            folder = root / "experiments" / task["id"]
            if (folder / "DONE.json").exists():
                verify_done(folder, identity)
                if rec["state"] == "RUNNING":
                    # The worker may have finished before the queue supervisor was
                    # interrupted. Charge elapsed GPU time exactly once on recovery.
                    elapsed = min(task["timeout_hours"], max(0., (time.time() - rec["started_unix"]) / 3600))
                    state["gpu_hours_used"] += elapsed
                    rec["charged_hours"] = elapsed
                rec["state"] = "DONE"
            elif rec["state"] == "RUNNING":
                # queue.lock verified that old parent and all child PIDs are dead.
                elapsed = min(task["timeout_hours"], max(0., (time.time() - rec["started_unix"]) / 3600))
                state["gpu_hours_used"] += elapsed
                rec.update(state="PENDING", reason="Interrupted; resume from last complete Epoch", charged_hours=elapsed)
            elif rec["state"] == "DONE":
                raise ValueError(f"DONE receipt missing: {task['id']}")
        write_json(path, state); report(root, state, plan)
        active = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["max_concurrent"]) as pool:
            while True:
                for future, (task, gpu) in list(active.items()):
                    if not future.done():
                        continue
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(state="FAILED", gpu_hours=task["timeout_hours"],
                                      reason=f"Worker supervisor: {type(exc).__name__}: {exc}", retryable=False)
                    state["gpu_hours_used"] += result.pop("gpu_hours")
                    rec = state["tasks"][task["id"]]; rec.update(result)
                    if rec["state"] == "FAILED":
                        folder = root / "experiments" / task["id"]; folder.mkdir(parents=True, exist_ok=True)
                        write_json(folder / f"FAILED_attempt_{rec['attempts']}.json", rec)
                        if rec.get("retryable") and rec["attempts"] <= task["max_retries"]:
                            rec["state"] = "PENDING"
                        else:
                            write_json(folder / "FAILED.json", rec)
                    del active[future]
                    write_json(path, state); report(root, state, plan)
                    if cfg["upload"].get("enabled"):
                        try:
                            from cxr.phase41.upload import upload
                            upload(root, cfg["upload"])
                        except Exception as exc:
                            write_json(root / "reports/UPLOAD_FAILED.json", dict(
                                type=type(exc).__name__, reason=str(exc), task=task["id"]))
                busy = {gpu for _, gpu in active.values()}
                for gpu in cfg["gpu_ids"]:
                    if gpu in busy or len(active) >= cfg["max_concurrent"] or (root / "STOP").exists():
                        continue
                    for task in plan["tasks"]:
                        rec = state["tasks"][task["id"]]
                        if rec["state"] != "PENDING":
                            continue
                        reserved = sum(t["timeout_hours"] for t, _ in active.values())
                        if state["gpu_hours_used"] + reserved + task["timeout_hours"] > cfg["gpu_hours"]:
                            continue
                        rec.update(state="RUNNING", gpu=gpu, attempts=rec["attempts"] + 1,
                                   started_unix=time.time(), reserved_hours=task["timeout_hours"])
                        write_json(path, state)
                        active[pool.submit(worker, task, root, server_file, gpu, identity, rec["attempts"])] = (task, gpu)
                        break
                report(root, state, plan)
                if not active:
                    break
                time.sleep(poll)
        write_json(path, state); report(root, state, plan)
        return state
    finally:
        guard.unlink(missing_ok=True)
