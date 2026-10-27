"""Mandatory Linux 4090 checks before Phase41 can start any formal task."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cxr.batch.plan import write_json
from cxr.phase41.protocol import ROOT, digest, fingerprint
from cxr.phase41.train import fold_rows


def _run(command: list[str], log: Path, timeout: float, env: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Remote preflight failed (exit {result.returncode}); see {log}")


def prepare(plan: dict, cfg: dict) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Formal Phase41 preflight requires the friend's authorized Linux server")
    out = Path(cfg["output_root"]); out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == ROOT.resolve() or ROOT.resolve() in out.resolve().parents:
        raise ValueError("Output must be independent of the code checkout")
    if shutil.disk_usage(out).free < cfg["minimum_free_disk_gib"] * 2**30:
        raise RuntimeError("Insufficient output disk; retain frozen Fold1 and checkpoints")
    if cfg["upload"].get("enabled"):
        if shutil.which("gh") is None or shutil.which("git") is None:
            raise RuntimeError("Opt-in result upload needs authenticated gh and git")
        permission = json.loads(subprocess.check_output([
            "gh", "repo", "view", cfg["upload"]["repository"], "--json", "isPrivate,viewerPermission"], text=True))
        if not permission["isPrivate"] or permission["viewerPermission"] not in ("ADMIN", "WRITE", "MAINTAIN"):
            raise RuntimeError("Private result repository write permission missing")
    if (os.cpu_count() or 0) < 2 * cfg["max_concurrent"]:
        raise RuntimeError("Insufficient CPU cores for configured concurrent workers")
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
    if mem["MemAvailable"] < cfg["minimum_free_memory_gib_per_worker"] * cfg["max_concurrent"] * 2**30:
        raise RuntimeError("Insufficient available RAM for configured concurrency")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    devices = {int(parts[0]): parts[1:] for line in raw.strip().splitlines()
               if (parts := [v.strip() for v in line.split(",")])}
    if len(devices) < 8 or any("4090" not in info[0] or float(info[1]) < 22000 for info in devices.values()):
        raise RuntimeError("Expected one Linux host with eight 24GB RTX4090 GPUs")
    for gpu in cfg["gpu_ids"]:
        info = devices.get(gpu)
        if info is None or float(info[2]) > 500 or float(info[3]) > 5:
            raise RuntimeError(f"Authorized GPU {gpu} is occupied")
    for name, expected in plan["fold_sha256"].items():
        if digest(Path(cfg["protocol_root"]) / name) != expected:
            raise ValueError(f"Frozen patient fold file mismatch: {name}")
    cached = Path(cfg["pretrained_root"]) / "hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth"
    if digest(cached) != plan["pretrained_sha256"]:
        raise ValueError("Approved ImageNet initialization missing or changed")
    from cxr.data.archive import read_split_rows
    archive = Path(cfg["data_root"])
    train, val, test = [read_split_rows(archive, split) for split in ("train", "val", "test")]
    if [len(rows) for rows in (train, val, test)] != [30344, 1422, 2218]:
        raise ValueError("Official Train/Val/Test image counts differ")
    if len({(r["Subject_id"], r["Study_id"]) for r in test}) != 1414:
        raise ValueError("Official Test Study count differs")
    if {r["Subject_id"] for r in train} & {r["Subject_id"] for r in val}:
        raise ValueError("Official Train/Val patient overlap")
    for task in plan["tasks"]:
        from cxr.phase41.protocol import resolved_config
        fold_rows(plan, resolved_config(plan, task), archive, Path(cfg["protocol_root"]))
    reused = out / "reused_fold1"
    for variant, record in plan["fold1"].items():
        source = Path(cfg["fold1_results_root"]) / record["model_id"]
        target = reused / record["model_id"]
        target.mkdir(parents=True, exist_ok=True)
        for name, expected in record["assets"].items():
            src, dst = source / name, target / name
            if digest(src) != expected:
                raise ValueError(f"Frozen Fold1 source mismatch: {variant}/{name}")
            if dst.exists() and digest(dst) != expected:
                raise ValueError(f"Existing reused Fold1 artifact differs: {variant}/{name}")
            if not dst.exists():
                shutil.copy2(src, dst)
        manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["task"] != record["model_id"] or manifest["best_epoch"] != 4 or manifest["official_val_accessed"] or manifest["test_accessed"]:
            raise ValueError("Fold1 provenance or training split changed")
    source_names = ("src/cxr/phase39_model.py", "src/cxr/phase39_run.py", "src/cxr/phase41/protocol.py",
                    "src/cxr/phase41/train.py", "src/cxr/phase41/queue.py", "src/cxr/phase41/preflight.py",
                    "src/cxr/phase41/analysis.py", "src/cxr/phase41/upload.py",
                    "scripts/phase41_worker.py", "scripts/phase41_run.py")
    from cxr.batch.artifacts import commit
    return dict(host=platform.node(), code_commit=commit(), plan_fingerprint=fingerprint(plan),
                code_sha256={name: digest(ROOT / name) for name in source_names},
                data_archive_bytes=archive.stat().st_size,
                fold1_sha256={variant: record["assets"] for variant, record in plan["fold1"].items()},
                devices={str(i): devices[i][:2] for i in cfg["gpu_ids"]},
                authorization=dict(gpu_ids=cfg["gpu_ids"], max_concurrent=cfg["max_concurrent"],
                                   gpu_hours=cfg["gpu_hours"]))


def preflight(plan: dict, cfg: dict, config_path: Path, identity: dict) -> dict:
    root = Path(cfg["output_root"]); folder = root / "preflight"; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "REMOTE_PREFLIGHT.json"
    identity_hash = fingerprint(identity)
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["identity_sha256"] != identity_hash or len(record["probes"]) != len(cfg["gpu_ids"]):
            raise ValueError("Existing remote preflight identity/probes changed")
        for probe in record["probes"]:
            path = root / probe["path"]
            if digest(path) != probe["sha256"]:
                raise ValueError("Preflight probe receipt changed")
        if set(record.get("fold1_done", [])) != {"BASE", "FUSION"}:
            raise ValueError("Frozen Fold1 inference preflight incomplete")
        for variant in ("BASE", "FUSION"):
            source = plan["fold1"][variant]
            folder = root / "reused_fold1" / source["model_id"]
            for name, expected in source["assets"].items():
                if digest(folder / name) != expected:
                    raise ValueError(f"Frozen Fold1 reuse changed: {variant}/{name}")
            reused = json.loads((folder / "REUSED.json").read_text(encoding="utf-8"))
            if any(digest(folder / name) != sha for name, sha in reused["sha256"].items()):
                raise ValueError(f"Frozen Fold1 inference changed: {variant}")
        return record
    ledger = folder / "ledger.json"
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(identity_sha256=identity_hash, gpu_hours=0., probes=[], fold1_done=[])
    if state["identity_sha256"] != identity_hash or state.get("running"):
        raise RuntimeError("Interrupted or changed preflight; inspect child PIDs before retry")
    base_env = os.environ.copy(); base_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + base_env.get("PYTHONPATH", "")
    for index, gpu in enumerate(cfg["gpu_ids"]):
        if any(item["gpu"] == gpu for item in state["probes"]):
            continue
        task = plan["tasks"][index]
        reserve = .5
        if state["gpu_hours"] + reserve > cfg["gpu_hours"]:
            raise RuntimeError("GPU-hour budget insufficient for mandatory remote preflight")
        probe = folder / f"gpu_{gpu}"; probe.mkdir(exist_ok=True)
        env = base_env.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [sys.executable, str(ROOT / "scripts/phase41_worker.py"), "--config", str(config_path),
                   "--mode", "smoke", "--task", task["id"], "--gpu", str(gpu), "--probe", str(probe)]
        state["running"] = task["id"]; write_json(ledger, state); start = time.monotonic()
        try:
            _run(command, probe / "preflight.log", reserve * 3600, env)
            result = json.loads((probe / "receipt.json").read_text())
            if result["state"] != "PASS" or result["peak_reserved_mib"] >= 22 * 1024:
                raise RuntimeError("Remote GPU smoke result or VRAM margin failed")
            state["probes"].append(dict(gpu=gpu, task=task["id"], path=(probe / "receipt.json").relative_to(root).as_posix(),
                                        sha256=digest(probe / "receipt.json")))
        finally:
            state["gpu_hours"] += (time.monotonic() - start) / 3600
            state.pop("running", None); write_json(ledger, state)
    for variant in ("BASE", "FUSION"):
        if variant in state["fold1_done"]:
            continue
        reserve = 1.
        if state["gpu_hours"] + reserve > cfg["gpu_hours"]:
            raise RuntimeError("GPU-hour budget insufficient for frozen Fold1 Val/Test inference")
        env = base_env.copy(); env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_ids"][0])
        command = [sys.executable, str(ROOT / "scripts/phase41_worker.py"), "--config", str(config_path),
                   "--mode", "fold1-infer", "--variant", variant, "--gpu", str(cfg["gpu_ids"][0])]
        state["running"] = "fold1-" + variant; write_json(ledger, state); start = time.monotonic()
        try:
            _run(command, folder / f"fold1_{variant}.log", reserve * 3600, env)
            state["fold1_done"].append(variant)
        finally:
            state["gpu_hours"] += (time.monotonic() - start) / 3600
            state.pop("running", None); write_json(ledger, state)
    if len(state["probes"]) != len(cfg["gpu_ids"]) or len(state["fold1_done"]) != 2:
        raise RuntimeError("Remote preflight incomplete")
    state["state"] = "PASS"
    write_json(receipt, state)
    return state
