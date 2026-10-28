"""Bounded real-data diagnostic of one failed Phase41 task; never writes formal results."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.phase39_run import optimizer_scheduler, step_group
from cxr.phase39_support import save_torch, write_json
from cxr.phase41.protocol import digest, load_plan, resolved_config, server_config
from cxr.phase41.train import build, fold_rows, loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", default="P41_FUSION_F02")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.steps <= 128:
        raise ValueError("Diagnostic steps must be 1..128")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 1 or not visible.isdecimal():
        raise ValueError("Bind exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    server, plan = server_config(args.config), load_plan()
    if int(visible) not in server["gpu_ids"] or args.task == "P41_BASE_F02":
        raise ValueError("Choose an authorized failed task and GPU")
    task = next(t for t in plan["tasks"] if t["id"] == args.task)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    out = Path(server["output_root"]) / "diagnostic" / args.task
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise ValueError("Diagnostic already exists; use a fresh output root")
    os.environ["TORCH_HOME"] = server["pretrained_root"]
    cfg = resolved_config(plan, task)
    train, oof = fold_rows(plan, cfg, Path(server["data_root"]), Path(server["protocol_root"]))
    tr, _ = loaders(cfg, Path(server["data_root"]), train, oof)
    model = build(task["variant"], cfg, pretrained=True).cuda()
    ema = AveragedModel(model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)
    opt, sch = optimizer_scheduler(cfg, model, math.ceil(len(tr) / cfg["accumulation"]))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    iterator = iter(tr); events = []; updates = 0; skips = 0; replay = None
    try:
        for step in range(1, args.steps + 1):
            micro = list(itertools.islice(iterator, cfg["accumulation"]))
            if not micro:
                raise RuntimeError("Diagnostic data exhausted")
            if step == 1:
                save_torch(out / "probe_before.pt", dict(model=model.state_dict(), ema=ema.state_dict(),
                           optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict(),
                           cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
            diag = {}
            value, skipped = step_group(cfg, model, ema, micro, opt, sch, scaler,
                                        torch.device("cuda"), diag)
            if step == 1:
                first_value, first_skipped = value, skipped
                before = torch.load(out / "probe_before.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(before["model"]); ema.load_state_dict(before["ema"])
                opt.load_state_dict(before["optimizer"]); sch.load_state_dict(before["scheduler"])
                scaler.load_state_dict(before["scaler"])
                torch.set_rng_state(before["cpu_rng"]); torch.cuda.set_rng_state(before["cuda_rng"])
                diag = {}
                value, skipped = step_group(cfg, model, ema, micro, opt, sch, scaler,
                                            torch.device("cuda"), diag)
                replay = dict(same_batch_loss_delta=abs(value - first_value),
                              same_skip_decision=skipped == first_skipped)
                if replay["same_batch_loss_delta"] > 1e-4 or not replay["same_skip_decision"]:
                    raise RuntimeError("Same-batch diagnostic replay is unstable")
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite diagnostic loss")
            updates += not skipped; skips += skipped
            event = dict(step=step, loss=value, skipped=skipped, scale=float(scaler.get_scale()), **diag)
            events.append(event)
            if skips >= 20 or (len(events) >= 4 and all(x["skipped"] for x in events[-4:])):
                raise FloatingPointError("Persistent AMP overflow in bounded diagnostic")
        save_torch(out / "resume.pt", dict(model=model.state_dict(), ema=ema.state_dict(),
                  optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict()))
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["optimizer"]); sch.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        if not updates or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("No optimizer update or nonfinite restored parameter")
        result = dict(state="PASS", task=args.task, steps=args.steps, optimizer_updates=updates,
                      amp_skips=skips, checkpoint_reload="PASS", same_batch_replay=replay, events=events)
    except Exception as exc:
        opt.zero_grad(set_to_none=True)
        result = dict(state="FAIL", task=args.task, reason=f"{type(exc).__name__}: {exc}", events=events)
        write_json(out / "result.json", result)
        raise
    result.update(config_sha256=digest(args.config),
                  training_code_sha256=digest(ROOT / "src/cxr/phase39_run.py"),
                  historical_done_sha256=digest(Path(server["recovery_source_root"]) /
                                                "experiments/P41_BASE_F02/DONE.json"))
    write_json(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "events"}))


if __name__ == "__main__":
    main()
