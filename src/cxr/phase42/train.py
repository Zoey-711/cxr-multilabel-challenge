"""Phase42 trainer adapted from Phase41; fixed new head, identical fold protocol."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.data.patient_folds import select_fold_rows
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.phase42.models import build_model
from cxr.phase39_run import infer
from cxr.phase42.methods import optimizer_scheduler, step_group
from cxr.phase39_support import digest, save_torch, seed_all, write_csv, write_json
from cxr.phase42.protocol import ROOT, fingerprint, resolved_config
from cxr.utils.training_resume import capture_rng_state, restore_rng_state


def model_kind(variant: str) -> str:
    return variant


def fold_rows(plan: dict, cfg: dict, archive: Path, protocol_root: Path):
    all_rows = read_split_rows(archive, "train")
    fold = dict(kind="patient_fold_v1", directory=str(protocol_root), fold=cfg["fold"],
                sha256=plan["fold_sha256"], expected_counts=cfg["fold_expected_counts"])
    train = select_fold_rows(all_rows, fold, role="train")
    oof = select_fold_rows(all_rows, fold, role="oof")
    if {r["Subject_id"] for r in train} & {r["Subject_id"] for r in oof}:
        raise ValueError("Train/OOF patient overlap")
    return train, oof


def loaders(cfg: dict, archive: Path, train_rows: list[dict], oof_rows: list[dict]):
    aug = dict(enabled=True, name="conservative_cxr_v1", affine_probability=.5,
               rotation_degrees=3., translation_fraction=.02, photometric_probability=.5,
               brightness_contrast_fraction=.05)
    # Phase39 Fold1 used num_workers=0, microbatch=4 and accumulation=4.  Keep
    # these for both variants: BatchNorm sees the same number of images/update.
    shared = dict(batch_size=4, num_workers=0, preprocessing="black_pad_v1",
                  input_size=384, persistent_workers=False)
    tr = make_dataloader(archive, "train", rows=train_rows, shuffle=True, seed=42,
                         augmentation=aug, **shared)
    ev = make_dataloader(archive, "train", rows=oof_rows, shuffle=False, seed=43, **shared)
    return tr, ev


def build(variant: str, cfg: dict, pretrained: bool):
    seed_all(cfg["seed"])
    return build_model(model_kind(variant), pretrained=pretrained)




def real_smoke(plan: dict, task: dict, server: dict, probe: Path, device=None) -> dict:
    """One real AMP optimizer step and a restored OOF/Val mini-forward."""
    cfg = resolved_config(plan, task)
    os.environ["TORCH_HOME"] = server["pretrained_root"]
    archive = Path(server["data_root"])
    train, oof = fold_rows(plan, cfg, archive, Path(server["protocol_root"]))
    tr, ev = loaders(cfg, archive, train[:16], oof[:4])
    device = torch.device('cuda') if device is None else torch.device(device)
    model = build(task["variant"], cfg, pretrained=device.type=='cuda').to(device)
    ema = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
    optimizer, scheduler = optimizer_scheduler(cfg, model, 1)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=device.type=='cuda')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    loss, skipped = step_group(cfg, model, ema, list(itertools.islice(iter(tr), 4)),
                               optimizer, scheduler, scaler, device)
    if skipped or not math.isfinite(loss) or int(ema.n_averaged) != 1:
        raise FloatingPointError("Real-data AMP preflight failed")
    probe.mkdir(parents=True, exist_ok=True)
    resume = probe / "resume.pt"
    save_torch(resume, dict(model=model.state_dict(), ema=ema.state_dict(),
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            scaler=scaler.state_dict(), **capture_rng_state(tr, ev)))
    state = torch.load(resume, map_location="cpu", weights_only=True)
    del model, ema, optimizer, scheduler, scaler
    model = build(task['variant'], cfg, pretrained=False).to(device)
    ema = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
    optimizer, scheduler = optimizer_scheduler(cfg, model, 1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type=='cuda')
    model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"]); restore_rng_state(state, tr, ev)
    resumed_loss, resumed_skip = step_group(cfg, model, ema, list(itertools.islice(iter(tr), 4)), optimizer, scheduler, scaler, device)
    if resumed_skip or not math.isfinite(resumed_loss) or int(ema.n_averaged)!=2:
        raise FloatingPointError('Restored optimizer/EMA continuation failed')
    ema.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type=='cuda'):
        oof_logits = ema(next(iter(ev))["image"].to(device))
        val_loader = make_dataloader(archive, "val", rows=read_split_rows(archive, "val")[:2],
                                      batch_size=1, num_workers=0, input_size=384, preprocessing="black_pad_v1")
        val_logits = ema(next(iter(val_loader))['image'].to(device))
    if any(not bool(torch.isfinite(x).all()) for x in (oof_logits, val_logits)):
        raise FloatingPointError("Nonfinite OOF/Val probe")
    result = dict(state="PASS", task=task["id"], gpu=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU_SYNTHETIC_TEST',
                  loss=loss, resumed_loss=resumed_loss, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device.type=='cuda' else 0.,
                  resume="PASS", oof_val_forward="PASS", amp_skips=0)
    write_json(probe / "receipt.json", result)
    return result


def run_task(plan: dict, task: dict, server: dict, identity_hash: str, device=None) -> dict:
    device = torch.device('cuda') if device is None else torch.device(device)
    cfg = resolved_config(plan, task)
    if cfg["micro_batch"] != 4 or cfg["accumulation"] != 4 or cfg["ema_use_buffers"] is not True:
        raise ValueError("Phase39 batch/BatchNorm/EMA protocol changed")
    run = Path(server["output_root"]) / "experiments" / task["id"]
    run.mkdir(parents=True, exist_ok=True)
    done = run / "DONE.json"
    if done.exists():
        record = json.loads(done.read_text(encoding="utf-8"))
        if record["identity_sha256"] != identity_hash or any(digest(run / p) != h for p, h in record["sha256"].items()):
            raise ValueError("Completed task artifact or identity changed")
        return dict(state="DONE", reused=True)
    resolved = run / "resolved_config.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if resolved.exists() and resolved.read_text(encoding="utf-8") != text:
        raise ValueError("Resolved training config changed")
    if not resolved.exists():
        resolved.write_text(text, encoding="utf-8")
    os.environ["TORCH_HOME"] = server["pretrained_root"]
    archive = Path(server["data_root"])
    train_rows, oof_rows = fold_rows(plan, cfg, archive, Path(server["protocol_root"]))
    tr, ev = loaders(cfg, archive, train_rows, oof_rows)
    model = build(task["variant"], cfg, pretrained=device.type=='cuda').to(device)
    ema = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
    steps = math.ceil(len(tr) / cfg["accumulation"])
    optimizer, scheduler = optimizer_scheduler(cfg, model, steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=device.type=='cuda')
    tracker = EarlyStopTracker(patience=cfg["patience"])
    resume = run / "last_training.pt"
    history = []
    start_epoch = 1
    training_complete = False
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["task"] != task["id"] or state["identity_sha256"] != identity_hash or state["config_sha256"] != digest(resolved):
            raise ValueError("Resume identity or config differs")
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"]); restore_rng_state(state, tr, ev)
        tracker = EarlyStopTracker(**state["tracker"])
        history, start_epoch = state["history"], state["epoch"] + 1
        training_complete = state["training_complete"]
        for filename, key in (("ema_best.pth", "best_ema"), ("raw_at_ema_best.pth", "best_raw")):
            target = run / filename
            if not target.exists() or digest(target) != state[key + "_sha256"]:
                save_torch(target, state[key])
    with (run / "train.log").open("a", encoding="utf-8") as log:
        def emit(message):
            log.write(message + "\n"); log.flush(); print(task["id"], message, flush=True)
        emit(f"START epoch={start_epoch} steps={steps} micro=4 accum=4 workers=0")
        for epoch in range(start_epoch, 1 if training_complete else cfg["epochs"] + 1):
            model.train(); started = time.perf_counter()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            iterator = iter(tr); total = 0.; skips = 0; updates = 0; consecutive_skips = 0
            for step in range(1, steps + 1):
                micro = list(itertools.islice(iterator, cfg["accumulation"]))
                diagnostic = {}
                try:
                    value, skipped = step_group(cfg, model, ema, micro, optimizer, scheduler,
                                                scaler, device, diagnostic)
                except Exception:
                    if diagnostic:
                        write_json(run / "nonfinite_diagnostic.json",
                                   dict(task=task["id"], epoch=epoch, step=step, **diagnostic))
                    optimizer.zero_grad(set_to_none=True)
                    raise
                total += value; skips += skipped; updates += not skipped
                consecutive_skips = consecutive_skips + 1 if skipped else 0
                if skipped:
                    event = dict(task=task["id"], epoch=epoch, step=step, **diagnostic)
                    with (run / "amp_events.jsonl").open("a", encoding="utf-8") as events:
                        events.write(json.dumps(event, sort_keys=True) + "\n"); events.flush()
                    emit(f"AMP_SKIP {json.dumps(event, sort_keys=True)}")
                if device.type=='cuda' and torch.cuda.max_memory_reserved() / 2**30 > server["max_vram_gib"]:
                    raise RuntimeError("VRAM_BUDGET_EXCEEDED; no automatic batch adjustment")
                if skips >= 20 or consecutive_skips >= 4:
                    write_json(run / "nonfinite_diagnostic.json",
                               dict(task=task["id"], epoch=epoch, step=step,
                                    reason="Excessive AMP skips", epoch_skips=skips,
                                    consecutive_skips=consecutive_skips, **diagnostic))
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Excessive AMP skips")
                if step == 1 or step % 100 == 0 or step == steps:
                    write_json(run / "status.json", dict(state="RUNNING", task=task["id"], epoch=epoch,
                                    step=step, steps=steps, best_epoch=tracker.best_epoch, best_auc=tracker.best_auc))
            auc = infer(ema, ev, study_targets_from_rows(oof_rows), None, device)["macro_auc"]
            decision = tracker.observe(auc, epoch)
            record = dict(epoch=epoch, train_loss=total / steps, oof_ema_macro_auc=auc,
                          best_epoch=tracker.best_epoch, best_auc=tracker.best_auc,
                          early_stopping_stale=tracker.stale, optimizer_updates=updates,
                          ema_updates=int(ema.n_averaged), amp_skips=skips,
                          lr_backbone=optimizer.param_groups[0]["lr"], lr_head=optimizer.param_groups[1]["lr"],
                          seconds=time.perf_counter() - started,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type=='cuda' else 0.,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device.type=='cuda' else 0.)
            history.append(record)
            if decision.new_best_checkpoint:
                save_torch(run / "ema_best.pth", ema.module.state_dict())
                save_torch(run / "raw_at_ema_best.pth", model.state_dict())
            save_torch(resume, dict(task=task["id"], identity_sha256=identity_hash,
                                    config_sha256=digest(resolved), epoch=epoch, model=model.state_dict(),
                                    ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                                    training_complete=decision.should_stop or epoch == cfg["epochs"],
                                    best_ema=torch.load(run / "ema_best.pth", map_location="cpu", weights_only=True),
                                    best_raw=torch.load(run / "raw_at_ema_best.pth", map_location="cpu", weights_only=True),
                                    best_ema_sha256=digest(run / "ema_best.pth"),
                                    best_raw_sha256=digest(run / "raw_at_ema_best.pth"),
                                    tracker=dict(patience=tracker.patience, min_delta=tracker.min_delta,
                                                 best_auc=tracker.best_auc, best_epoch=tracker.best_epoch,
                                                 patience_reference_auc=tracker.patience_reference_auc,
                                                 stale=tracker.stale, last_epoch=tracker.last_epoch),
                                    **capture_rng_state(tr, ev)))
            write_csv(run / "train_history.csv", history)
            emit(f"EPOCH_DONE {json.dumps(record)}")
            if decision.should_stop:
                emit(f"EARLY_STOP epoch={epoch} best={tracker.best_epoch}")
                break
    if not (run / "ema_best.pth").exists():
        raise ValueError("Missing best EMA checkpoint")
    del model, ema, optimizer, scheduler, scaler
    if device.type=='cuda': torch.cuda.empty_cache()
    best = build_model(model_kind(task["variant"]), pretrained=False).to(device)
    best.load_state_dict(torch.load(run / "ema_best.pth", map_location="cpu", weights_only=True))
    oof_metrics = infer(best, ev, study_targets_from_rows(oof_rows), run / "oof_predictions.csv", device)
    val_rows = read_split_rows(archive, "val")
    val_loader = make_dataloader(archive, "val", rows=val_rows, batch_size=4, num_workers=0,
                                  input_size=384, preprocessing="black_pad_v1")
    val_metrics = infer(best, val_loader, study_targets_from_rows(val_rows), run / "val_predictions.csv", device)
    # No Test access: Phase42 structure selection uses only the fixed OOF Fold.
    write_csv(run / "per_class_metrics.csv", [dict(scope="OOF", **r) for r in oof_metrics["per_class"]] +
              [dict(scope="VAL", **r) for r in val_metrics["per_class"]])
    write_json(run / "metrics.json", dict(oof=oof_metrics, val=val_metrics, best_epoch=tracker.best_epoch,
                stop_epoch=tracker.last_epoch, training_seconds=sum(r["seconds"] for r in history),
                peak_reserved_mib=max(r["peak_reserved_mib"] for r in history),
                ema_decay=cfg["ema_decay"], ema_updates=history[-1]["ema_updates"], ema_use_buffers=True))
    from cxr.batch.artifacts import commit
    write_json(run / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU_SYNTHETIC_TEST', code_commit=commit()))
    write_json(run / "run_manifest.json", dict(code_commit=commit(),task=task["id"], fold=task["fold"], variant=task["variant"],
               identity_sha256=identity_hash, config_sha256=digest(resolved),
               fold_sha256=plan["fold_sha256"], pretrained_sha256=plan["pretrained_sha256"],
               best_epoch=tracker.best_epoch, source_model_kind=model_kind(task["variant"]),
               test_labels_used=False, official_val_used_for_training=False))
    write_json(run / "model_size.json", dict(inference_weights_and_config_bytes=
               (run / "ema_best.pth").stat().st_size + resolved.stat().st_size, limit_bytes=500000000,
               latency_status="T4_UNVERIFIED"))
    write_json(run / "status.json", dict(state="POSTPROCESS", task=task["id"], best_epoch=tracker.best_epoch,
               oof_macro_auc=oof_metrics["macro_auc"], val_macro_auc=val_metrics["macro_auc"]))
    from cxr.phase42.benchmark import benchmark
    benchmark(best, val_loader.dataset, run, cfg)
    (run / 'ema_best.sha256').write_text(digest(run / 'ema_best.pth') + '  ema_best.pth\n', encoding='utf-8')
    write_json(run / 'status.json', dict(state='DONE', task=task['id'], best_epoch=tracker.best_epoch,
               oof_macro_auc=oof_metrics['macro_auc'], val_macro_auc=val_metrics['macro_auc']))
    files = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name not in
             ("DONE.json", "last_training.pt")}
    write_json(done, dict(state="DONE", identity_sha256=identity_hash, sha256=files))
    return dict(state="DONE", best_epoch=tracker.best_epoch, oof_auc=oof_metrics["macro_auc"],
                val_auc=val_metrics["macro_auc"])
