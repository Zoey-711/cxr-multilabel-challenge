"""Isolated H50-BASE Fold1 from ImageNet; real-data smoke, resume and fixed SWA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.data.patient_folds import select_fold_rows
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.models.factory import create_model
from cxr.utils.training_resume import capture_rng_state, restore_rng_state
from cxr.phase39_run import infer
from cxr.phase34_aux import save_torch, write_csv, write_json, digest

OUT = ROOT / "outputs/experiments/phase50_local_base"
RUN = OUT / "H50_BASE_F1"
CONFIG = OUT / "config.yaml"
WEIGHT = ROOT / ".wheels/torch_cache/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth"
FOLDS = ROOT / "outputs/analysis/phase17_patient_folds"
ARCHIVE = Path("F:/\u6570\u636e\u96c6/cxr_first.zip")


def setup():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg["micro_batch"] * cfg["accumulation"] != 16 or cfg["epochs"] != 4 or cfg["train_folds"] != [2,3,4,5]:
        raise ValueError("H50 fixed protocol changed")
    if digest(WEIGHT) != cfg["pretrained_sha256"] or any(digest(FOLDS/k) != v for k,v in cfg["fold_sha256"].items()):
        raise ValueError("Pretrained or fold SHA mismatch")
    rows = read_split_rows(ARCHIVE, "train")
    subset = dict(kind="patient_fold_v1", directory=str(FOLDS), fold=1, sha256=cfg["fold_sha256"],
                  expected_counts=dict(train=dict(patients=8844, studies=14826, images=24332),
                                       oof=dict(patients=2211, studies=3687, images=6012)))
    train = select_fold_rows(rows, subset, role="train")
    heldout = select_fold_rows(rows, subset, role="oof")
    if set(r["Subject_id"] for r in train) & set(r["Subject_id"] for r in heldout):
        raise ValueError("Patient leakage")
    return cfg, train, heldout


def loaders(cfg, train, heldout):
    aug = dict(enabled=True, name="conservative_cxr_v1", affine_probability=.5,
               rotation_degrees=3., translation_fraction=.02, photometric_probability=.5,
               brightness_contrast_fraction=.05)
    args = dict(batch_size=cfg["micro_batch"], num_workers=0, preprocessing=cfg["preprocessing"],
                input_size=cfg["input_size"], persistent_workers=False)
    tr = make_dataloader(ARCHIVE, "train", rows=train, shuffle=True, seed=42, augmentation=aug, **args)
    ev = make_dataloader(ARCHIVE, "train", rows=heldout, shuffle=False, seed=43, **args)
    return tr, ev


def model_new(pretrained):
    os.environ["TORCH_HOME"] = str(ROOT / ".wheels/torch_cache")
    return create_model("efficientnet_v2_s", num_classes=10, pretrained=pretrained)


def optimizer_scheduler(cfg, model, steps):
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    back = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([dict(params=back, lr=cfg["backbone_lr"]),
                             dict(params=head, lr=cfg["head_lr"])], weight_decay=cfg["weight_decay"])
    warm = cfg["warmup_epochs"]*steps
    total = cfg["epochs"]*steps
    def factor(step):
        return (step+1)/warm if step < warm else .5*(1+math.cos(math.pi*min(1., (step-warm)/max(1,total-warm))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def update_group(micro, model, ema, optimizer, scheduler, scaler, cfg):
    valid = sum(int(b["valid_mask"].sum()) for b in micro)
    if valid <= 0:
        raise ValueError("No valid labels")
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for batch in micro:
        images = batch["image"].to("cuda")
        target = batch["target"].to("cuda")
        mask = batch["valid_mask"].to("cuda")
        if not torch.isfinite(images).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Nonfinite input/label")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite logits")
            loss = (F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none") * mask).sum()/valid
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        scaler.scale(loss).backward()
        total += float(loss.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=False)
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before
    if not skipped:
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient without GradScaler skip")
        scheduler.step()
        ema.update_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    return total, bool(skipped), float(norm) if torch.isfinite(norm) else None


def make_state(cfg, epoch, model, ema, opt, sched, scaler, tracker, history, train_loader, val_loader):
    return dict(config_sha256=digest(CONFIG), epoch=epoch, model=model.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                tracker=dict(patience=tracker.patience, min_delta=tracker.min_delta, best_auc=tracker.best_auc,
                             best_epoch=tracker.best_epoch, patience_reference_auc=tracker.patience_reference_auc,
                             stale=tracker.stale, last_epoch=tracker.last_epoch), history=history,
                **capture_rng_state(train_loader, val_loader))


def smoke(cfg, train, heldout):
    path = OUT / "smoke"
    if path.exists():
        raise FileExistsError("Smoke directory already exists; inspect prior run")
    path.mkdir(parents=True)
    torch.manual_seed(cfg["seed"])
    tr, ev = loaders(cfg, train, heldout)
    model = model_new(True).cuda().train()
    ema = AveragedModel(model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)
    opt, sched = optimizer_scheduler(cfg, model, math.ceil(len(tr)/cfg["accumulation"]))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    micro = list(itertools.islice(iter(tr), cfg["accumulation"]))
    loss, skip, norm = update_group(micro, model, ema, opt, sched, scaler, cfg)
    if skip:
        raise FloatingPointError("Real-image smoke skipped optimizer update")
    state = make_state(cfg, 0, model, ema, opt, sched, scaler, EarlyStopTracker(patience=6), [], tr, ev)
    save_torch(path/"resume.pt", state)
    recovered = torch.load(path/"resume.pt", map_location="cpu", weights_only=True)
    second = model_new(False).cuda().eval()
    second.load_state_dict(recovered["model"])
    first = model.eval()
    sample = next(iter(ev))["image"].to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        delta = float((first(sample)-second(sample)).abs().max())
    if delta != 0 or recovered["config_sha256"] != digest(CONFIG):
        raise ValueError("Checkpoint restore mismatch")
    report = dict(state="PASS", micro_batch=cfg["micro_batch"], accumulation=cfg["accumulation"],
                  actual_images=sum(len(b["image"]) for b in micro), loss=loss, gradient_norm=norm,
                  optimizer_updates=1, amp_skips=0, ema_updates=int(ema.n_averaged),
                  checkpoint_restore_max_abs_error=delta, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  seconds=time.perf_counter()-start, checkpoint_sha256=digest(path/"resume.pt"))
    write_json(path/"report.json", report)
    return report


def train(cfg, train_rows, heldout):
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/"DONE.json").exists():
        raise FileExistsError("Completed H50 BASE is frozen")
    if (RUN/"resolved_config.yaml").exists():
        if digest(RUN/"resolved_config.yaml") != digest(CONFIG):
            raise ValueError("Resume config differs")
    else:
        (RUN/"resolved_config.yaml").write_bytes(CONFIG.read_bytes())
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    tr, ev = loaders(cfg, train_rows, heldout)
    model = model_new(True).cuda()
    ema = AveragedModel(model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)
    steps = math.ceil(len(tr)/cfg["accumulation"])
    opt, sched = optimizer_scheduler(cfg, model, steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    tracker = EarlyStopTracker(patience=cfg["early_stopping_patience"])
    history=[]; first_epoch=1
    if (RUN/"last_training.pt").exists():
        state=torch.load(RUN/"last_training.pt", map_location="cpu", weights_only=True)
        if state["config_sha256"] != digest(CONFIG):
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"]); restore_rng_state(state,tr,ev)
        tracker=EarlyStopTracker(**state["tracker"]); history=state["history"]
        first_epoch=state["epoch"]+1
        if len(history)!=state["epoch"]:
            raise ValueError("Resume history mismatch")
    targets=study_targets_from_rows(heldout)
    with (RUN/"train.log").open("a", encoding="utf-8") as log:
        def emit(value):
            print(value,flush=True); log.write(value+"\n"); log.flush()
        emit(f"START epoch={first_epoch} micro={cfg['micro_batch']} accum={cfg['accumulation']} steps={steps}")
        for epoch in range(first_epoch,cfg["epochs"]+1):
            model.train(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter(); iterator=iter(tr)
            total=0.; updates=0; skips=0; consecutive=0
            for step in range(1,steps+1):
                micro=list(itertools.islice(iterator,cfg["accumulation"]))
                if not micro: raise ValueError("DataLoader ended early")
                loss,skip,norm=update_group(micro,model,ema,opt,sched,scaler,cfg)
                total+=loss; skips+=skip; updates+=not skip
                consecutive=consecutive+1 if skip else 0
                if skip:
                    with (RUN/"amp_events.jsonl").open("a",encoding="utf-8") as events:
                        events.write(json.dumps(dict(epoch=epoch,step=step,scaler=scaler.get_scale(),norm=norm,optimizer_skipped=True))+"\n")
                if consecutive>=cfg["max_consecutive_amp_skips"]:
                    raise FloatingPointError("Persistent AMP overflow")
                if step==1 or step%100==0 or step==steps:
                    write_json(RUN/"status.json",dict(state="RUNNING",epoch=epoch,step=step,steps=steps,
                                                         best_epoch=tracker.best_epoch,best_auc=tracker.best_auc,
                                                         amp_skips=skips,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
                    emit(f"epoch={epoch} step={step}/{steps} updates={updates} skips={skips}")
            auc=infer(ema,ev,targets,None,torch.device("cuda"))["macro_auc"]
            decision=tracker.observe(auc,epoch)
            record=dict(epoch=epoch,train_loss=total/steps,fold1_ema_macro_auc=auc,
                        optimizer_updates=updates,ema_updates=int(ema.n_averaged),amp_skips=skips,
                        best_epoch=tracker.best_epoch,best_auc=tracker.best_auc,
                        seconds=time.perf_counter()-start,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            history.append(record)
            if decision.new_best_checkpoint:
                save_torch(RUN/"ema_best.pth",ema.module.state_dict())
            if epoch in cfg["swa_raw_epochs"]:
                save_torch(RUN/f"raw_epoch_{epoch:02d}.pth",model.state_dict())
            save_torch(RUN/"last_training.pt",make_state(cfg,epoch,model,ema,opt,sched,scaler,tracker,history,tr,ev))
            write_csv(RUN/"train_history.csv",history)
            emit("EPOCH_DONE "+json.dumps(record))
            if decision.should_stop:break
    best=model_new(False).cuda().eval()
    best.load_state_dict(torch.load(RUN/"ema_best.pth",map_location="cpu",weights_only=True))
    metrics=infer(best,ev,targets,RUN/"oof_predictions.csv",torch.device("cuda"))
    write_csv(RUN/"per_class_auc.csv",metrics["per_class"])
    metrics.update(best_epoch=tracker.best_epoch,stop_epoch=tracker.last_epoch,scope="FOLD1_OOF",
                   training_seconds=sum(r["seconds"] for r in history),peak_allocated_mib=max(r["peak_allocated_mib"] for r in history))
    write_json(RUN/"metrics.json",metrics)
    # Official Val is exported only after all checkpoint selection is complete.
    valrows=read_split_rows(ARCHIVE,"val")
    val_loader=make_dataloader(ARCHIVE,"val",rows=valrows,shuffle=False,seed=44,
        batch_size=cfg["micro_batch"],num_workers=0,preprocessing=cfg["preprocessing"],input_size=cfg["input_size"])
    valmetrics=infer(best,val_loader,study_targets_from_rows(valrows),RUN/"val_predictions.csv",torch.device("cuda"))
    write_json(RUN/"val_metrics.json",dict(val_macro_auc=valmetrics["macro_auc"],role="post_selection_reference"))
    write_json(RUN/"run_manifest.json",dict(config_sha256=digest(CONFIG),pretrained_sha256=cfg["pretrained_sha256"],
        fold_sha256=cfg["fold_sha256"],train_patients=8844,fold1_patients=2211,teacher_used=False,
        frozen_h17_weight_used=False,official_val_used_for_selection=False,test_used=False))
    write_json(RUN/"status.json",dict(state="DONE",best_epoch=tracker.best_epoch,oof_auc=metrics["macro_auc"]))
    hashes={p.name:digest(p) for p in RUN.iterdir() if p.is_file() and p.name!="DONE.json"}
    write_json(RUN/"DONE.json",dict(state="DONE",sha256=hashes))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--action",choices=["smoke","train"],required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    cfg,tr,ev=setup()
    if args.action=="smoke": print(json.dumps(smoke(cfg,tr,ev))); return
    smoke_report=OUT/"smoke/report.json"
    if not smoke_report.is_file() or json.loads(smoke_report.read_text())["state"]!="PASS":
        raise ValueError("Real-data smoke gate has not passed")
    train(cfg,tr,ev)


if __name__=="__main__":main()
