"""Isolated, fixed-epoch Fold1 DenseNet auxiliary-supervision comparison."""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.data.dataset import CXRArchiveDataset, preprocess_image, study_targets_from_rows
from cxr.data.patient_folds import select_fold_rows
from cxr.engine.metrics import binary_auc, study_multilabel_metrics
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.models.factory import create_model
from cxr.utils.training_resume import capture_rng_state, restore_rng_state

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("AUX-BASE", "AUX-01")
CODE_FILES = ["src/cxr/phase34_aux.py", "src/cxr/data/dataset.py", "src/cxr/data/archive.py",
              "src/cxr/data/patient_folds.py", "src/cxr/engine/metrics.py", "src/cxr/models/factory.py",
              "src/cxr/utils/training_resume.py", "scripts/run_phase34_aux.py"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def save_torch(path, state):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp); temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AuxiliaryDenseNet(nn.Module):
    """Aux head after denseblock3; main topology/state_dict unchanged."""
    def __init__(self, main, enabled):
        super().__init__()
        self.main = main
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3401)
            self.auxiliary = nn.Linear(1024, 10) if enabled else None

    def forward(self, x):
        auxiliary = None
        for name, stage in self.main.features.named_children():
            x = stage(x)
            if name == "denseblock3" and self.auxiliary is not None:
                auxiliary = self.auxiliary(F.adaptive_avg_pool2d(F.relu(x, inplace=False), (1, 1)).flatten(1))
        main = self.main.classifier(F.adaptive_avg_pool2d(F.relu(x, inplace=True), (1, 1)).flatten(1))
        return main, auxiliary


def combined_loss(main_logits, auxiliary_logits, targets, mask, alpha, denominator):
    if main_logits.shape != targets.shape or mask.shape != targets.shape or targets.shape[1] != 10:
        raise ValueError("Class shape/order mismatch")
    main_sum = (F.binary_cross_entropy_with_logits(main_logits.float(), targets.float(), reduction="none") * mask).sum()
    aux_sum = main_sum.new_zeros(())
    if alpha:
        if auxiliary_logits is None or auxiliary_logits.shape != targets.shape:
            raise ValueError("Missing auxiliary logits")
        aux_sum = (F.binary_cross_entropy_with_logits(auxiliary_logits.float(), targets.float(), reduction="none") * mask).sum()
    if denominator <= 0: raise ValueError("No valid labels")
    loss = (main_sum + alpha * aux_sum) / denominator
    if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite BCE loss")
    return loss, main_sum, aux_sum


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg["architecture"] != "densenet121" or cfg["tasks"] != list(TASKS): raise ValueError("Unexpected paired plan")
    if cfg["epochs"] != 5 or cfg["aux_alpha"] != 0.2 or cfg["fold"]["fold"] != 1: raise ValueError("Frozen protocol differs")
    if cfg["selection"] != "fixed_final_epoch_no_val_early_stopping": raise ValueError("Epoch selection not permitted")
    if cfg["micro_batch"] * cfg["accumulation"] != 16: raise ValueError("Effective batch must be 16")
    if cfg["input_size"] != 384 or cfg["augmentation"] is not None: raise ValueError("B01 preprocessing changed")
    return cfg


def audit(cfg, out):
    weights = ROOT / cfg["pretrained_file"]
    if digest(weights) != cfg["pretrained_sha256"]: raise ValueError("ImageNet weight hash mismatch")
    archive = archive_path({"data": {"archive_path": cfg["archive_path"]}})
    rows = read_split_rows(archive, "train")  # official Val and Test are not opened
    train_rows = select_fold_rows(rows, cfg["fold"], role="train")
    oof_rows = select_fold_rows(rows, cfg["fold"], role="oof")
    train_subjects = {r["Subject_id"] for r in train_rows}; oof_subjects = {r["Subject_id"] for r in oof_rows}
    if train_subjects & oof_subjects: raise ValueError("Patient leakage")
    if {(r["Subject_id"], r["Study_id"]) for r in train_rows} & {(r["Subject_id"], r["Study_id"]) for r in oof_rows}:
        raise ValueError("Study leakage")
    def row_hash(selected):
        return hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    record = {"state": "PASS", "pretrained_sha256": digest(weights), "train_row_sha256": row_hash(train_rows),
              "oof_row_sha256": row_hash(oof_rows), "fold_sha256": cfg["fold"]["sha256"],
              "train": cfg["fold"]["expected_counts"]["train"], "evaluation": cfg["fold"]["expected_counts"]["oof"],
              "initialization": "ImageNet only; no H17/E08/Phase17 finetuned checkpoint used",
              "patient_overlap": 0, "study_overlap": 0, "official_val_used": False, "test_used": False,
              "evaluation_scope": "PARTIAL_OOF: Fold1; fixed final epoch; historically reused Fold1, exploratory",
              "archive": str(archive), "archive_bytes": archive.stat().st_size if archive.is_file() else None,
              "source_code_sha256": {p: digest(ROOT / p) for p in CODE_FILES},
              "config": cfg}
    path = out / "protocol_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Protocol/data/code changed after sealing; refusing continuation")
    else:
        write_json(path, record)
    return archive, train_rows, oof_rows, record


def worker_init(_):
    torch.set_num_threads(1)


def loaders(cfg, archive, train_rows, oof_rows):
    train = CXRArchiveDataset(archive, "train", rows=train_rows, input_size=384, preprocessing="black_pad_v1", augmentation=None)
    oof = CXRArchiveDataset(archive, "train", rows=oof_rows, input_size=384, preprocessing="black_pad_v1", augmentation=None)
    shared = dict(batch_size=cfg["micro_batch"], num_workers=cfg["num_workers"], pin_memory=True,
                  persistent_workers=False, worker_init_fn=worker_init)
    return (DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), **shared),
            DataLoader(oof, shuffle=False, generator=torch.Generator().manual_seed(cfg["seed"] + 1), **shared))


def build(cfg, task, device):
    seed_all(cfg["seed"])
    os.environ["TORCH_HOME"] = str((ROOT / cfg["pretrained_file"]).parents[2])
    model = AuxiliaryDenseNet(create_model("densenet121", pretrained=True), task == "AUX-01").to(device)
    return model


def optimizer_and_schedule(cfg, model, steps):
    head = list(model.main.classifier.parameters())
    if model.auxiliary is not None: head += list(model.auxiliary.parameters())
    ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params":backbone,"lr":cfg["backbone_lr"]},
                                  {"params":head,"lr":cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    warmup = cfg["warmup_epochs"] * steps
    total = cfg["epochs"] * steps
    def factor(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_group(cfg, model, microbatches, optimizer, scheduler, scaler, alpha, device):
    denominator = sum(int(b["valid_mask"].sum()) for b in microbatches)
    optimizer.zero_grad(set_to_none=True)
    values = np.zeros(3)
    for batch in microbatches:
        x = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device); mask = batch["valid_mask"].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            primary, auxiliary = model(x)
            loss, main_sum, aux_sum = combined_loss(primary, auxiliary, target, mask, alpha, denominator)
        scaler.scale(loss).backward()
        values += [float(main_sum.detach()), float(aux_sum.detach()), int(mask.sum())]
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    old = scaler.get_scale(); scaler.step(optimizer); scaler.update()
    skipped = scaler.get_scale() < old
    if not skipped:
        if not bool(torch.isfinite(norm)): raise FloatingPointError("Nonfinite unscaled gradients")
        scheduler.step()
    return values, bool(skipped)


def gradient_probe(model, batch, device):
    model.eval(); model.zero_grad(set_to_none=True)
    with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
        _, aux = model(batch["image"].to(device))
        loss = F.binary_cross_entropy_with_logits(aux.float(), batch["target"].to(device).float())
    loss.backward()
    grad = model.main.features.conv0.weight.grad
    if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.abs().sum() > 0):
        raise ValueError("Aux-only loss does not reach early backbone")
    later = model.main.features.denseblock4.denselayer1.conv1.weight.grad
    if later is not None and bool(later.abs().sum() > 0): raise ValueError("Unexpected downstream auxiliary-only gradient")
    result = {"aux_only_bce":float(loss.detach()), "conv0_grad_l2":float(grad.norm()),
              "finite":True, "auxiliary_reaches_backbone":True, "downstream_stage_aux_only_gradient":False}
    model.zero_grad(set_to_none=True)
    return result


def cpu_checks(cfg, out):
    seed_all(cfg["seed"])
    plain = create_model("densenet121", pretrained=False).eval()
    from copy import deepcopy
    wrapped = AuxiliaryDenseNet(deepcopy(plain), True).eval()
    batch = {"image":torch.randn(2,3,64,64), "target":torch.randint(0,2,(2,10)).float(),
             "valid_mask":torch.ones(2,10,dtype=torch.bool)}
    with torch.no_grad():
        original = plain(batch["image"])
        main, _ = wrapped(batch["image"])
    if not torch.equal(original, main): raise ValueError("Wrapper changed DenseNet main output")
    gradient = gradient_probe(wrapped, batch, torch.device("cpu"))
    x = torch.randn(4,10,requires_grad=True); a=torch.randn(4,10,requires_grad=True)
    target=torch.randint(0,2,(4,10)).float(); mask=torch.ones_like(target,dtype=torch.bool)
    actual,_,_=combined_loss(x,a,target,mask,.2,40)
    expected=F.binary_cross_entropy_with_logits(x,target)+.2*F.binary_cross_entropy_with_logits(a,target)
    if not torch.allclose(actual,expected): raise ValueError("Incorrect auxiliary loss factor")
    partial=sum(combined_loss(x[i:i+2],a[i:i+2],target[i:i+2],mask[i:i+2],.2,40)[0] for i in (0,2))
    if not torch.allclose(actual,partial): raise ValueError("Accumulation differs from full batch")
    receipt={"state":"PASS", "main_output_exact":True,"loss_formula_pass":True,
             "accumulation_pass":True,"auxiliary_gradient":gradient,"cuda_used":False}
    write_json(out / "cpu_tests.json", receipt)
    return receipt


def export_check(model, batch, device, path):
    model.eval()
    save_torch(path, {k:v.detach().cpu() for k,v in model.main.state_dict().items()})
    restored=create_model("densenet121",pretrained=False).to(device).eval()
    state=torch.load(path,map_location="cpu",weights_only=True)
    restored.load_state_dict(state,strict=True)
    if any("auxiliary" in key for key in state): raise ValueError("Auxiliary leaked into inference weight")
    with torch.inference_mode(), torch.autocast(device.type,enabled=device.type=="cuda",dtype=torch.float16):
        original=model(batch["image"].to(device))[0]
        deployed=restored(batch["image"].to(device))
    error=float((original-deployed).abs().max())
    if error != 0: raise ValueError(f"Exported main logits differ: {error}")
    return restored, {"max_abs_logit_error":error,"exact":True,"auxiliary_keys":0,
                      "weight_bytes":path.stat().st_size,"sha256":digest(path)}


def cuda_preflight(cfg, out, archive, train_rows, oof_rows):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    receipt={"state":"RUNNING","device":torch.cuda.get_device_name(),"torch":torch.__version__,"variants":{}}
    initial_hash=None
    for task in TASKS:
        train,holdout=loaders(cfg,archive,train_rows,oof_rows)
        model=build(cfg,task,device)
        initial=tensor_hash(model.main.state_dict())
        if initial_hash is not None and initial != initial_hash: raise ValueError("BASE/AUX main initialization differs")
        initial_hash=initial
        optimizer,scheduler=optimizer_and_schedule(cfg,model,math.ceil(len(train)/cfg["accumulation"]))
        scaler=torch.amp.GradScaler("cuda",init_scale=1024)
        batches=list(itertools.islice(iter(train),cfg["accumulation"]))
        if len(batches)!=cfg["accumulation"]: raise ValueError("Short real-data smoke batch")
        gradient=gradient_probe(model,batches[0],device) if task=="AUX-01" else None
        torch.cuda.reset_peak_memory_stats();model.train()
        values,skipped=train_group(cfg,model,batches,optimizer,scheduler,scaler,.2 if task=="AUX-01" else 0,device)
        if skipped: raise FloatingPointError("Smoke optimizer skipped; formal launch blocked")
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "scaler":scaler.state_dict(),**capture_rng_state(train,holdout)}
        path=out / "preflight" / task
        path.mkdir(parents=True,exist_ok=True)
        save_torch(path / "resume.pt",state)
        train_group(cfg,model,batches,optimizer,scheduler,scaler,.2 if task=="AUX-01" else 0,device)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        recovered=torch.load(path / "resume.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(recovered["model"]);optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"]);scaler.load_state_dict(recovered["scaler"])
        restore_rng_state(recovered,train,holdout)
        train_group(cfg,model,batches,optimizer,scheduler,scaler,.2 if task=="AUX-01" else 0,device)
        resume_error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        if resume_error != 0: raise ValueError(f"Resume continuation not identical: {resume_error}")
        restored,equivalence=export_check(model,batches[0],device,path / "inference.pth")
        peak=torch.cuda.max_memory_reserved()/2**20
        if peak>cfg["max_reserved_mib"]: raise RuntimeError("Preflight exceeds reserved memory cap")
        receipt["variants"][task]={"gradient":gradient,"main_init_hash":initial,"resume_max_abs_error":resume_error,
             "export":equivalence,"peak_reserved_mib":peak,"main_bce":float(values[0]/values[2]),
             "aux_bce":float(values[1]/values[2]),"loss_finite":True,
             "train_image_ids":[image for b in batches for image in b["image_id"]]}
        del model,optimizer,scheduler,scaler,restored,state,recovered,expected,batches,train,holdout
        torch.cuda.empty_cache()
    if receipt["variants"][TASKS[0]]["train_image_ids"]!=receipt["variants"][TASKS[1]]["train_image_ids"]:
        raise ValueError("Data order differs")
    receipt.update(state="PASS",protocol_sha256=digest(out / "protocol_manifest.json"),created_utc=utcnow())
    write_json(out / "cuda_preflight.json",receipt)
    return receipt


def log(run, text):
    line=f"{utcnow()} {text}"
    print(line,flush=True)
    with (run / "train.log").open("a",encoding="utf-8") as f: f.write(line+"\n")


def verify_done(run):
    done=json.loads((run / "DONE.json").read_text(encoding="utf-8"))
    for filename,expected in done["sha256"].items():
        if digest(run / filename)!=expected: raise ValueError(f"Completed file changed: {filename}")
    return done


def evaluate(model, loader, targets, output, *, wrapped):
    model.eval();predictions=[]
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(batch["image"].to("cuda",non_blocking=True))
                if wrapped: logits=logits[0]
                probabilities=torch.sigmoid(logits).float().cpu().tolist()
            for i,p in enumerate(probabilities):
                predictions.append(ImagePrediction(batch["subject_id"][i],batch["study_id"][i],batch["image_id"][i],tuple(p)))
    metrics=study_multilabel_metrics(predictions,targets)
    study=aggregate_mean_probabilities(predictions)
    rows=[{"Subject_id":s.subject_id,"Study_id":s.study_id,**{f"Prob_{j+1}":v for j,v in enumerate(s.probabilities)}} for s in study]
    write_csv(output,rows)
    labels={(t.subject_id,t.study_id):t.labels for t in targets}
    for rec in metrics["per_class"]:
        j=rec["class_id"]-1
        rec["positive_patients"]=len({s.subject_id for s in study if labels[s.subject_id,s.study_id][j]==1})
    return metrics,study


def latency(model, dataset, run, cfg):
    # Read physical JPEGs on every timed iteration, including decoding and CSV write.
    folder=run / "latency_inputs"; folder.mkdir(exist_ok=True)
    import zipfile
    rows=dataset.rows[:25]
    with zipfile.ZipFile(dataset.archive) as z:
        for row in rows:
            path=folder / (row["Dicom_id"]+".jpg")
            if not path.exists(): path.write_bytes(z.read(image_member(row,"train")))
    def predict_one(row):
        with Image.open(folder / (row["Dicom_id"]+".jpg")) as img:
            tensor=preprocess_image(img,384,preprocessing="black_pad_v1").unsqueeze(0)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            p=torch.sigmoid(model(tensor.to("cuda"))).float().cpu().tolist()[0]
        with (run / "latency_last_prediction.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f);writer.writerow(["Subject_id","Study_id"]+[f"Prob_{i}" for i in range(1,11)])
            writer.writerow([row["Subject_id"],row["Study_id"]]+p)
    model.eval()
    for row in rows[:5]: predict_one(row)
    times=[]
    for _ in range(3):
        for row in rows[5:]:
            torch.cuda.synchronize();start=time.perf_counter();predict_one(row);torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    value={"gpu":torch.cuda.get_device_name(),"precision":"FP16 autocast","batch_size":1,
           "mean_ms":float(np.mean(times)),"p50_ms":float(np.percentile(times,50)),
           "p95_ms":float(np.percentile(times,95)),"max_ms":max(times),"warmup_images":5,
           "timed_images_per_round":20,"rounds":3,"scope":"JPEG open/decode, pad/normalize, transfer, forward, CPU probabilities, CSV write+close",
           "source":"first 25 fixed Fold1 images; no Test","status":"T4_UNVERIFIED", "official_timing_pass":False}
    write_json(run / "latency_benchmark.json",value)
    return value


def train_one(cfg,out,archive,train_rows,oof_rows,task):
    run=out/task;run.mkdir(exist_ok=True)
    if (run / "DONE.json").exists(): return verify_done(run)
    alpha=0.2 if task=="AUX-01" else 0.0
    train,holdout=loaders(cfg,archive,train_rows,oof_rows)
    device=torch.device("cuda");model=build(cfg,task,device)
    init_hash=tensor_hash(model.main.state_dict())
    preflight=json.loads((out/"cuda_preflight.json").read_text())
    if init_hash!=preflight["variants"][task]["main_init_hash"]:raise ValueError("Formal initialization differs")
    steps=math.ceil(len(train)/cfg["accumulation"])
    optimizer,scheduler=optimizer_and_schedule(cfg,model,steps)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024)
    protocol_hash=digest(out/"protocol_manifest.json")
    resolved={**cfg,"task":task,"loss_alpha":alpha,"initial_main_state_sha256":init_hash,"protocol_sha256":protocol_hash}
    cfg_path=run/"resolved_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8"))!=resolved:raise ValueError("Runtime configuration differs")
    write_json(cfg_path,resolved)
    start_epoch=1;history=[]
    checkpoint=run/"last_training.pt"
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state["protocol_sha256"]!=protocol_hash or state["task"]!=task:raise ValueError("Resume identity differs")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        restore_rng_state(state,train,holdout)
        history=state["history"];start_epoch=state["epoch"]+1
        log(run,f"RESUME next_epoch={start_epoch}; no completed epochs repeated")
    log(run,f"START {task} epochs={cfg['epochs']} train_images={len(train_rows)} OOF_images={len(oof_rows)}")
    for epoch in range(start_epoch,cfg["epochs"]+1):
        model.train(); torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        total=np.zeros(3);skips=0;updates=0;consecutive_skips=0;data_order=hashlib.sha256()
        iterator=iter(train)
        for step in range(1,steps+1):
            micro=list(itertools.islice(iterator,cfg["accumulation"]))
            for b in micro:
                for image in b["image_id"]:data_order.update((image+"\n").encode())
            values,skipped=train_group(cfg,model,micro,optimizer,scheduler,scaler,alpha,device)
            total+=values;skips+=skipped;updates+=not skipped
            consecutive_skips=consecutive_skips+1 if skipped else 0
            if consecutive_skips>=20:raise FloatingPointError("20 consecutive AMP skips")
            if step==1 or step%50==0 or step==steps:
                current={"state":"RUNNING","task":task,"epoch":epoch,"step":step,"steps_per_epoch":steps,
                         "train_main_bce":float(total[0]/total[2]),"train_aux_bce":float(total[1]/total[2]),
                         "seconds":time.perf_counter()-started,"updated_utc":utcnow()}
                write_json(run/"status.json",current)
                log(run,f"epoch={epoch} step={step}/{steps} main={current['train_main_bce']:.5f} aux={current['train_aux_bce']:.5f} elapsed={current['seconds']:.1f}s")
        record={"epoch":epoch,"main_bce":float(total[0]/total[2]),"aux_bce":float(total[1]/total[2]),
                "total_loss":float((total[0]+alpha*total[1])/total[2]),"seconds":time.perf_counter()-started,
                "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,
                "optimizer_updates":int(updates),"amp_skips":int(skips),"data_order_sha256":data_order.hexdigest(),
                "backbone_lr":optimizer.param_groups[0]["lr"],"head_lr":optimizer.param_groups[1]["lr"]}
        history.append(record)
        state={"task":task,"protocol_sha256":protocol_hash,"epoch":epoch,"model":model.state_dict(),
               "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
               "history":history,**capture_rng_state(train,holdout)}
        save_torch(checkpoint,state)
        save_torch(run/f"epoch_{epoch:02d}.pt",{"model":model.state_dict(),"epoch":epoch,"task":task,"protocol_sha256":protocol_hash})
        write_csv(run/"train_history.csv",history)
        log(run,f"EPOCH_DONE {record}")
    write_json(run/"status.json",{"state":"EVALUATING","task":task,"epoch":cfg["epochs"]})
    targets=study_targets_from_rows(oof_rows,unlisted_policy="negative")
    metrics,study=evaluate(model,holdout,targets,run/"oof_predictions.csv",wrapped=True)
    probe=next(iter(holdout))
    restored,equivalence=export_check(model,probe,device,run/"inference_model.pth")
    exported_metrics,exported=evaluate(restored,holdout,targets,run/"exported_oof_predictions.csv",wrapped=False)
    if [(p.subject_id,p.study_id) for p in study]!=[(p.subject_id,p.study_id) for p in exported]:raise ValueError("Export OOF IDs differ")
    max_diff=max(abs(a-b) for p,q in zip(study,exported) for a,b in zip(p.probabilities,q.probabilities))
    if max_diff!=0 or metrics["macro_auc"]!=exported_metrics["macro_auc"]:raise ValueError("Complete OOF export equivalence failed")
    equivalence.update(full_oof_studies=len(study),full_oof_max_probability_error=max_diff)
    write_json(run/"export_equivalence.json",equivalence)
    write_csv(run/"per_class_metrics.csv",metrics["per_class"])
    metrics.update(task=task,selected_epoch=cfg["epochs"],selection=cfg["selection"],oof_scope="PARTIAL_OOF: Fold1 only",
                   patients=len({r["Subject_id"] for r in oof_rows}),validation_role="EXPLORATORY",
                   training_seconds=sum(r["seconds"] for r in history),gpu_peak_mib=max(r["peak_allocated_mib"] for r in history))
    write_json(run/"metrics.json",metrics)
    latency(restored,holdout.dataset,run,cfg)
    write_json(run/"model_size.json",{"bytes":equivalence["weight_bytes"],"size_status":"PASS" if equivalence["weight_bytes"]<=500000000 else "SIZE_FAIL",
                                       "latency_status":"T4_UNVERIFIED","auxiliary_in_inference":False})
    (run/"inference_model.sha256").write_text(equivalence["sha256"]+"  inference_model.pth\n",encoding="ascii")
    write_json(run/"run_manifest.json",{"task":task,"protocol_sha256":protocol_hash,"source_pretrained_sha256":cfg["pretrained_sha256"],
                "initial_main_state_sha256":init_hash,"fold_hashes":cfg["fold"]["sha256"],"completed_utc":utcnow(),
                "official_val_accessed":False,"test_accessed":False,"selected_epoch":cfg["epochs"],"resume_checkpoint":"last_training.pt"})
    write_json(run/"status.json",{"state":"DONE","task":task,"epoch":cfg["epochs"],"fold1_macro_auc":metrics["macro_auc"]})
    hashes={p.relative_to(run).as_posix():digest(p) for p in run.rglob("*") if p.is_file() and p.name!="DONE.json"}
    write_json(run/"DONE.json",{"state":"DONE","sha256":hashes})
    del model,optimizer,scheduler,scaler,restored,train,holdout
    torch.cuda.empty_cache()
    return verify_done(run)


def analyze_pair(cfg,out,oof_rows):
    for task in TASKS:verify_done(out/task)
    def read(path):
        with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
    tables=[]
    for task in TASKS:
        rows=read(out/task/"oof_predictions.csv")
        table={(r["Subject_id"],r["Study_id"]):[float(r[f"Prob_{j}"]) for j in range(1,11)] for r in rows}
        if len(table)!=len(rows):raise ValueError("Duplicate OOF Study")
        tables.append(table)
    targets=study_targets_from_rows(oof_rows,unlisted_policy="negative")
    truth={(t.subject_id,t.study_id):t.labels for t in targets};keys=sorted(truth)
    if any(set(t)!=set(truth) for t in tables):raise ValueError("Paired OOF coverage mismatch")
    y=np.asarray([truth[k] for k in keys],dtype=np.int8)
    base,aux=[np.array([t[k] for k in keys]) for t in tables]
    aucs=[[binary_auc(y[:,j],p[:,j]) for j in range(10)] for p in (base,aux)]
    per_class=[{"class_id":j+1,"base_auc":aucs[0][j],"aux_auc":aucs[1][j],"delta":aucs[1][j]-aucs[0][j]} for j in range(10)]
    write_csv(out/"paired_per_class.csv",per_class)
    patients=sorted({k[0] for k in keys});index={s:[] for s in patients}
    for i,k in enumerate(keys):index[k[0]].append(i)
    rng=np.random.default_rng(cfg["bootstrap_seed"]);draws=[];invalid=0
    while len(draws)<cfg["bootstrap_draws"]:
        selected=rng.choice(patients,len(patients),replace=True)
        ix=np.concatenate([index[s] for s in selected])
        if np.any(y[ix].sum(0)==0) or np.any(y[ix].sum(0)==len(ix)):
            invalid+=1
            if invalid>10000:raise ValueError("Too many invalid bootstrap draws")
            continue
        draws.append(float(np.mean([binary_auc(y[ix,j],aux[ix,j])-binary_auc(y[ix,j],base[ix,j]) for j in range(10)])))
    histories=[read(out/task/"train_history.csv") for task in TASKS]
    if any(a["data_order_sha256"]!=b["data_order_sha256"] for a,b in zip(*histories)):raise ValueError("Training image orders differ")
    if any(len(h)!=cfg["epochs"] for h in histories):raise ValueError("Fixed epoch budget not completed")
    base_auc=float(np.mean(aucs[0]));aux_auc=float(np.mean(aucs[1]));ci=np.quantile(draws,[.025,.975]).tolist()
    eligible=aux_auc-base_auc>=cfg["continue_delta"] and ci[0]>0 and min(r["delta"] for r in per_class)>=-cfg["max_class_drop"]
    report={"state":"COMPLETE","scope":"EXPLORATORY_PARTIAL_OOF","studies":len(keys),"patients":len(patients),
            "base_auc":base_auc,"aux_auc":aux_auc,"delta":aux_auc-base_auc,"ci95":ci,
            "bootstrap":{"seed":cfg["bootstrap_seed"],"valid":len(draws),"invalid":invalid,"unit":"patient cluster",
                         "invalid_rule":"discard draw if any class lacks positives or negatives"},
            "same_training_order":True,"fixed_epoch":cfg["epochs"],"official_val_used":False,
            "continuation_evidence_rule_met":eligible,"automatic_expansion":False,
            "limitation":"Fold1 has historical reuse; this single-seed pair is exploratory, not complete fivefold evidence."}
    write_json(out/"paired_bootstrap.json",report)
    write_csv(out/"experiment_summary.csv",[{"task":task,"fold1_macro_auc":auc,"selected_epoch":cfg["epochs"],
               "studies":len(keys),"patients":len(patients),"scope":"PARTIAL_OOF",
               "weight_bytes":json.loads((out/task/"model_size.json").read_text())["bytes"],
               "latency_status":"T4_UNVERIFIED"} for task,auc in zip(TASKS,(base_auc,aux_auc))])
    all_loss=[float(r["main_bce"]) for h in histories for r in h]
    lo=min(all_loss);hi=max(all_loss);span=max(hi-lo,1e-8)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="420" viewBox="0 0 700 420">',
         '<rect width="700" height="420" fill="white"/>',
         '<path d="M70 30 V350 H650" stroke="black" fill="none"/>',
         '<text x="290" y="400">Epoch</text><text x="70" y="20">Main training BCE</text>']
    for task,h,color in zip(TASKS,histories,("#1463b8","#c34c18")):
        points=" ".join(f"{70+580*(int(r['epoch'])-1)/max(1,cfg['epochs']-1):.2f},{340-300*(float(r['main_bce'])-lo)/span:.2f}" for r in h)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="480" y="{45 if task==TASKS[0] else 65}" fill="{color}">{task}</text>')
    svg.append(f'<text x="10" y="45">{hi:.4f}</text><text x="10" y="340">{lo:.4f}</text>')
    for epoch in range(1,cfg['epochs']+1):svg.append(f'<text x="{70+580*(epoch-1)/(cfg["epochs"]-1)}" y="373">{epoch}</text>')
    svg.append('</svg>');(out/"training_curve.svg").write_text("\n".join(svg),encoding="utf-8")
    text=(f"# Phase34 paired Pilot\n\nBASE Fold1 AUC {base_auc:.6f}; AUX {aux_auc:.6f}; delta {aux_auc-base_auc:+.6f}; "
          f"patient bootstrap 95% CI {ci}.\n\nCoverage: {len(keys)} Studies / {len(patients)} patients. "
          f"Both fixed at {cfg['epochs']} epochs; identical main initialization and image order. No official Val/Test accessed.\n\n"
          "Auxiliary-only gradient reaches conv0; pure DenseNet export contains no auxiliary head, complete Fold1 probabilities are exactly equal before/after export. "
          "Local GPU timing is reference only; T4_UNVERIFIED.\n\n"
          f"Predeclared followup evidence threshold met: {eligible}. Further folds require user decision; not auto-started. "
          "Single Fold1 is historically reused and does not establish a new independent generalization claim.\n")
    (out/"final_decision.md").write_text(text,encoding="utf-8")
    write_json(out/"pair_status.json",report)


def process_alive(pid):
    if os.name=="nt":
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()!=87
        code=ctypes.c_ulong()
        ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
        return not ok or code.value==259
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


@contextlib.contextmanager
def lock(out):
    path=out/"pipeline.lock"
    if path.exists():
        record=json.loads(path.read_text())
        if process_alive(record["pid"]):raise RuntimeError("Phase34 pipeline already active")
        path.unlink()
    with path.open("x") as f:json.dump({"pid":os.getpid(),"utc":utcnow()},f)
    try:yield
    finally:path.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--action",choices=["audit","preflight","run","status","analyze"],default="run")
    args=p.parse_args();cfg=load_config(args.config)
    torch.set_num_threads(cfg["cpu_threads"])
    out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True)
    if args.action=="status":
        for task in TASKS:
            path=out/task/"status.json";print(task,path.read_text() if path.exists() else "PENDING")
        return
    archive,train_rows,oof_rows,protocol=audit(cfg,out)
    if args.action=="audit":print(json.dumps(protocol,ensure_ascii=False));return
    with lock(out):
        if args.action=="analyze":analyze_pair(cfg,out,oof_rows);return
        cpu_checks(cfg,out)
        receipt_path=out/"cuda_preflight.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt["state"]!="PASS" or receipt["protocol_sha256"]!=digest(out/"protocol_manifest.json"):
                raise ValueError("Invalid preflight receipt")
        else:
            cuda_preflight(cfg,out,archive,train_rows,oof_rows)
        if args.action=="preflight":print("CPU and CUDA preflight PASS; no formal run started",flush=True);return
        for task in TASKS:
            try:train_one(cfg,out,archive,train_rows,oof_rows,task)
            except Exception as exc:
                write_json(out/task/"FAILED.json",{"state":"FAILED","error":f"{type(exc).__name__}: {exc}",
                            "resume_available":(out/task/"last_training.pt").exists(),"utc":utcnow()})
                write_json(out/"pair_status.json",{"state":"INCOMPLETE","failed_task":task,"automatic_restart":False})
                raise
        analyze_pair(cfg,out,oof_rows)


if __name__=="__main__":main()
