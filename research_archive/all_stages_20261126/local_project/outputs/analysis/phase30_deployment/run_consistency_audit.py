"""Read-only Phase30 batch/precision diagnostic on official Val; no Test access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P29=ROOT/"outputs/analysis/phase29_deployment"
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(P29/"deployment"))
from cxr.data.archive import image_member, parse_positive_classes, read_split_rows
from cxr.data.dataset import CXRArchiveDataset, preprocess_image
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from predict import load_models, NAMES

ARCHIVE=Path("F:/数据集/cxr_first.zip")
FRIEND=Path("E:/计挑赛/cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/experiments/H17_DISTILL_02")
E08=ROOT/"outputs/submissions/E08_EMA_20261020"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def read_wide(path):
    with path.open(encoding="utf-8-sig",newline="") as f: records=list(csv.DictReader(f))
    prefix="probability_" if "probability_1" in records[0] else "Prob_"
    result={(r["Subject_id"],r["Study_id"]):np.array([float(r[f"{prefix}{i}"]) for i in range(1,11)]) for r in records}
    if len(result)!=859 or len(records)!=859:raise ValueError(f"Study coverage mismatch: {path}")
    return result

def aggregate(image_keys, image_prob):
    grouped=defaultdict(list)
    for key,prob in zip(image_keys,image_prob):grouped[key].append(prob.astype(np.float64))
    return {k:np.mean(v,axis=0) for k,v in grouped.items()}

def fuse(parts):
    e01=(parts["B01"]+parts["B02"]+parts["B03"])/3
    return .5*(.5*e01+.5*parts["M02"])+.5*parts["N01_EMA"]

def inspect_preprocessing(rows,dataset,all_images=False):
    sampled=range(len(rows)) if all_images else np.linspace(0,len(rows)-1,105,dtype=int)
    diffs=[]; identical=0; single_vs_double=[]; single_identical=0
    with zipfile.ZipFile(ARCHIVE) as zf:
        for index in sampled:
            r=rows[int(index)]
            with Image.open(io.BytesIO(zf.read(image_member(r,"val")))) as im:
                fresh=preprocess_image(im.convert("RGB"),384,preprocessing="black_pad_v1")
                single=preprocess_image(im,384,preprocessing="black_pad_v1")
            old=dataset[int(index)]["image"]
            diff=float((fresh-old).abs().max())
            diffs.append(diff);identical+=int(torch.equal(fresh,old))
            single_vs_double.append(float((fresh-single).abs().max()))
            single_identical+=int(torch.equal(fresh,single))
    return {"sampled_images":len(sampled),"input_tensor_bit_identical":identical,
            "max_abs_tensor_diff":max(diffs),"single_vs_double_rgb_identical":single_identical,
            "single_vs_double_rgb_max_abs_tensor_diff":max(single_vs_double),
            "rgb":"PIL convert RGB", "resize":"PIL ImageOps.pad bilinear 384 black",
            "normalize":"torchvision ToTensor then ImageNet mean/std", "class_order":"1..10"}

def forward(model,tensor,precision):
    with torch.inference_mode():
        if precision=="fp16":
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(tensor)
                probability=torch.sigmoid(logits)
        else:
            logits=model(tensor)
            probability=torch.sigmoid(logits)
    return logits.float().cpu().numpy(),probability.float().cpu().numpy()

def factor_sample(dataset,models):
    # Same tensor x in a realistic 16-image batch and as a singleton.
    first=torch.stack([dataset[i]["image"] for i in range(16)]).cuda()
    x=first[:1].clone()
    if not torch.equal(x,first[:1]):raise ValueError("Input tensor unexpectedly changed")
    result={}
    for name,model in models.items():
        modes={}
        for precision in ("fp16","fp32"):
            batch_logits,batch_prob=forward(model,first,precision)
            single_logits,single_prob=forward(model,x,precision)
            batch=batch_logits[0];single=single_logits[0]
            modes[precision]={"same_input":True,"max_abs_logit_batch16_vs_batch1":float(np.max(np.abs(batch-single))),
                              "max_abs_probability_batch16_vs_batch1":float(np.max(np.abs(batch_prob[0]-single_prob[0]))) }
            modes[precision]["batch16_logits"] = batch.tolist()
            modes[precision]["batch1_logits"] = single.tolist()
        modes["fp16_vs_fp32_single_max_abs_logit"] = float(np.max(np.abs(
            np.array(modes["fp16"]["batch1_logits"])-np.array(modes["fp32"]["batch1_logits"]))))
        result[name]=modes
    return result

def full_val(rows,dataset,models,precision,batch_size):
    loader=DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=False)
    image_keys=[]; outputs={name:[] for name in models}
    for batch in loader:
        x=batch["image"].cuda()
        image_keys.extend(zip(batch["subject_id"],batch["study_id"]))
        for name,model in models.items():
            _,p=forward(model,x,precision)
            outputs[name].append(p)
    result={name:aggregate(image_keys,np.concatenate(chunks)) for name,chunks in outputs.items()}
    return result,len(image_keys)

def summarize(new,reference,labels):
    keys=sorted(reference)
    if set(new)!=set(reference) or len(keys)!=859:raise ValueError("Val Study IDs not aligned")
    p=np.stack([new[k] for k in keys]);q=np.stack([reference[k] for k in keys]);y=np.stack([labels[k] for k in keys])
    diff=np.abs(p-q)
    classes=[]
    for c in range(10):
        order=np.argsort(diff[:,c])[::-1]
        classes.append({"class_id":c+1,"max_abs_diff":float(diff[:,c].max()),
            "mean_abs_diff":float(diff[:,c].mean()),"p99_abs_diff":float(np.percentile(diff[:,c],99)),
            "studies_above_1e-4":int(np.sum(diff[:,c]>1e-4)),
            "auc_new":binary_auc(y[:,c],p[:,c]),"auc_frozen":binary_auc(y[:,c],q[:,c]),
            "worst_subject":keys[int(order[0])][0],"worst_study":keys[int(order[0])][1],
            "worst_new_prob":float(p[order[0],c]),"worst_frozen_prob":float(q[order[0],c])})
    top=np.unravel_index(np.argmax(diff),diff.shape)
    return {"studies":len(keys),"max_abs_diff":float(diff.max()),
        "mean_abs_diff":float(diff.mean()),"p99_abs_diff":float(np.percentile(diff,99)),
        "macro_auc_new":float(np.mean([x["auc_new"] for x in classes])),
        "macro_auc_frozen":float(np.mean([x["auc_frozen"] for x in classes])),
        "max_diff_study":keys[int(top[0])],"max_diff_class":int(top[1])+1,"per_class":classes}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["sample","full_preprocessing","full_batch1","full_baseline","full_fp32"])
    parser.add_argument("--candidate",choices=["E08_EMA","H17_02"],required=True)
    args=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    rows=read_split_rows(ARCHIVE,"val")
    dataset=CXRArchiveDataset(ARCHIVE,"val",rows=rows,input_size=384,preprocessing="black_pad_v1")
    models={} if args.mode=="full_preprocessing" else load_models(args.candidate)
    if not all(not m.training for m in models.values()):raise ValueError("Model not in eval mode")
    if args.mode=="full_preprocessing":
        result={"candidate":args.candidate,"preprocessing":inspect_preprocessing(rows,dataset,all_images=True)}
    elif args.mode=="sample":
        result={"candidate":args.candidate,"preprocessing":inspect_preprocessing(rows,dataset),
                "precision_batch_factors":factor_sample(dataset,models),"model_eval":True,
                "torch_grad_enabled_during_forward":False}
    else:
        precision="fp32" if args.mode=="full_fp32" else "fp16"
        batch_size=1 if args.mode=="full_batch1" else 16
        # E08 baseline preserves frozen per-model microbatch 8 for M02/N01, proven exactly by Phase20.
        if args.candidate=="E08_EMA" and args.mode=="full_baseline":
            raise ValueError("Use frozen Phase20 exact-equivalence evidence for mixed 16/8 baseline")
        predictions,image_count=full_val(rows,dataset,models,precision,batch_size)
        labels={}
        for item in rows:
            key=(item["Subject_id"],item["Study_id"])
            positive=parse_positive_classes(item["Predict_class"])
            value=np.array([int(c in positive) for c in range(1,11)])
            if key in labels and not np.array_equal(labels[key],value):raise ValueError("Inconsistent label")
            labels[key]=value
        if args.candidate=="E08_EMA":
            keys=sorted(predictions[NAMES[0]])
            combined={k:fuse({name:predictions[name][k] for name in NAMES}) for k in keys}
            ref=read_wide(E08/"val_predictions.csv")
            per_model={}
            manifest=json.loads((E08/"manifest.json").read_text(encoding="utf-8"))
            for name in NAMES:
                meta=manifest["models"][name]
                path=ROOT/"outputs/experiments"/meta["run_id"]/(
                    "ema_validation_analysis/study_predictions.csv" if name=="N01_EMA" else
                    "validation_analysis/study_predictions.csv")
                old,_,subjects,studies=read_study_predictions(path)
                per_model[name]=summarize(predictions[name],dict(zip(zip(subjects,studies),old)),labels)
        else:
            combined=predictions["H17_02"]
            ref=read_wide(FRIEND/"val_predictions.csv")
            per_model={}
        result={"candidate":args.candidate,"mode":args.mode,"precision":precision,"batch_size":batch_size,
                "images":image_count,"model_eval":True,"aggregation":"mean of image sigmoid probabilities",
                "fusion":"frozen nested formula" if args.candidate=="E08_EMA" else "none",
                "ensemble":summarize(combined,ref,labels),"per_model":per_model}
    dataset.close()
    out=HERE/f"{args.candidate}_{args.mode}.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"path":str(out),"candidate":args.candidate,"mode":args.mode,
        "max_abs_diff":result.get("ensemble",{}).get("max_abs_diff")},ensure_ascii=False))

if __name__=="__main__":main()
