"""Independently reload H17-02 raw best weight and verify 859 official Val Studies."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(HERE/"deployment"))
from cxr.data.archive import read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.engine.b01_analysis import binary_auc
from predict import load_models

ARCHIVE=Path("F:/数据集/cxr_first.zip")
FRIEND=Path("E:/计挑赛/cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/experiments/H17_DISTILL_02/val_predictions.csv")

def main():
    model=load_models("H17_02")["H17_02"]
    rows=read_split_rows(ARCHIVE,"val")
    dataset=CXRArchiveDataset(ARCHIVE,"val",rows=rows,input_size=384,preprocessing="black_pad_v1")
    loader=DataLoader(dataset,batch_size=8,num_workers=0,shuffle=False)
    images=defaultdict(list); labels={}
    with torch.inference_mode():
        for batch in loader:
            tensor=batch["image"].cuda()
            with torch.autocast("cuda",dtype=torch.float16):
                values=torch.sigmoid(model(tensor)).float().cpu().numpy()
            for i,p in enumerate(values):
                key=(batch["subject_id"][i],batch["study_id"][i])
                images[key].append(p.astype(np.float64))
                truth=batch["target"][i].numpy()
                if key in labels and not np.array_equal(labels[key],truth):
                    raise ValueError("Inconsistent Val label")
                labels[key]=truth
    dataset.close()
    prediction={k:np.mean(v,axis=0) for k,v in images.items()}
    with FRIEND.open(encoding="utf-8-sig",newline="") as f:
        reference={(r["Subject_id"],r["Study_id"]):np.array([float(r[f"Prob_{c}"]) for c in range(1,11)])
                   for r in csv.DictReader(f)}
    if set(prediction)!=set(reference) or len(prediction)!=859:
        raise ValueError("Val Study coverage mismatch")
    keys=sorted(prediction)
    y=np.stack([labels[k] for k in keys])
    p=np.stack([prediction[k] for k in keys])
    q=np.stack([reference[k] for k in keys])
    report={"model":"H17_02","images":len(rows),"studies":len(keys),
            "inference_batch":8,"reference_reported_batch":16,
            "recomputed_macro_auc":float(np.mean([binary_auc(y[:,c],p[:,c]) for c in range(10)])),
            "reference_macro_auc":float(np.mean([binary_auc(y[:,c],q[:,c]) for c in range(10)])),
            "max_abs_probability_diff":float(np.max(np.abs(p-q))),
            "status":"PASS" if np.max(np.abs(p-q))<=1e-3 else "NUMERIC_DIFFERENCE_REVIEW"}
    (HERE/"h17_weight_val_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))

if __name__=="__main__":main()
