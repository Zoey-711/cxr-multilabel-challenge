"""Paired E08 local timing of one-vs-two RGB conversions; Val images only."""
import csv
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P29=ROOT/"outputs/analysis/phase29_deployment"
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(P29/"deployment"))
from cxr.data.dataset import preprocess_image
from predict import load_models

def metric(v):
    return {"mean_ms":float(statistics.mean(v)),"p50_ms":float(np.percentile(v,50)),
            "p95_ms":float(np.percentile(v,95)),"max_ms":float(max(v))}

def timed(path,models,writer,out,mode,stages):
    torch.cuda.synchronize();total=time.perf_counter()
    t=time.perf_counter();raw=path.read_bytes();stages["file_read"].append((time.perf_counter()-t)*1000)
    t=time.perf_counter()
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        stages["decode"].append((time.perf_counter()-t)*1000)
        t=time.perf_counter()
        if mode=="double_rgb":tensor=preprocess_image(im.convert("RGB"),384,preprocessing="black_pad_v1")
        else:tensor=preprocess_image(im,384,preprocessing="black_pad_v1")
    stages["preprocess"].append((time.perf_counter()-t)*1000)
    t=time.perf_counter();x=tensor.unsqueeze(0).cuda();torch.cuda.synchronize()
    stages["host_to_device"].append((time.perf_counter()-t)*1000)
    t=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        parts={name:torch.sigmoid(model(x)).squeeze(0).float().cpu().numpy().astype(np.float64)
               for name,model in models.items()}
        e01=(parts["B01"]+parts["B02"]+parts["B03"])/3
        p=.5*(.5*e01+.5*parts["M02"])+.5*parts["N01_EMA"]
    torch.cuda.synchronize();stages["forward_and_cpu_result"].append((time.perf_counter()-t)*1000)
    t=time.perf_counter();writer.writerow([mode,path.name,*[format(float(v),".17g") for v in p]]);out.flush()
    stages["write_result"].append((time.perf_counter()-t)*1000)
    torch.cuda.synchronize();stages["end_to_end"].append((time.perf_counter()-total)*1000)
    return p

def main():
    measured=sorted((P29/"benchmark_images/pictures_100").glob("*.jpg"))
    warm=sorted((P29/"benchmark_images/warmup_5").glob("*.jpg"))
    if len(measured)!=100 or len(warm)!=5:raise ValueError("Expected 5 warmup and 100 Val images")
    models=load_models("E08_EMA")
    for path in warm:
        for mode in ("double_rgb","single_rgb"):
            with Image.open(path) as im:
                x=preprocess_image(im.convert("RGB") if mode=="double_rgb" else im,384,
                                   preprocessing="black_pad_v1").unsqueeze(0).cuda()
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                for model in models.values():model(x)
            torch.cuda.synchronize()
    stages={mode:defaultdict(list) for mode in ("double_rgb","single_rgb")}
    max_diff=0.0
    with (HERE/"e08_stage_outputs.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.writer(out);writer.writerow(["mode","image",*[f"Prob_{i}" for i in range(1,11)]])
        for round_id in range(3):
            for path in measured:
                order=("double_rgb","single_rgb") if round_id%2==0 else ("single_rgb","double_rgb")
                value={mode:timed(path,models,writer,out,mode,stages[mode]) for mode in order}
                max_diff=max(max_diff,float(np.max(np.abs(value["double_rgb"]-value["single_rgb"]))))
    report={"device":torch.cuda.get_device_name(),"precision":"fp16 autocast",
            "scope":"local Val files; explicit file read, decode, preprocessing, H2D, forward, CSV write; models preloaded",
            "warmup":5,"images":100,"rounds":3,"batch":1,
            "max_probability_diff_same_image":max_diff,
            "timing_instrumented_not_directly_comparable_to_cost_py":True,
            "modes":{mode:{stage:metric(values) for stage,values in steps.items()} for mode,steps in stages.items()}}
    (HERE/"e08_stage_breakdown.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"max_diff":max_diff,"double_ms":report["modes"]["double_rgb"]["end_to_end"]["mean_ms"],
                      "single_ms":report["modes"]["single_rgb"]["end_to_end"]["mean_ms"]}))

if __name__=="__main__":main()
