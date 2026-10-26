"""CPU-only Phase 19 Study-level error and fixed-candidate analysis.

Requires CXR_ARCHIVE to point at the original official ZIP. Reads images only
for a small, deterministic local quality sample; writes no image copies.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "outputs" / "experiments"
SUB = ROOT / "outputs" / "submissions"
PROB = [f"probability_{i}" for i in range(1, 11)]
TARG = [f"target_{i}" for i in range(1, 11)]
WEAK = {8: "Lung Opacity", 4: "Pneumonia", 1: "Enlarged Cardiomediastinum"}
RUN = {"B01":"b01_densenet121_384_20261016",
       "B02":"b02_densenet121_wd1e3_20261016_elevated",
       "B03":"b03_densenet121_lr3e5_20261016",
       "M02":"m02_convnext_tiny_384_aug_20261016_elevated",
       "T01":"t01_swin_tiny_384_20261018",
       "N01_EMA":"n01_efficientnet_v2_s_384_amp_retry_20261018"}


def frame(path: Path):
    df=pd.read_csv(path,dtype={"Subject_id":str,"Study_id":str})
    assert df.Study_id.is_unique and df.Subject_id.notna().all(),path
    assert set(PROB).issubset(df.columns),path
    assert np.isfinite(df[PROB].to_numpy(float)).all(),path
    return df.set_index("Study_id").sort_index()


label_path=EXP/RUN["B01"]/"validation_analysis"/"study_predictions.csv"
truth=frame(label_path)
assert len(truth)==859 and truth.Subject_id.nunique()==228
y=truth[TARG].to_numpy(int); subjects=truth.Subject_id.to_numpy(); ids=truth.index
def aligned(path):
    df=frame(path)
    assert len(df)==len(ids) and set(df.index)==set(ids),path
    df=df.loc[ids]
    assert df.Subject_id.equals(truth.Subject_id),path
    if set(TARG).issubset(df):assert np.array_equal(df[TARG].to_numpy(int),y),path
    return df[PROB].to_numpy(float)

p={}
for name,run in RUN.items():
    section="ema_validation_analysis" if name=="N01_EMA" else "validation_analysis"
    p[name]=aligned(EXP/run/section/"study_predictions.csv")
p["E01"]=(p["B01"]+p["B02"]+p["B03"])/3
p["E03"]=(p["E01"]+p["M02"])/2
p["E07"]=(p["E03"]+p["T01"])/2
p["E08_EMA"]=(p["E03"]+p["N01_EMA"])/2
p["E17_F1"]=aligned(SUB/"E17_F1_20261020"/"val_predictions.csv")
assert np.max(np.abs(p["E08_EMA"]-aligned(SUB/"E08_EMA_20261020"/"val_predictions.csv")))<1e-12

archive=os.environ.get("CXR_ARCHIVE")
if not archive or not Path(archive).is_file():raise RuntimeError("Set CXR_ARCHIVE to the original ZIP")
with zipfile.ZipFile(archive) as z:
    with z.open("cxr_first/valid.csv") as f:
        val=pd.read_csv(f,dtype=str)
    members=set(z.namelist())
assert len(val)==1422
assert set(val.Study_id)==set(ids)
assert val.groupby("Study_id").Subject_id.nunique().max()==1
val["Rows"]=val.Rows.astype(int);val["Columns"]=val.Columns.astype(int)
val["pad_fraction"]=1-np.minimum(val.Rows,val.Columns)/np.maximum(val.Rows,val.Columns)
grouped=val.groupby("Study_id",sort=False)
meta=pd.DataFrame({"Subject_id":grouped.Subject_id.first(),"image_count":grouped.size(),
    "view_positions":grouped.ViewPosition.agg(lambda x:";".join(sorted(set(v for v in x.dropna() if v)))),
    "mean_pad_fraction":grouped.pad_fraction.mean(),"max_pad_fraction":grouped.pad_fraction.max(),
    "min_source_side":grouped[["Rows","Columns"]].min().min(axis=1),
    "images":grouped.Dicom_id.agg(lambda x:";".join(x))}).loc[ids]
assert meta.Subject_id.equals(truth.Subject_id)


def auc(yy,pp):
    return float(roc_auc_score(yy,pp)) if len(set(yy))==2 else None


weak_rows=[]
for cls in WEAK:
    k=cls-1;labels=y[:,k]; a=p["E07"][:,k];b=p["E08_EMA"][:,k]
    pos_scores=b[labels==1];neg_scores=b[labels==0]
    for idx,study in enumerate(ids):
        score=b[idx]
        rank_error=(float(np.mean(neg_scores>score)+.5*np.mean(neg_scores==score)) if labels[idx]
                    else float(np.mean(pos_scores<score)+.5*np.mean(pos_scores==score)))
        weak_rows.append({"class_id":cls,"class_name":WEAK[cls],"Subject_id":subjects[idx],"Study_id":study,
          "label":int(labels[idx]),"e07_probability":a[idx],"e08_probability":score,
          "e08_minus_e07_probability":score-a[idx],"opposite_label_rank_error_fraction":rank_error,
          "image_count":int(meta.iloc[idx].image_count),"view_positions":meta.iloc[idx].view_positions,
          "mean_pad_fraction":float(meta.iloc[idx].mean_pad_fraction),
          "max_pad_fraction":float(meta.iloc[idx].max_pad_fraction),
          "min_source_side":int(meta.iloc[idx].min_source_side),
          "cooccurs_consolidation":int(y[idx,2]),"cooccurs_pneumonia":int(y[idx,3]),
          "cooccurs_lung_opacity":int(y[idx,7]),
          "source_predictions":"outputs/submissions/E08_EMA_20261020/val_predictions.csv",
          "source_labels":"outputs/experiments/"+RUN["B01"]+"/validation_analysis/study_predictions.csv",
          "source_metadata":"cxr_first/valid.csv inside original official ZIP"})
pd.DataFrame(weak_rows).to_csv(OUT/"weak_class_error_analysis.csv",index=False)


strat=[]
views=meta.view_positions.fillna("").str.split(";")
groups={"single_image":meta.image_count==1,"multi_image":meta.image_count>1,
        "AP_present":views.apply(lambda x:"AP" in x),"PA_present":views.apply(lambda x:"PA" in x),
        "lateral_present":views.apply(lambda x:any(v in ("LATERAL","LL") for v in x))}
for cls in WEAK:
    k=cls-1
    for g,mask in groups.items():
        m=mask.to_numpy();yy=y[m,k]
        strat.append({"class_id":cls,"class_name":WEAK[cls],"group":g,
           "groups_overlap":g.endswith("_present"),"study_count":int(m.sum()),
           "patient_count":int(len(set(subjects[m]))),"positive_studies":int(yy.sum()),
           "negative_studies":int(len(yy)-yy.sum()),
           "positive_patients":int(len(set(subjects[m & (y[:,k]==1)]))),
           "e07_auc":auc(yy,p["E07"][m,k]),"e08_auc":auc(yy,p["E08_EMA"][m,k]),
           "e08_ap":float(average_precision_score(yy,p["E08_EMA"][m,k])) if yy.sum() else None,
           "source":"weak_class_error_analysis.csv and original valid.csv"})
pd.DataFrame(strat).to_csv(OUT/"study_view_stratification.csv",index=False)


# Small deterministic image-quality sample: metadata and thumbnail statistics
# only. No pixels, thumbnails or reports are exported.
cases=pd.DataFrame(weak_rows)
sample=[]
for cls in WEAK:
    for label in (0,1):
        x=cases[(cases.class_id==cls)&(cases.label==label)].nlargest(3,"opposite_label_rank_error_fraction")
        sample.extend(x.Study_id.tolist())
sample=list(dict.fromkeys(sample))
quality=[]
with zipfile.ZipFile(archive) as z:
    for study in sample:
        row=val[val.Study_id==study].iloc[0]
        member=f"cxr_first/valid/{row.Subject_id}/{row.Study_id}/{row.Dicom_id}.jpg"
        if member not in members:
            quality.append({"Study_id":study,"image_id":row.Dicom_id,"status":"UNAVAILABLE: member missing"})
            continue
        with Image.open(io.BytesIO(z.read(member))) as im:
            original_mode=im.mode; original_size=im.size
            gray=np.asarray(im.convert("L").resize((128,128)),dtype=np.float32)
        edge=np.concatenate([gray[:8].ravel(),gray[-8:].ravel(),gray[:,:8].ravel(),gray[:,-8:].ravel()])
        quality.append({"Study_id":study,"Subject_id":row.Subject_id,"image_id":row.Dicom_id,
         "status":"READ_OK","mode":original_mode,"actual_width":original_size[0],
         "actual_height":original_size[1],"metadata_width":int(row.Columns),"metadata_height":int(row.Rows),
         "metadata_dimensions_match":bool(original_size==(int(row.Columns),int(row.Rows))),
         "gray_mean_128":float(gray.mean()),"gray_std_128":float(gray.std()),
         "dark_pixel_fraction_128":float(np.mean(gray<15)),
         "bright_pixel_fraction_128":float(np.mean(gray>245)),
         "edge_dark_fraction_128":float(np.mean(edge<15)),
         "mean_pad_fraction":float(meta.loc[study].mean_pad_fraction),
         "source_member":member})
pd.DataFrame(quality).to_csv(OUT/"local_image_quality_review.csv",index=False)

co={"pneumonia_positive_studies":int(y[:,3].sum()),
    "pneumonia_and_consolidation":int(np.sum((y[:,3]==1)&(y[:,2]==1))),
    "pneumonia_and_lung_opacity":int(np.sum((y[:,3]==1)&(y[:,7]==1))),
    "pneumonia_and_both":int(np.sum((y[:,3]==1)&(y[:,2]==1)&(y[:,7]==1))),
    "consolidation_and_lung_opacity":int(np.sum((y[:,2]==1)&(y[:,7]==1))),
    "no_finding_and_any_disease":int(np.sum((y[:,9]==1)&(y[:,:9].any(axis=1))))}
(OUT/"weak_label_cooccurrence.json").write_text(json.dumps(co,indent=2),encoding="utf-8")


# Prespecified, existing probability schemes only; no new weights or search.
deploy=pd.read_csv(ROOT/"outputs"/"analysis"/"phase18_leaderboard_diagnosis"/"deployment_feasibility.csv")
candidates=["E01","E03","E07","E08_EMA","M02","T01","N01_EMA","E17_F1"]
rows=[]
for name in candidates:
    key="E17-F1" if name=="E17_F1" else name
    source=deploy[deploy.candidate==key].iloc[0]
    vals=p[name]
    rows.append({"candidate":name,"val_studies":859,
      "val_macro_auc":float(np.mean([roc_auc_score(y[:,j],vals[:,j]) for j in range(10)])),
      "per_class_auc_json":json.dumps([roc_auc_score(y[:,j],vals[:,j]) for j in range(10)]),
      "model_count":int(source.inference_model_count) if "inference_model_count" in source else len(str(source.weights_included).split("+")),
      "weights_included":source.weights_included,"weight_bytes":int(source.weight_bytes),
      "size_status":"PASS" if source.weight_bytes<=500_000_000 else "SIZE_FAIL",
      "local_4060_mean_ms":source.local_4060_mean_ms,
      "benchmark_hardware":"RTX 4060 Laptop" if source.local_4060_mean_ms!="UNAVAILABLE" else "UNAVAILABLE",
      "target_latency_ms":"UNAVAILABLE","deployment_status":"LATENCY_UNVERIFIED",
      "score_source":"frozen aligned Study-level Val probabilities; no hidden Test labels"})
pd.DataFrame(rows).to_csv(OUT/"lightweight_ensemble_candidates.csv",index=False)


# Patient-cluster paired bootstrap; all studies from sampled patient retained.
rng=np.random.default_rng(20261021)
unique=np.unique(subjects); by_patient={s:np.flatnonzero(subjects==s) for s in unique}
comps=["E01","E03","E07","M02","T01","N01_EMA","E17_F1"]
draws={k:[] for k in comps};class1=[];invalid=0
for attempt in range(10000):
    picked=rng.choice(unique,size=len(unique),replace=True)
    ii=np.concatenate([by_patient[s] for s in picked]); yy=y[ii]
    if any(yy[:,j].min()==yy[:,j].max() for j in range(10)):
        invalid+=1;continue
    base=[roc_auc_score(yy[:,j],p["E08_EMA"][ii,j]) for j in range(10)]
    class1.append(base[0])
    for name in comps:
        v=[roc_auc_score(yy[:,j],p[name][ii,j]) for j in range(10)]
        draws[name].append(float(np.mean(np.array(v)-base)))
    if len(class1)==1000:break
if len(class1)<1000:raise RuntimeError("Could not obtain 1000 valid patient-cluster samples")
bootstrap={"seed":20261021,"patients":len(unique),"valid":1000,"invalid":invalid,
 "unit":"Subject_id, retaining all Studies", "invalid_policy":"discard and redraw if any of 10 classes has no positive or negative Study",
 "class1_e08_auc":roc_auc_score(y[:,0],p["E08_EMA"][:,0]),
 "class1_e08_auc_ci95":np.quantile(class1,[.025,.975]).tolist(),
 "comparisons_vs_e08":{name:{"delta_macro_auc":float(np.mean([roc_auc_score(y[:,j],p[name][:,j])-roc_auc_score(y[:,j],p["E08_EMA"][:,j]) for j in range(10)])),
   "ci95":np.quantile(draws[name],[.025,.975]).tolist()} for name in comps},
 "warning":"All candidates reused official Val for checkpoint selection; intervals do not remove selection bias"}
(OUT/"bootstrap_comparisons.json").write_text(json.dumps(bootstrap,indent=2),encoding="utf-8")
print(json.dumps({"quality_images_checked":len(quality),"cooccurrence":co,
 "class1_auc_ci":bootstrap["class1_e08_auc_ci95"],"candidate_count":len(rows),
 "bootstrap_invalid":invalid},indent=2))
