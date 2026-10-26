"""Read-only frozen artifact audit and deterministic local Val timing sample."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cxr.data.archive import image_member, parse_positive_classes, read_split_rows
from cxr.engine.b01_analysis import binary_auc
from cxr.utils.labels import CLASS_NAMES

ARCHIVE = Path("/AUTHORIZED/cxr_first.zip")
E08 = ROOT / "outputs/submissions/E08_EMA_20261020"
FRIEND = Path("/LOCAL_WORKSPACE/cxr-friend-results/versions/cbfc5997438f371086cd1fbe232890590cd67ffb/results/phase17_batch/experiments/H17_DISTILL_02")

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def read_probs(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    keys = [(r["Subject_id"], r["Study_id"]) for r in records]
    if len(keys) != len(set(keys)) or len(keys) != 859:
        raise ValueError(f"Study coverage invalid: {path}")
    values = {}
    for r, key in zip(records, keys):
        cols = [f"probability_{i}" for i in range(1,11)] if "probability_1" in r else [f"Prob_{i}" for i in range(1,11)]
        p = np.array([float(r[c]) for c in cols])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid probability")
        values[key] = p
    return values, records

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    m = json.loads((E08 / "manifest.json").read_text(encoding="utf-8"))
    size = 0
    assets = []
    for name, item in m["models"].items():
        path = ROOT / "outputs/experiments" / item["run_id"] / item["checkpoint"]
        config = path.parent / "config.yaml"
        got = digest(path)
        if got != item["checkpoint_sha256"] or path.stat().st_size != item["checkpoint_bytes"]:
            raise ValueError(f"E08 checkpoint mismatch: {name}")
        if not config.is_file() or digest(config) != item["config_sha256"]:
            raise ValueError(f"E08 frozen config mismatch: {name}")
        assets.append({"model": name, "checkpoint": str(path), "sha256": got, "bytes": path.stat().st_size,
                       "config": str(config), "config_sha256": digest(config) if config.is_file() else None})
        size += path.stat().st_size
    if size != m["total_checkpoint_bytes"]:
        raise ValueError("E08 total size mismatch")
    config_bytes=sum(Path(r["config"]).stat().st_size for r in assets)
    h17_weight=HERE / "weights/H17_DISTILL_02_best_model.pth"
    h17_expected="35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69"
    h17_actual_bytes=h17_weight.stat().st_size if h17_weight.is_file() else None
    h17_actual_sha=digest(h17_weight) if h17_weight.is_file() else None
    h17_config=FRIEND/"resolved_config.yaml"
    h17_config_bytes=h17_config.stat().st_size
    if h17_weight.is_file() and (h17_actual_bytes!=81651999 or h17_actual_sha!=h17_expected):
        raise ValueError("Downloaded H17 weight mismatches frozen manifest")
    (HERE / "model_size_report.json").write_text(json.dumps({"E08_EMA": {"models": assets,
        "weight_bytes": size,"necessary_config_bytes":config_bytes,"weight_plus_config_bytes":size+config_bytes,
        "under_500m_decimal": size+config_bytes <= 500_000_000},
        "H17_02": {"expected_weight_bytes": 81651999, "expected_sha256":h17_expected,
        "actual_weight_bytes":h17_actual_bytes,"actual_sha256":h17_actual_sha,
        "necessary_config_bytes":h17_config_bytes,
        "weight_plus_config_bytes":h17_actual_bytes+h17_config_bytes if h17_actual_bytes else None,
        "local_weight_available":h17_weight.is_file(),
        "under_500m_decimal":bool(h17_actual_bytes and h17_actual_bytes+h17_config_bytes<=500_000_000)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    e08, e08_rows = read_probs(E08 / "val_predictions.csv")
    h17, _ = read_probs(FRIEND / "val_predictions.csv")
    if set(e08) != set(h17):
        raise ValueError("H17/E08 Val Study mismatch")
    rows=read_split_rows(ARCHIVE,"val")
    labels = {}
    for row in rows:
        key=(row["Subject_id"],row["Study_id"])
        positives=parse_positive_classes(row["Predict_class"])
        value=np.array([int(i in positives) for i in range(1,11)])
        if key in labels and not np.array_equal(labels[key],value):
            raise ValueError(f"Inconsistent Val labels: {key}")
        labels[key]=value
    if set(labels)!=set(e08):
        raise ValueError("Official Val label Study IDs do not match prediction IDs")
    keys = sorted(e08)
    y = np.stack([labels[k] for k in keys]); a = np.stack([e08[k] for k in keys]); b = np.stack([h17[k] for k in keys])
    per = []
    for c, name in enumerate(CLASS_NAMES):
        au, bu = binary_auc(y[:,c], a[:,c]), binary_auc(y[:,c], b[:,c])
        per.append({"class_id":c+1,"class_name":name,"positive_studies":int(y[:,c].sum()),
                    "e08_auc":au,"h17_02_auc":bu,"h17_minus_e08_auc":bu-au})
    with (HERE / "h17_e08_per_class.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=per[0]);w.writeheader();w.writerows(per)
    macro_a=float(np.mean([r["e08_auc"] for r in per]));macro_b=float(np.mean([r["h17_02_auc"] for r in per]))
    # Deterministic patient-cluster bootstrap; resample whole patient groups.
    groups = {}
    for i,k in enumerate(keys):groups.setdefault(k[0],[]).append(i)
    blocks=list(groups.values()); rng=np.random.default_rng(44); deltas=[]; invalid=0
    while len(deltas)<2000:
        sel=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        ac=[binary_auc(y[sel,c],a[sel,c]) for c in range(10)]
        bc=[binary_auc(y[sel,c],b[sel,c]) for c in range(10)]
        if any(x is None for x in ac+bc):invalid+=1;continue
        deltas.append(float(np.mean(bc)-np.mean(ac)))
    report={"patients":len(blocks),"studies":len(keys),"e08_macro_auc":macro_a,"h17_macro_auc":macro_b,
            "h17_minus_e08":macro_b-macro_a,"bootstrap_seed":44,"valid":len(deltas),"invalid":invalid,
            "ci95":[float(v) for v in np.percentile(deltas,[2.5,97.5])],
            "h17_val_sha256":digest(FRIEND/"val_predictions.csv"),"e08_val_sha256":digest(E08/"val_predictions.csv")}
    (HERE/"val_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # Reference sample: 105 distinct single-image Val studies to avoid confusing image with Study mean.
    counts=Counter((r["Subject_id"],r["Study_id"]) for r in rows)
    eligible=[r for r in rows if counts[(r["Subject_id"],r["Study_id"])]==1]
    eligible.sort(key=lambda r: (r["ViewPosition"],int(r["Rows"])/max(1,int(r["Columns"])),r["Dicom_id"]))
    chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,105,dtype=int)]
    base=HERE/"benchmark_images"
    base.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        for index,row in enumerate(chosen):
            group="warmup_5" if index<5 else "pictures_100"
            target=base/group/f"{row['Dicom_id']}.jpg"
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.write_bytes(zf.read(image_member(row,"val")))
    with (base/"E08_image_reference.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["image"]+[f"Prob_{i}" for i in range(1,11)])
        for row in chosen[5:]:
            p=e08[(row["Subject_id"],row["Study_id"])]
            w.writerow([f"{row['Dicom_id']}.jpg"]+[format(float(v),".17g") for v in p])
    with (base/"H17_image_reference.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["image"]+[f"Prob_{i}" for i in range(1,11)])
        for row in chosen[5:]:
            p=h17[(row["Subject_id"],row["Study_id"])]
            w.writerow([f"{row['Dicom_id']}.jpg"]+[format(float(v),".17g") for v in p])
    with (base/"sample_image_index.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["image","Subject_id","Study_id","study_image_count"])
        for row in chosen[5:]:
            w.writerow([f"{row['Dicom_id']}.jpg",row["Subject_id"],row["Study_id"],1])
    (base/"sample_manifest.json").write_text(json.dumps({"source":"official Val images, never hidden Test",
        "selection":"105 single-image Studies, deterministic view/aspect sort and linspace",
        "warmup":5,"measured":100,"image_ids":[r["Dicom_id"] for r in chosen],
        "archive":str(ARCHIVE)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":main()
