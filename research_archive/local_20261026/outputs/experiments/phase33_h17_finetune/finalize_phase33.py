"""CPU-only post-training audit for the single H17 head-only pilot."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cxr.engine.metrics import binary_auc

HERE = Path(__file__).resolve().parent
RUN = HERE / "P33_H17_HEAD_01"
ROOT = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(p)


def average_precision(y, p):
    """Average precision grouped by distinct score, matching threshold AP."""
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    positive = int(ys.sum())
    if positive == 0:
        raise ValueError("No positive Val Studies")
    tp = 0
    value = 0.0
    start = 0
    while start < len(ys):
        stop = start + 1
        while stop < len(ys) and ps[stop] == ps[start]:
            stop += 1
        new_tp = int(ys[start:stop].sum())
        tp += new_tp
        value += (new_tp / positive) * (tp / stop)
        start = stop
    return float(value)


def main():
    state = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    if state["state"] != "TRAINING_COMPLETE":
        raise RuntimeError("Training has not completed; final metrics remain PENDING")
    truth = {(r["Subject_id"], r["Study_id"]):r for r in read(ROOT / "batch_assets/val_labels.csv")}
    before = {(r["Subject_id"], r["Study_id"]):r for r in read(RUN / "baseline_val_predictions.csv")}
    after = {(r["Subject_id"], r["Study_id"]):r for r in read(RUN / "val_predictions.csv")}
    if len(truth) != 859 or set(before) != set(truth) or set(after) != set(truth):
        raise ValueError("Val Study IDs or counts differ")
    keys = sorted(truth)
    y = np.array([[int(truth[k][f"Target_{j}"]) for j in range(1, 11)] for k in keys])
    p0 = np.array([[float(before[k][f"Prob_{j}"]) for j in range(1, 11)] for k in keys])
    p1 = np.array([[float(after[k][f"Prob_{j}"]) for j in range(1, 11)] for k in keys])
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or np.any((p0 < 0)|(p0 > 1)) or np.any((p1 < 0)|(p1 > 1)):
        raise ValueError("Invalid Val probability")
    rows = []
    for j in range(10):
        ids = {k[0] for i,k in enumerate(keys) if y[i,j] == 1}
        a, b = binary_auc(y[:,j],p0[:,j]),binary_auc(y[:,j],p1[:,j])
        rows.append({"Class_ID":j+1,"Positive_Studies":int(y[:,j].sum()),"Positive_Patients":len(ids),
                     "Baseline_AUC":a,"Pilot_AUC":b,"Delta_AUC":b-a,
                     "Baseline_AP":average_precision(y[:,j],p0[:,j]),
                     "Pilot_AP":average_precision(y[:,j],p1[:,j])})
    with (RUN / "per_class_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    patients = sorted({k[0] for k in keys})
    groups = {subject:np.array([i for i,k in enumerate(keys) if k[0]==subject]) for subject in patients}
    rng=np.random.default_rng(42); deltas=[]; invalid=0
    while len(deltas)<2000:
        sampled=rng.choice(patients,size=len(patients),replace=True)
        index=np.concatenate([groups[s] for s in sampled])
        if any(len(np.unique(y[index,j]))<2 for j in range(10)):
            invalid+=1;continue
        delta=np.mean([binary_auc(y[index,j],p1[index,j])-binary_auc(y[index,j],p0[index,j]) for j in range(10)])
        deltas.append(float(delta))
    base_auc=float(np.mean([r["Baseline_AUC"] for r in rows]))
    pilot_auc=float(np.mean([r["Pilot_AUC"] for r in rows]))
    comparison={"baseline_auc":base_auc,"pilot_auc":pilot_auc,"delta":pilot_auc-base_auc,
                "paired_patient_cluster_bootstrap":{"seed":42,"patients":len(patients),"valid":len(deltas),
                    "invalid":invalid,"ci95":np.quantile(deltas,[.025,.975]).tolist(),
                    "invalid_rule":"discard draws with any class lacking positive or negative studies"},
                "label":"EXPLORATORY: official Val repeatedly used for selection; full Train student has no OOF"}
    write(RUN / "bootstrap_comparison.json",comparison)
    weight=RUN / "best_student.pth"
    model_size={"bytes":weight.stat().st_size,"sha256":sha(weight),"under_500000000_bytes":weight.stat().st_size<=500000000,
                "t4_latency":"T4_UNVERIFIED","local_4060_latency":"PENDING: only necessary consistency check authorized"}
    write(RUN / "model_size.json",model_size)
    manifest=json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_sha256"]={p.name:sha(p) for p in RUN.iterdir() if p.is_file() and p.name not in ("manifest.json","last_training.pt","source_model.pth")}
    write(RUN / "manifest.json",manifest)
    improved=comparison["paired_patient_cluster_bootstrap"]["ci95"][0]>0
    decision="Only exploratory evidence; do not add more hyperparameter trials." if not improved else "Exploratory improvement; independent patient evaluation required before replacement."
    report=("# Phase33 H17-02 分类头微调\n\n"
            f"原始同入口 Val AUC：{base_auc:.6f}；Pilot 最佳 Val AUC：{pilot_auc:.6f}；"
            f"差值 {pilot_auc-base_auc:+.6f}；患者配对 Bootstrap 95% CI {comparison['paired_patient_cluster_bootstrap']['ci95']}。\n\n"
            f"最佳 Epoch：{state['best_epoch']}；权重：{weight.stat().st_size} 字节；T4_UNVERIFIED。\n\n"
            "这次学生使用全部官方 Train；没有独立 OOF。Val 已反复用于开发，只能解释为探索性。"
            f"\n\n决策：{decision}\n")
    (RUN / "final_decision.md").write_text(report,encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
