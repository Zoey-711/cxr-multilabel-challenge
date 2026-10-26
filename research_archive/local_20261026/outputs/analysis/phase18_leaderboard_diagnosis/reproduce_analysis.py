"""Read-only analysis of frozen Phase 17 and earlier validation artifacts.

Run from the project root with a CPU Python environment containing pandas,
numpy, scikit-learn, matplotlib and PyYAML. Writes only into this directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "outputs" / "experiments"
AN = ROOT / "outputs" / "analysis"
SUB = ROOT / "outputs" / "submissions"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = ["Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
         "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
         "Lung Opacity", "Pleural Effusion", "No Finding"]
PROBS = [f"probability_{i}" for i in range(1, 11)]
TARGETS = [f"target_{i}" for i in range(1, 11)]
RUNS = {
    "B01": "b01_densenet121_384_20261016",
    "B02": "b02_densenet121_wd1e3_20261016_elevated",
    "B03": "b03_densenet121_lr3e5_20261016",
    "M01": "m01_convnext_tiny_384_20261016",
    "M02": "m02_convnext_tiny_384_aug_20261016_elevated",
    "S01": "s01_convnext_tiny_study_balanced_384_20261017",
    "L01": "l01_convnext_tiny_asl_384_20261017_elevated",
    "R01": "r01_convnext_tiny_512_20261017_elevated",
    "T01": "t01_swin_tiny_384_20261018",
    "N01": "n01_efficientnet_v2_s_384_amp_retry_20261018",
    "D17-F1": "d17_f1_20261019", "C17-F1": "c17_f1_20261019",
    "S17-F1": "s17_f1_20261019", "D17-F2": "d17_f2_20261020",
    "C17-F2": "c17_f2_20261020", "S17-F2": "s17_f2_20261020",
    "P15A": "p15a_densenet121_internal_train_384",
    "P15B": "p15b_convnext_tiny_internal_train_384",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_studies(path: Path, *, labels: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Subject_id": str, "Study_id": str})
    if len(df) != df.Study_id.nunique() or df.Subject_id.isna().any():
        raise ValueError(f"Duplicate Study or missing Subject in {path}")
    cols = PROBS + (TARGETS if labels else [])
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {set(cols)-set(df.columns)}")
    if not np.isfinite(df[PROBS].to_numpy(float)).all():
        raise ValueError(f"Nonfinite probabilities: {path}")
    if not df[PROBS].apply(lambda x: x.between(0, 1).all()).all():
        raise ValueError(f"Out-of-range probabilities: {path}")
    return df.set_index("Study_id").sort_index()


def aligned(frame: pd.DataFrame, truth: pd.DataFrame, *, labels: bool = False) -> np.ndarray:
    if set(frame.index) != set(truth.index) or len(frame) != len(truth):
        raise ValueError("Validation Study IDs differ")
    frame = frame.loc[truth.index]
    if not frame.Subject_id.equals(truth.Subject_id):
        raise ValueError("Subject IDs differ after Study alignment")
    if labels and not np.array_equal(frame[TARGETS].to_numpy(int), truth[TARGETS].to_numpy(int)):
        raise ValueError("Labels differ after Study alignment")
    return frame[PROBS].to_numpy(float)


source_path = EXP / RUNS["B01"] / "validation_analysis" / "study_predictions.csv"
truth = read_studies(source_path, labels=True)
assert len(truth) == 859 and truth.Subject_id.nunique() == 228
y = truth[TARGETS].to_numpy(int)
subjects = truth.Subject_id.to_numpy()
pred = {}
sources = {}
for name, run in RUNS.items():
    p = EXP / run / "validation_analysis" / "study_predictions.csv"
    if p.exists():
        pred[name] = aligned(read_studies(p, labels=True), truth, labels=True)
        sources[name] = str(p.relative_to(ROOT))
for name, folder in (("D17-F1", "d17_f1_20261019"), ("C17-F1", "c17_f1_20261019"),
                     ("S17-F1", "s17_f1_20261019"), ("D17-F2", "d17_f2_20261020"),
                     ("C17-F2", "c17_f2_20261020"), ("S17-F2", "s17_f2_20261020")):
    assert name in pred, folder
ema_path = EXP / RUNS["N01"] / "ema_validation_analysis" / "study_predictions.csv"
pred["N01_EMA"] = aligned(read_studies(ema_path, labels=True), truth, labels=True)
sources["N01_EMA"] = str(ema_path.relative_to(ROOT))
pred["E01"] = (pred["B01"] + pred["B02"] + pred["B03"]) / 3
pred["E03"] = (pred["E01"] + pred["M02"]) / 2
pred["E07"] = (pred["E03"] + pred["T01"]) / 2
pred["E08_EMA"] = (pred["E03"] + pred["N01_EMA"]) / 2
sources["E01"] = "calculated: equal B01/B02/B03 Study probabilities"
sources["E03"] = "calculated: 0.5 E01 + 0.5 M02"
sources["E07"] = "calculated: 0.5 E03 + 0.5 T01; outputs/submissions/E07_20261018/fusion_config.json"
sources["E08_EMA"] = "calculated: 0.5 E03 + 0.5 N01_EMA; outputs/submissions/E08_EMA_20261020/manifest.json"
saved_e08 = read_studies(SUB / "E08_EMA_20261020" / "val_predictions.csv")
e08_diff = float(np.max(np.abs(pred["E08_EMA"] - aligned(saved_e08, truth))))
assert e08_diff < 1e-12, e08_diff


def class_stats(values: np.ndarray) -> list[dict]:
    return [{"class_id": i + 1, "class_name": NAMES[i],
             "positive_studies": int(y[:, i].sum()), "negative_studies": int(len(y)-y[:, i].sum()),
             "positive_patients": int(len(set(subjects[y[:, i] == 1]))),
             "auc": float(roc_auc_score(y[:, i], values[:, i])),
             "ap": float(average_precision_score(y[:, i], values[:, i]))}
            for i in range(10)]


def macro(values: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, j], values[:, j]) for j in range(10)]))


expected = {"E07": 0.8115193037842005, "E08_EMA": 0.8116686461316818,
            "E03": 0.8099309507263751}
for name, value in expected.items():
    assert abs(macro(pred[name]) - value) < 1e-10, (name, macro(pred[name]), value)


def rank_rescue(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> dict:
    pos = labels == 1
    apair = a[pos, None] - a[None, ~pos]
    bpair = b[pos, None] - b[None, ~pos]
    # Ties are not silently called correct; count only strict opposite decisions.
    return {"a_only_correct_pairs": int(np.count_nonzero((apair > 0) & (bpair < 0))),
            "b_only_correct_pairs": int(np.count_nonzero((bpair > 0) & (apair < 0))),
            "pair_count": int(apair.size),
            "tie_pairs_either": int(np.count_nonzero((apair == 0) | (bpair == 0)))}


per_class = []
for j in range(10):
    a, b = pred["E07"][:, j], pred["E08_EMA"][:, j]
    yy = y[:, j]
    ra = rank_rescue(a, b, yy)
    per_class.append({"class_id": j+1, "class_name": NAMES[j],
        "positive_studies": int(yy.sum()), "negative_studies": int(len(yy)-yy.sum()),
        "positive_patients": int(len(set(subjects[yy == 1]))),
        "e07_auc": roc_auc_score(yy, a), "e08_auc": roc_auc_score(yy, b),
        "delta_auc_e08_minus_e07": roc_auc_score(yy, b)-roc_auc_score(yy, a),
        "e07_ap": average_precision_score(yy, a), "e08_ap": average_precision_score(yy, b),
        "e07_e08_pearson": float(np.corrcoef(a, b)[0, 1]),
        "e07_e08_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
        "e07_positive_p50": float(np.median(a[yy == 1])),
        "e08_positive_p50": float(np.median(b[yy == 1])),
        "e07_negative_p50": float(np.median(a[yy == 0])),
        "e08_negative_p50": float(np.median(b[yy == 0])),
        "e07_only_correct_pairs": ra["a_only_correct_pairs"],
        "e08_only_correct_pairs": ra["b_only_correct_pairs"],
        "positive_negative_pairs": ra["pair_count"],
        "pairwise_ties_either": ra["tie_pairs_either"],
        "source_e07": sources["E07"], "source_e08": sources["E08_EMA"],
        "label_source": str(source_path.relative_to(ROOT))})
pc = pd.DataFrame(per_class)
pc.to_csv(OUT / "e07_e08_per_class.csv", index=False, encoding="utf-8")


# Cluster bootstrap: each selected patient contributes every one of their Studies.
rng = np.random.default_rng(20261021)
patient_ids = np.unique(subjects)
patient_rows = {p: np.flatnonzero(subjects == p) for p in patient_ids}
boot = []
per_class_boot = []
invalid = 0
for attempt in range(20000):
    sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    rows = np.concatenate([patient_rows[p] for p in sampled])
    yy = y[rows]
    if any(yy[:, j].min() == yy[:, j].max() for j in range(10)):
        invalid += 1
        continue
    deltas = [roc_auc_score(yy[:, j], pred["E08_EMA"][rows, j]) -
              roc_auc_score(yy[:, j], pred["E07"][rows, j]) for j in range(10)]
    boot.append(float(np.mean(deltas)))
    per_class_boot.append(deltas)
    if len(boot) == 2000:
        break
if len(boot) < 2000:
    raise RuntimeError("Could not collect 2000 valid patient-cluster bootstrap samples")
bootstrap = {"seed": 20261021, "unit": "Subject_id; all Studies retained per sampled patient",
             "patients": len(patient_ids), "valid": len(boot), "invalid": invalid,
             "invalid_policy": "discard and redraw sample if any of ten classes lacks a positive or negative Study",
             "e08_minus_e07_auc_delta": macro(pred["E08_EMA"])-macro(pred["E07"]),
             "ci95_percentile": np.quantile(boot, [0.025, 0.975]).tolist(),
             "per_class_delta_ci95": np.quantile(per_class_boot, [0.025, 0.975], axis=0).T.tolist(),
             "source": [sources["E07"], sources["E08_EMA"]]}
(OUT / "e07_e08_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")


plt.rcParams["font.family"] = "DejaVu Sans"
x = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x-.18, pc.e07_auc, width=.36, label="E07")
ax.bar(x+.18, pc.e08_auc, width=.36, label="E08-EMA")
ax.set(xticks=x, ylim=(0, 1), xlabel="Original class ID", ylabel="Study AUROC")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "e07_e08_auc.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, pc.delta_auc_e08_minus_e07, color=np.where(pc.delta_auc_e08_minus_e07 >= 0, "#367c5c", "#b64c44"))
ax.axhline(0, color="black", lw=.8)
ax.set(xticks=x, xlabel="Original class ID", ylabel="E08-EMA - E07 AUROC")
fig.tight_layout(); fig.savefig(FIG / "e08_auc_delta.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(pc.positive_studies, pc.e08_auc)
for _, r in pc.iterrows(): ax.annotate(str(int(r.class_id)), (r.positive_studies, r.e08_auc))
ax.set(xlabel="Positive validation Studies", ylabel="E08-EMA Study AUROC")
fig.tight_layout(); fig.savefig(FIG / "support_vs_auc.png", dpi=160); plt.close(fig)


# Same 859 Studies for every model and every column in this comparison.
compare = [n for n in ("E01", "E03", "E07", "E08_EMA", "M02", "T01", "N01", "N01_EMA",
                        "B01", "B02", "B03") if n in pred]
complement = []
for i, a in enumerate(compare):
    for b in compare[i+1:]:
        for j in range(10):
            pa, pb = pred[a][:, j], pred[b][:, j]
            rr = rank_rescue(pa, pb, y[:, j])
            complement.append({"model_a": a, "model_b": b, "class_id": j+1,
                "pearson": float(np.corrcoef(pa, pb)[0,1]),
                "spearman": float(pd.Series(pa).corr(pd.Series(pb), method="spearman")),
                "auc_a": roc_auc_score(y[:,j],pa), "auc_b": roc_auc_score(y[:,j],pb),
                "a_only_correct_pairs": rr["a_only_correct_pairs"],
                "b_only_correct_pairs": rr["b_only_correct_pairs"],
                "positive_negative_pairs": rr["pair_count"],
                "source_a": sources[a], "source_b": sources[b]})
pd.DataFrame(complement).to_csv(OUT / "model_complementarity.csv", index=False)
fixed = [("E01", "E03", "add M02 as 0.5 E01 + 0.5 M02"),
         ("E03", "E07", "add T01 as 0.5 E03 + 0.5 T01"),
         ("E03", "E08_EMA", "add N01_EMA as 0.5 E03 + 0.5 N01_EMA"),
         ("N01", "N01_EMA", "ordinary versus EMA checkpoints"),
         ("E07", "E08_EMA", "replace T01 branch with N01_EMA branch")]
pd.DataFrame([{"baseline": a, "candidate": b, "frozen_comparison": d,
               "baseline_auc": macro(pred[a]), "candidate_auc": macro(pred[b]),
               "delta_auc": macro(pred[b])-macro(pred[a])} for a,b,d in fixed]).to_csv(
                   OUT / "fixed_ablation.csv", index=False)


# Inventory only actual inference checkpoints. Do not treat resumable optimizer
# states or failed/preflight directories as trained models.
inv = []
for run in sorted(EXP.iterdir()):
    if not run.is_dir(): continue
    ck = run / "best_model.pt"
    if not ck.exists(): continue
    name = next((k for k,v in RUNS.items() if v == run.name), run.name)
    config_path = run / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    metrics_path = run / "best_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    valp = run / "validation_analysis" / "study_predictions.csv"
    oofp = run / "oof_analysis" / "study_predictions.csv"
    valauc = macro(pred[name]) if name in pred else metrics.get("val",{}).get("macro_auc")
    oofauc = None
    if oofp.exists():
        df = read_studies(oofp, labels=True)
        oofauc = float(np.mean([roc_auc_score(df[t], df[p]) for t,p in zip(TARGETS,PROBS)]))
    scope = "official Train full"
    if "train_subset" in config.get("data",{}):
        scope = f"official Train excluding fold {config['data']['train_subset'].get('fold')}"
    if name.startswith("P15"):
        scope = "Phase15 internal Train only; Holdout not for new selection"
    row = {"model_id":name,"run_dir":str(run.relative_to(ROOT)),
        "architecture":config.get("model",{}).get("name", "UNAVAILABLE"),
        "config_path":str(config_path.relative_to(ROOT)) if config_path.exists() else "UNAVAILABLE",
        "best_epoch":metrics.get("epoch", "UNAVAILABLE"),"val_macro_auc":valauc,
        "val_per_class_auc_json":json.dumps([r["auc"] for r in class_stats(pred[name])]) if name in pred else "UNAVAILABLE",
        "val_per_class_ap_json":json.dumps([r["ap"] for r in class_stats(pred[name])]) if name in pred else "UNAVAILABLE",
        "val_positive_studies_json":json.dumps(y.sum(axis=0).tolist()) if name in pred else "UNAVAILABLE",
        "oof_studies":len(pd.read_csv(oofp)) if oofp.exists() else 0,
        "oof_macro_auc":oofauc if oofp.exists() else "N/A",
        "oof_scope": "single frozen patient fold" if oofp.exists() else "N/A",
        "official_val_reused_for_early_stopping":True,
        "train_scope":scope,"input_size":config.get("data",{}).get("input_size","UNAVAILABLE"),
        "preprocessing":"black_pad_v1 (see Phase6 audit; verify run config)" if config else "UNAVAILABLE",
        "study_aggregation":config.get("validation",{}).get("study_fusion","UNAVAILABLE"),
        "weight_bytes":ck.stat().st_size,"checkpoint_sha256":sha(ck),
        "inference_model_count":1,
        "val_prediction_source":str(valp.relative_to(ROOT)) if valp.exists() else "UNAVAILABLE",
        "oof_prediction_source":str(oofp.relative_to(ROOT)) if oofp.exists() else "N/A",
        "target_latency_status":"LATENCY_UNVERIFIED","official_score":"UNAVAILABLE"}
    inv.append(row)
    if name == "N01" and (run / "best_ema_model.pt").exists():
        ema = row.copy(); ema["model_id"]="N01_EMA"; ema["weight_bytes"]=(run/"best_ema_model.pt").stat().st_size
        ema["checkpoint_sha256"]=sha(run/"best_ema_model.pt")
        ema["val_macro_auc"]=macro(pred["N01_EMA"])
        ema["best_epoch"]=json.loads((run/"best_ema_metrics.json").read_text(encoding="utf-8")).get("epoch")
        ema["val_prediction_source"]=sources["N01_EMA"]
        ema["val_per_class_auc_json"]=json.dumps([r["auc"] for r in class_stats(pred["N01_EMA"])])
        ema["val_per_class_ap_json"]=json.dumps([r["ap"] for r in class_stats(pred["N01_EMA"])])
        inv.append(ema)
# Frozen-feature heads are real experiments, but their tiny head checkpoints
# cannot run without M02's backbone. Count both files for deployment.
for name in ("V00", "V01", "V02", "V03"):
    base = EXP / ("v03_m02_multiview_residual" if name == "V03" else "v00_v01_v02_m02_frozen_features")
    run = base if name == "V03" else base / name.lower()
    ck, p = run / "best_head.pt", run / "study_predictions.csv"
    if not ck.exists() or not p.exists(): continue
    frame = read_studies(p)
    values = aligned(frame, truth)
    label_cols = [f"label_{i}" for i in range(1,11)]
    assert np.array_equal(frame.loc[truth.index,label_cols].to_numpy(int),y)
    summary_path = run / "summary.json" if name != "V03" else run / "comparison.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    epoch = summary_data.get("best_epoch", "UNAVAILABLE")
    inv.append({"model_id":name,"run_dir":str(run.relative_to(ROOT)),"architecture":"M02 frozen features + "+name+" head",
      "config_path":str((base/"config.yaml").relative_to(ROOT)),"best_epoch":epoch,
      "val_macro_auc":macro(values),
      "val_per_class_auc_json":json.dumps([roc_auc_score(y[:,j],values[:,j]) for j in range(10)]),
      "val_per_class_ap_json":json.dumps([average_precision_score(y[:,j],values[:,j]) for j in range(10)]),
      "val_positive_studies_json":json.dumps(y.sum(axis=0).tolist()),
      "oof_studies":0,"oof_macro_auc":"N/A","oof_scope":"N/A",
      "official_val_reused_for_early_stopping":True,"train_scope":"official Train frozen M02 features",
      "input_size":384,"preprocessing":"M02 black_pad_v1; deterministic cached features",
      "study_aggregation":name+" feature head", "weight_bytes":ck.stat().st_size+(EXP/RUNS["M02"]/"best_model.pt").stat().st_size,
      "inference_model_count":1,
      "checkpoint_sha256":"M02 backbone: "+sha(EXP/RUNS["M02"]/"best_model.pt")+"; head: "+sha(ck),
      "val_prediction_source":str(p.relative_to(ROOT)),"oof_prediction_source":"N/A",
      "target_latency_status":"LATENCY_UNVERIFIED","official_score":"UNAVAILABLE"})
score = {"E01":"75.51 (user-reported)","E03":"76.45 (user-reported)",
         "E07":"76.65 (user-reported)","E08_EMA":"77.75 (user-reported)"}
members = {"E01":["B01","B02","B03"],"E03":["B01","B02","B03","M02"],
           "E07":["B01","B02","B03","M02","T01"],
           "E08_EMA":["B01","B02","B03","M02","N01_EMA"],
           "E17-F1":["D17-F1","C17-F1","S17-F1"],
           "E17-F2":["D17-F2","C17-F2","S17-F2"]}
sizes={r["model_id"]:r["weight_bytes"] for r in inv}
for name, members_ in members.items():
    if name in pred: auc = macro(pred[name])
    elif name == "E17-F1": auc = 0.8068890332710275
    else: auc = 0.79890
    inv.append({"model_id":name,"run_dir":"frozen probability ensemble", "architecture":" + ".join(members_),
      "config_path":sources.get(name,"outputs/analysis/phase17_fold2_fixed_fusion_20261020/fold2_summary.json"),
      "best_epoch":"N/A","val_macro_auc":auc,
      "val_per_class_auc_json":json.dumps([r["auc"] for r in class_stats(pred[name])]) if name in pred else "UNAVAILABLE",
      "val_per_class_ap_json":json.dumps([r["ap"] for r in class_stats(pred[name])]) if name in pred else "UNAVAILABLE",
      "val_positive_studies_json":json.dumps(y.sum(axis=0).tolist()),
      "oof_studies":3687 if name=="E17-F1" else 3731 if name=="E17-F2" else 0,
      "oof_macro_auc":0.8068045229571222 if name=="E17-F1" else 0.80798 if name=="E17-F2" else "N/A",
      "oof_scope":"single frozen patient fold" if name.startswith("E17-") else "N/A",
      "official_val_reused_for_early_stopping":True,"train_scope":"see component models",
      "input_size":384,"preprocessing":"black_pad_v1; see component models",
      "study_aggregation":"mean_probability then fixed weighted mean",
      "weight_bytes":sum(sizes.get(m,0) for m in members_),
      "inference_model_count":len(members_),
      "checkpoint_sha256":"see component model rows","val_prediction_source":sources.get(name,"see Fold fusion report"),
      "oof_prediction_source":"see Fold fusion report" if name.startswith("E17-") else "N/A",
      "target_latency_status":"LATENCY_UNVERIFIED","official_score":score.get(name,"UNAVAILABLE")})
pd.DataFrame(inv).to_csv(OUT/"experiment_inventory.csv",index=False,encoding="utf-8")


# Patient-fold integrity: check all six OOF files against immutable split IDs.
fold_dir=AN/"phase17_patient_folds"
study_index=pd.read_csv(fold_dir/"study_folds.csv",dtype=str).set_index("Study_id")
image_index=pd.read_csv(fold_dir/"image_folds.csv",dtype=str)
patient_index=pd.read_csv(fold_dir/"patient_folds.csv",dtype=str)
assert patient_index.Subject_id.is_unique and study_index.index.is_unique
assert image_index.Dicom_id.is_unique and image_index.archive_member.is_unique
image_join=image_index.join(study_index[["Subject_id","fold"]],on="Study_id",rsuffix="_study",validate="many_to_one")
assert image_join.Subject_id.equals(image_join.Subject_id_study)
assert image_join.fold.equals(image_join.fold_study)
patient_join=study_index.reset_index().join(patient_index.set_index("Subject_id")[["fold"]],on="Subject_id",rsuffix="_patient",validate="many_to_one")
assert patient_join.fold.equals(patient_join.fold_patient)
oof_checks=[]; by_fold={}
for fold in (1,2):
    expected=study_index[study_index.fold.astype(int)==fold]
    by_fold[fold]=expected
    for backbone in ("d","c","s"):
        run=EXP/f"{backbone}17_f{fold}_202610{19 if fold==1 else 20}"
        p=run/"oof_analysis"/"study_predictions.csv"
        df=read_studies(p,labels=True)
        good=set(df.index)==set(expected.index) and df.Subject_id.equals(expected.loc[df.index].Subject_id)
        oof_checks.append({"model":run.name,"fold":fold,"study_count":len(df),
                           "patient_count":df.Subject_id.nunique(),"matches_frozen_fold":bool(good),
                           "source":str(p.relative_to(ROOT))})
assert all(r["matches_frozen_fold"] for r in oof_checks)
assert not (set(by_fold[1].index)&set(by_fold[2].index))
assert not (set(by_fold[1].Subject_id)&set(by_fold[2].Subject_id))
partial_pred=[];partial_y=[]
for fold in (1,2):
    model_frames=[read_studies(EXP/f"{m}17_f{fold}_202610{19 if fold==1 else 20}"/
                               "oof_analysis"/"study_predictions.csv",labels=True)
                  for m in ("d","c","s")]
    ids=model_frames[0].index
    assert all(f.index.equals(ids) and f.Subject_id.equals(model_frames[0].Subject_id) and
               np.array_equal(f[TARGETS].to_numpy(int),model_frames[0][TARGETS].to_numpy(int))
               for f in model_frames[1:])
    partial_pred.append(.25*model_frames[0][PROBS].to_numpy(float)+
                        .25*model_frames[1][PROBS].to_numpy(float)+.5*model_frames[2][PROBS].to_numpy(float))
    partial_y.append(model_frames[0][TARGETS].to_numpy(int))
partial_pred=np.concatenate(partial_pred);partial_y=np.concatenate(partial_y)
partial_auc=float(np.mean([roc_auc_score(partial_y[:,j],partial_pred[:,j]) for j in range(10)]))
audit=json.loads((ROOT/"outputs"/"audits"/"data_audit.json").read_text(encoding="utf-8"))
oof_report={"fold_index_sha256":{p.name:sha(p) for p in fold_dir.glob("*folds.csv")},
 "frozen_fold_source":str((fold_dir/"fold_report.json").relative_to(ROOT)),
 "official_train_studies":len(study_index),"official_train_patients":patient_index.Subject_id.nunique(),
 "image_index_rows":len(image_index),"folds_checked":[1,2],"model_checks":oof_checks,
 "patient_study_image_fold_mapping_valid":True,
 "fold1_fold2_patient_overlap":len(set(by_fold[1].Subject_id)&set(by_fold[2].Subject_id)),
 "fold1_fold2_study_overlap":len(set(by_fold[1].index)&set(by_fold[2].index)),
 "partial_oof_studies":len(partial_y),"partial_oof_auc":partial_auc,
 "partial_oof_warning":"Only Folds 1 and 2; never call this complete five-fold OOF",
 "complete_five_fold_oof_status":"INCOMPLETE: Folds 3-5 absent",
 "data_audit_source":"outputs/audits/data_audit.json"}
(OUT/"oof_integrity_report.json").write_text(json.dumps(oof_report,indent=2),encoding="utf-8")


# Target-device latency is absent. Local RTX 4060 readings are diagnostic only.
bench_n=json.loads((EXP/RUNS["N01"]/"inference_benchmark.json").read_text(encoding="utf-8"))
bench_t=json.loads((EXP/RUNS["T01"]/"inference_benchmark.json").read_text(encoding="utf-8"))
local_ms={"N01":bench_n["results"]["N01_raw"]["latency_ms_mean"],
          "N01_EMA":bench_n["results"]["N01_EMA"]["latency_ms_mean"],
          "E08_EMA":bench_n["results"]["E08_all_five_models"]["latency_ms_mean"],
          "M02":bench_t["results"]["M02"]["latency_ms_mean"],
          "T01":bench_t["results"]["T01"]["latency_ms_mean"]}
deploy=[]
for row in inv:
    name=row["model_id"]; size=row["weight_bytes"]
    if not isinstance(size,(int,np.integer)) or size<=0:continue
    local=local_ms.get(name)
    if name=="E07":local=None # Component timings cannot stand in for ensemble timing.
    deploy.append({"candidate":name,"val_macro_auc":row["val_macro_auc"],
      "weight_bytes":size,"weight_mb_decimal":size/1e6,
      "size_status":"PASS" if size<=500_000_000 else "SIZE_FAIL",
      "local_4060_mean_ms":local if local is not None else "UNAVAILABLE",
      "local_4060_scope":"ZIP decode + resize/pad + H2D + FP16 forward + sigmoid + CPU copy" if local is not None else "UNAVAILABLE",
      "target_hardware_ms":"UNAVAILABLE","latency_status":"LATENCY_UNVERIFIED",
      "deployment_status":"SIZE_FAIL" if size>500_000_000 else "LATENCY_UNVERIFIED",
      "weights_included": "+".join(members.get(name,[name])),
      "source":row["run_dir"]})
pd.DataFrame(deploy).to_csv(OUT/"deployment_feasibility.csv",index=False)


summary={"validation_studies":len(y),"validation_patients":len(patient_ids),
 "e07_auc":macro(pred["E07"]),"e08_auc":macro(pred["E08_EMA"]),
 "e08_saved_vs_reconstructed_max_abs_error":e08_diff,
 "per_class_e08":class_stats(pred["E08_EMA"]),
 "bootstrap":bootstrap,"partial_oof_auc":partial_auc,
 "source_paths":sources,"official_scores_user_reported":score,
 "data_audit_path":"outputs/audits/data_audit.json"}
(OUT/"computed_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({"E07":summary["e07_auc"],"E08":summary["e08_auc"],
                  "delta_ci":bootstrap["ci95_percentile"],"invalid":invalid,
                  "partial_oof":partial_auc,"inventory_rows":len(inv)},indent=2))
