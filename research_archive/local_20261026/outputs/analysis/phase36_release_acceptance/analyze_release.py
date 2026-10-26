"""Read-only Phase32/E08-Tune Release acceptance and patient-cluster analysis."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
CODE = WORKSPACE / "cxr-phase32-friend"
sys.path.insert(0, str(CODE / "src"))

from cxr.batch.results import align, bootstrap, concatenate_oof, read_table, scores
from cxr.phase35.analysis import validate_oof

P32_COMMIT = "9fd6afc68b904ac4cf7f319053f09023507ca5ee"
E08_COMMIT = "19c8d7e083aba66bb842373c364537d3aac61a37"
BASE_COMMIT = "cbfc5997438f371086cd1fbe232890590cd67ffb"
P32 = WORKSPACE / "cxr-friend-results" / "versions" / P32_COMMIT / "results/phase32_batch"
E08 = WORKSPACE / "cxr-friend-results" / "versions" / E08_COMMIT / "results/e08_tuning"
BASE = WORKSPACE / "cxr-friend-results" / "versions" / BASE_COMMIT / "results/phase17_batch"
FOLDS = WORKSPACE / "cxr-multilabel-challenge/outputs/analysis/phase17_patient_folds/study_folds.csv"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verify_text_tree(root):
    manifest = load(root / "export_manifest.json")
    missing, wrong = [], []
    for rel, expected in manifest["sha256"].items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != expected:
            wrong.append(rel)
    if missing or wrong:
        raise ValueError(f"Text-result SHA mismatch {root}: missing={missing[:5]}, changed={wrong[:5]}")
    delivery = load(root / "github_delivery_manifest.json")
    if delivery["source_commit"] != "01fa68bdac65ce56d04a1fc99e8f5996ab09aa20":
        raise ValueError("Training source commit mismatch")
    return {"commit": P32_COMMIT if root == P32 else E08_COMMIT,
            "verified_small_files": len(manifest["sha256"]), "missing": [], "wrong_sha256": []}


def table(run, ema=True):
    prefix = "ema_" if ema else ""
    return read_table(run / (prefix + "oof_analysis") / "study_predictions.csv")


def main():
    release = load(HERE / "release_api.json")
    binary = load(HERE / "binary_release_manifest.json")
    asset_index = {a["name"]: a for a in release["assets"]}
    if release["tagName"] != binary["release_tag"] or release["targetCommitish"] != "01fa68bdac65ce56d04a1fc99e8f5996ab09aa20":
        raise ValueError("Wrong Release/tag/source commit")
    if release["isDraft"] or release["isPrerelease"]:
        raise ValueError("Release is not final")
    if len(binary["files"]) != binary["file_count"] or len(release["assets"]) != len(binary["files"]) + 1:
        raise ValueError("Release binary attachment count mismatch")
    for b in binary["files"]:
        a = asset_index[b["asset_name"]]
        if a["state"] != "uploaded" or a["size"] != b["bytes"] or a["digest"] != "sha256:" + b["sha256"]:
            raise ValueError("Release asset hash/size/state mismatch: " + b["asset_name"])
    if sha(HERE / "binary_release_manifest.json") != asset_index["binary_release_manifest.json"]["digest"].split(":")[1]:
        raise ValueError("Release manifest download differs from GitHub digest")
    release_check = {"repository": binary["repository"], "tag": release["tagName"],
                     "target_commit": release["targetCommitish"], "asset_count": len(asset_index),
                     "binary_file_count": len(binary["files"]), "binary_total_bytes": binary["bytes"],
                     "binary_assets_verified_by_api_digest": len(binary["files"]),
                     "lightweight_assets_on_release": 1,
                     "phase35_binary_assets": sum("phase35" in b["path"].lower() for b in binary["files"]),
                     "text_sources": [verify_text_tree(P32), verify_text_tree(E08)]}
    dump(HERE / "release_integrity.json", release_check)
    write_rows("release_assets.csv", [{"name": a["name"], "size_bytes": a["size"], "state": a["state"],
                                      "sha256": a["digest"].split(":")[-1]} for a in release["assets"]])

    folds = list(csv.DictReader(FOLDS.open(encoding="utf-8-sig", newline="")))
    if sha(FOLDS) != "d92fac966dece9f797e0cf323100be1bfc35da698473c1d36016df8350b62edf":
        raise ValueError("Frozen fold index changed")

    p32_plan = yaml.safe_load((CODE / "configs/phase32_plan.yaml").read_text(encoding="utf-8"))
    pilots = []
    for task in p32_plan["tasks"]:
        run = P32 / "experiments" / task["id"]
        done = load(run / "DONE.json")
        if done["state"] != "DONE":
            raise ValueError("Incomplete Phase32 Pilot: " + task["id"])
        ema = task["ema"]
        t = table(run, ema=ema)
        validate_oof(t, 1, folds)
        base_run = BASE / "experiments" / task["baseline_id"]
        reference = table(base_run, ema=ema)
        validate_oof(reference, 1, folds)
        align(reference, t)
        m, b = scores(t), scores(reference)
        if any(sha(run / key) != value for key, value in done["sha256"].items()
               if (run / key).is_file()):
            raise ValueError("DONE SHA mismatch: " + task["id"])
        run_manifest = load(run / "run_manifest.json")
        if run_manifest["fold_hashes"]["study_folds.csv"] != sha(FOLDS):
            raise ValueError("Model fold provenance mismatch")
        history = list(csv.DictReader((run / "train_history.csv").open(encoding="utf-8-sig")))
        stat = load(run / "status.json")
        cls = [x["auc"] - y["auc"] for x, y in zip(m["per_class"], b["per_class"])]
        pilots.append((task["id"], t, reference, m, b, cls, run, ema, stat, history, task["baseline_id"]))

    e08_plan = yaml.safe_load((CODE / "configs/e08_tune_plan.yaml").read_text(encoding="utf-8"))
    for task in e08_plan["tasks"]:
        if task["fold"] != 1:
            continue
        run = E08 / "pilot/experiments" / task["id"]
        done = load(run / "DONE.json")
        if done["state"] != "DONE":
            raise ValueError("Incomplete E08 Pilot: " + task["id"])
        t = table(run, ema=True)
        validate_oof(t, 1, folds)
        ref = table(BASE / "experiments/N17_F01", ema=True)
        align(ref, t)
        m, b = scores(t), scores(ref)
        if any(sha(run / key) != value for key, value in done["sha256"].items()
               if (run / key).is_file()):
            raise ValueError("E08 DONE SHA mismatch: " + task["id"])
        history = list(csv.DictReader((run / "train_history.csv").open(encoding="utf-8-sig")))
        stat = load(run / "status.json")
        cls = [x["auc"] - y["auc"] for x, y in zip(m["per_class"], b["per_class"])]
        pilots.append((task["id"], t, ref, m, b, cls, run, True, stat, history, "N17_F01"))

    # Paired bootstrap uses one patient resample per candidate/baseline pair.
    # 2000 accepted draws match the frozen selection plans.
    rows, per_class, boot = [], [], {}
    for id_, t, ref, m, b, cls, run, ema, stat, history, baseline in pilots:
        ci = bootstrap({"candidate": t, "baseline": ref}, [("candidate", "baseline")],
                       draws=2000, seed=42)["differences"]["candidate_minus_baseline"]
        boot[id_] = {"comparison": baseline, "ci95": ci["ci95"], "point": ci["point"],
                     "seed": 42, "valid_draws": 2000, "scope": "Fold1 held-out patients"}
        min_class = min(cls)
        rule = p32_plan["selection"] if id_.startswith("P32_") else e08_plan["selection"]
        passes = ci["point"] >= rule["min_delta"] and ci["ci95"][0] > 0 and min_class >= -rule["max_class_regression"]
        weight = run / ("ema_best_model.pth" if ema else "best_model.pth")
        elapsed = sum(float(h["elapsed_seconds"]) for h in history) / 3600
        row = {"Experiment_ID": id_, "Comparison": baseline, "OOF_Scope": "FOLD1_ONLY",
               "Study_Count": m["studies"], "Patient_Count": m["patients"],
               "Fold1_OOF_AUC": m["macro_auc"], "Baseline_Fold1_OOF_AUC": b["macro_auc"],
               "Delta": ci["point"], "CI95_Low": ci["ci95"][0], "CI95_High": ci["ci95"][1],
               "Worst_Class_Delta": min_class, "Preset_Rule_Pass": passes,
               "Fivefold_Recommendation": "CONFIRMATION_ONLY" if passes else "NO",
               "Best_Epoch": load(run / "best_ema_metrics.json")["epoch"] if ema else stat.get("best_epoch"),
               "Training_Hours": elapsed, "Weight_Bytes": next((x["bytes"] for x in binary["files"]
                           if x["path"] == ("phase32_batch/experiments/" + id_ + "/" + weight.name)
                           or x["path"] == ("e08_tuning/pilot/experiments/" + id_ + "/" + weight.name)), None),
               "Fold_Provenance": sha(FOLDS), "Source_Prediction_SHA256": sha(run / ("ema_oof_analysis" if ema else "oof_analysis") / "study_predictions.csv")}
        rows.append(row)
        for j, (a, r) in enumerate(zip(m["per_class"], b["per_class"]), 1):
            per_class.append({"Experiment_ID": id_, "Class": j, "AUC": a["auc"], "Baseline_AUC": r["auc"],
                              "Delta": cls[j-1], "Positive_Studies": a["positive_studies"],
                              "Positive_Patients": a["positive_patients"]})
    write_rows("pilot_comparison.csv", rows)
    write_rows("pilot_per_class.csv", per_class)
    dump(HERE / "pilot_bootstrap.json", boot)

    low_runs = [E08 / "pilot/experiments/ET_DECAY_LOW_F01"] + [E08 / f"expansion/experiments/ET_DECAY_LOW_F{i:02d}" for i in range(2, 6)]
    low = [table(run, ema=True) for run in low_runs]
    baseline = [table(BASE / f"experiments/N17_F{i:02d}", ema=True) for i in range(1, 6)]
    for i, (a, b) in enumerate(zip(low, baseline), 1):
        validate_oof(a, i, folds)
        validate_oof(b, i, folds)
        align(b, a)
    whole, whole_base = concatenate_oof(low), concatenate_oof(baseline)
    confirm, confirm_base = concatenate_oof(low[1:]), concatenate_oof(baseline[1:])
    if len(whole["keys"]) != 18513 or len(set(s for s, _ in whole["keys"])) != 11055:
        raise ValueError("Incomplete E08 low-decay OOF")
    ci_all = bootstrap({"low": whole, "N17": whole_base}, [("low", "N17")], draws=2000, seed=42)
    ci_confirm = bootstrap({"low": confirm, "N17": confirm_base}, [("low", "N17")], draws=2000, seed=42)
    low_result = {"complete_fivefold": {"candidate_auc": scores(whole)["macro_auc"],
                   "baseline_auc": scores(whole_base)["macro_auc"], "bootstrap": ci_all},
                  "folds2to5_confirmation": {"candidate_auc": scores(confirm)["macro_auc"],
                   "baseline_auc": scores(confirm_base)["macro_auc"], "bootstrap": ci_confirm},
                  "fold1_was_selected": True}
    dump(HERE / "low_decay_fivefold.json", low_result)

    phase35 = yaml.safe_load((CODE / "configs/phase35_plan.yaml").read_text(encoding="utf-8"))
    phase35_rows = []
    for variant, info in phase35["reused_fold1"].items():
        run = P32 / "experiments" / info["id"]
        checked, pending = [], []
        for rel, expected in info["sha256"].items():
            path = run / rel
            if not path.exists() and rel.endswith(".pth"):
                path = HERE / "fold1_weights" / ("phase32_batch--experiments--" + info["id"] + "--" + rel)
            if path.is_file() and not (rel.endswith(".pth") and path.stat().st_size != next(
                    b["bytes"] for b in binary["files"] if b["path"] ==
                    "phase32_batch/experiments/" + info["id"] + "/" + rel)):
                if sha(path) != expected:
                    raise ValueError("Phase35 pinned Fold1 mismatch: " + str(path))
                checked.append(rel)
            else:
                pending.append(rel)
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        if cfg["data"]["train_subset"]["fold"] != 1 or cfg["data"]["train_subset"]["sha256"] != phase35["fold_hashes"]:
            raise ValueError("Phase35 Fold1 config/index mismatch")
        pretrained = cfg["model"]["pretrained"]
        phase35_rows.append({"Variant": variant, "Pilot_ID": info["id"],
                             "Status": "FOLD1_READY" if not pending else "FOLD1_WEIGHTS_PENDING_DOWNLOAD",
                             "Pinned_Files_Checked": len(checked), "Pinned_Files_Missing": ";".join(pending),
                             "ImageNet_Config": json.dumps(pretrained, ensure_ascii=False),
                             "Source_Commit": P32_COMMIT})
    write_rows("phase35_fold1_readiness.csv", phase35_rows)
    result = {"release": release_check, "pilot_count": len(rows), "pilot_rows": rows,
              "low_decay": low_result, "phase35": phase35_rows,
              "limitations": ["Fold1 pilot selection is exploratory", "Official Val reused for epoch selection",
                              "4090 research latency does not establish T4 compliance", "No Test labels used"]}
    dump(HERE / "acceptance_results.json", result)
    print(json.dumps({"pilots": len(rows), "release_binary_files": len(binary["files"]),
                      "phase35": [(x["Variant"], x["Status"]) for x in phase35_rows],
                      "low_decay_complete_auc": low_result["complete_fivefold"]["candidate_auc"]}))


if __name__ == "__main__":
    main()
