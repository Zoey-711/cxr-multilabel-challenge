"""Recompute delivered Val/OOF metrics and fixed E17 on CPU, never Test AUC."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from cxr.batch.results import bootstrap, combine, concatenate_oof, grouped, scores
from import_phase17_friend_results import label_keys, prediction_keys


def table(path: Path, labels_path: Path, expected_keys):
    prediction_keys(path, expected_keys)
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {(r["Subject_id"], r["Study_id"]): r for r in csv.DictReader(handle)}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["Subject_id"], r["Study_id"]) for r in rows]
    return {"keys": keys,
            "p": np.asarray([[float(r[f"Prob_{i}"]) for i in range(1, 11)] for r in rows]),
            "y": np.asarray([[int(labels[key][f"Target_{i}"]) for i in range(1, 11)] for key in keys]),
            "counts": np.asarray([int(labels[key]["image_count"]) for key in keys])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen friend's batch_plan.yaml")
    parser.add_argument("--root", required=True, type=Path, help="Verified local import folder")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = json.loads((root / "IMPORT_VERIFIED.json").read_text(encoding="utf-8"))
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if receipt["state"] != "VERIFIED" or receipt["source_commit"] != json.loads(
            (root / "manifests/delivery_manifest.json").read_text(encoding="utf-8"))["git_commit"]:
        raise ValueError("Result folder lacks matching import verification")
    output = root / "analysis/local_recomputed"
    if output.exists():
        raise FileExistsError(output)
    val_labels = root / "manifests/val_labels.csv"
    train_labels = root / "manifests/train_labels.csv"
    val_keys = label_keys(val_labels)
    train_keys = label_keys(train_labels)
    metrics = {"models": {}, "complete_oof": {}, "val_ensembles": {},
               "bootstrap": None, "test_auc": None,
               "caution": "Official Val was repeatedly used for epoch selection; it is not an independent test set."}
    val_tables, oof_tables = {}, {}
    model_paths = {}
    task_map = {task["id"]: task for task in plan["tasks"]}
    selection = root / "manifests/extension_selection.json"
    if selection.is_file():
        task_map.update({task["id"]: task for task in json.loads(selection.read_text(encoding="utf-8"))["tasks"]})
    for task_id in receipt["completed_tasks"]:
        task = task_map[task_id]
        run = root / "experiments" / task_id
        raw = table(run / "val_predictions.csv", val_labels, val_keys)
        val_tables[task_id] = raw
        model_paths[task_id] = run / "best_model.pth"
        entry = {"val": grouped(raw), "oof": "N/A"}
        if task.get("fold") is not None:
            held_out = table(run / "oof_predictions.csv", train_labels,
                             label_keys(train_labels, int(task["fold"])))
            oof_tables[task_id] = held_out
            entry["oof"] = grouped(held_out)
        metrics["models"][task_id] = entry
        if task.get("ema"):
            ema_id = task_id + "_EMA"
            ema_val = table(run / "ema_val_predictions.csv", val_labels, val_keys)
            val_tables[ema_id] = ema_val
            model_paths[ema_id] = run / "ema_best_model.pth"
            ema_entry = {"val": grouped(ema_val), "oof": "N/A"}
            if task.get("fold") is not None:
                ema_oof = table(run / "ema_oof_predictions.csv", train_labels,
                                label_keys(train_labels, int(task["fold"])))
                oof_tables[ema_id] = ema_oof
                ema_entry["oof"] = grouped(ema_oof)
            metrics["models"][ema_id] = ema_entry
    for branch in ("D17", "C17", "S17", "N17"):
        ids = [f"{branch}_F{i:02d}" for i in range(1, 6)]
        if all(i in val_tables and i in oof_tables for i in ids):
            val_tables[branch + "_5fold"] = combine([val_tables[i] for i in ids], [0.2] * 5)
            whole_oof = concatenate_oof([oof_tables[i] for i in ids])
            if set(whole_oof["keys"]) != train_keys:
                raise ValueError(f"Complete OOF coverage mismatch: {branch}")
            metrics["complete_oof"][branch] = grouped(whole_oof)
            metrics["val_ensembles"][branch] = grouped(val_tables[branch + "_5fold"])
        else:
            metrics["complete_oof"][branch] = "INCOMPLETE"
        ema_ids = [i + "_EMA" for i in ids]
        if all(i in val_tables and i in oof_tables for i in ema_ids):
            val_tables[branch + "_EMA_5fold"] = combine([val_tables[i] for i in ema_ids], [0.2] * 5)
            ema_whole_oof = concatenate_oof([oof_tables[i] for i in ema_ids])
            if set(ema_whole_oof["keys"]) != train_keys:
                raise ValueError(f"Complete EMA OOF coverage mismatch: {branch}")
            metrics["complete_oof"][branch + "_EMA"] = grouped(ema_whole_oof)
            metrics["val_ensembles"][branch + "_EMA"] = grouped(val_tables[branch + "_EMA_5fold"])
    if all(branch + "_5fold" in val_tables for branch in ("D17", "C17", "S17")):
        e17 = combine([val_tables[b + "_5fold"] for b in ("D17", "C17", "S17")], [0.25, 0.25, 0.50])
        metrics["val_ensembles"]["E17"] = grouped(e17)
        e07_path = root / "validation/E07_val_probabilities.csv"
        if e07_path.is_file():
            e07 = table(e07_path, val_labels, val_keys)
            metrics["val_ensembles"]["E07"] = grouped(e07)
            metrics["bootstrap"] = bootstrap({"E17": e17, "E07": e07},
                                             [("E17", "E07")], draws=args.bootstrap_draws, seed=42)
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model_ID", "Val_Macro_AUC", "OOF_Macro_AUC", "OOF_Status", "Test_AUC"])
        for name, record in metrics["models"].items():
            oof = record["oof"]
            writer.writerow([name, record["val"]["all"]["macro_auc"],
                             oof["all"]["macro_auc"] if isinstance(oof, dict) else "N/A",
                             "PASS" if isinstance(oof, dict) else "N/A", "N/A"])
    per_class = []
    for name, record in metrics["models"].items():
        for split in ("val", "oof"):
            grouped_result = record[split]
            if isinstance(grouped_result, dict):
                for item in grouped_result["all"]["per_class"]:
                    per_class.append({"Model_ID": name, "Split": split, **item})
    for name, record in metrics["val_ensembles"].items():
        for item in record["all"]["per_class"]:
            per_class.append({"Model_ID": name, "Split": "val", **item})
    if per_class:
        with (output / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
    deployment = []
    presets = {name: [name] for name in model_paths}
    for branch in ("D17", "C17", "S17", "N17"):
        ids = [f"{branch}_F{i:02d}" for i in range(1, 6)]
        if all(i in model_paths for i in ids):
            presets[branch + "_5fold"] = ids
        ema_ids = [i + "_EMA" for i in ids]
        if all(i in model_paths for i in ema_ids):
            presets[branch + "_EMA_5fold"] = ema_ids
    if all(x in presets for x in ("D17_5fold", "C17_5fold", "S17_5fold")):
        presets["E17"] = (presets["D17_5fold"] + presets["C17_5fold"] +
                          presets["S17_5fold"])
    for name, dependencies in presets.items():
        bytes_used = sum(model_paths[dependency].stat().st_size for dependency in dict.fromkeys(dependencies))
        val = metrics["models"].get(name, {}).get("val", {})
        val = metrics["val_ensembles"].get(name, val)
        deployment.append({"Candidate_ID": name, "Source_Models": ";".join(dependencies),
                           "Val_Macro_AUC": val.get("all", {}).get("macro_auc"),
                           "Weight_Bytes": bytes_used,
                           "Size_Status": "PASS" if bytes_used <= 500_000_000 else "SIZE_FAIL",
                           "Official_Target_Latency_MS": None,
                           "Latency_Status": "LATENCY_UNVERIFIED",
                           "Deployment_Status": "SIZE_FAIL" if bytes_used > 500_000_000 else "LATENCY_UNVERIFIED"})
    with (output / "deployment_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deployment[0]) if deployment else
                                ["Candidate_ID", "Source_Models", "Val_Macro_AUC", "Weight_Bytes",
                                 "Size_Status", "Official_Target_Latency_MS", "Latency_Status", "Deployment_Status"])
        writer.writeheader()
        writer.writerows(deployment)
    print(json.dumps({"models": len(metrics["models"]), "full_oof": metrics["complete_oof"].keys(),
                      "e17_val_auc": metrics["val_ensembles"].get("E17", {}).get("all", {}).get("macro_auc"),
                      "test_auc": None}, ensure_ascii=False, default=list))


if __name__ == "__main__":
    main()
