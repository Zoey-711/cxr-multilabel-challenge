"""Run the approved Phase 15 models sequentially, then lock and score Holdout once."""

import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from cxr.engine.metrics import binary_auc
from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def require_complete(run: Path, *, pid: int | None = None) -> None:
    while True:
        status_file = run / "status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("state") == "complete":
                return
            if status.get("state") == "failed":
                raise RuntimeError(f"Run failed: {run}: {status.get('error')}")
        if pid is not None:
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                    capture_output=True, text=True, check=True)
            if str(pid) not in result.stdout and status_file.is_file():
                raise RuntimeError(f"Run process {pid} disappeared before completion: {run}")
        time.sleep(30)


def run_command(args: list[str], stdout: Path, stderr: Path) -> None:
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")


def verify_and_freeze(label: str, item: dict, audit: Path) -> dict:
    run = ROOT / "outputs" / "experiments" / item["run_id"]
    config = ROOT / item["config"]
    require_complete(run)
    actual_config = load_config(run / "config.yaml")
    if actual_config != load_config(config):
        raise ValueError(f"{label} runtime config mismatch")
    source = json.loads((run / "pretrained_source.json").read_text(encoding="utf-8"))
    if source["sha256"] != item["pretrained_sha256"]:
        raise ValueError(f"{label} pretrained SHA mismatch")
    validation = run / "validation_analysis"
    if validation.exists():
        raise FileExistsError(f"Independent validation output already exists: {validation}")
    run_command(["scripts/analyze_validation.py", "--config", item["config"],
                 "--run-dir", str(run)], audit / f"{label}_validation_stdout.log",
                audit / f"{label}_validation_stderr.log")
    analysis = json.loads((validation / "analysis.json").read_text(encoding="utf-8"))
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if (analysis["study_count"] != 859 or not analysis["best_metrics_crosscheck_passed"] or
            analysis["best_epoch"] != best["epoch"] or status["best_epoch"] != best["epoch"] or
            len(rows) != status["stop_epoch"]):
        raise ValueError(f"{label} checkpoint or log cross-check failed")
    manifest = {
        "label": label, "run_id": item["run_id"], "state": "frozen",
        "pretrained_sha256": source["sha256"], "checkpoint_sha256": digest(run / "best_model.pt"),
        "config_sha256": digest(run / "config.yaml"),
        "validation_predictions_sha256": digest(validation / "study_predictions.csv"),
        "best_epoch": best["epoch"], "stop_epoch": status["stop_epoch"],
        "stop_reason": status["stop_reason"], "official_val_macro_auc": analysis["macro_auc"],
        "official_val_macro_f1": analysis["macro_f1"],
        "training_seconds": sum(row["elapsed_seconds"] for row in rows),
        "peak_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in rows),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_new_json(run / "phase15_freeze_manifest.json", manifest)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    return manifest


def locked_official_fusion(protocol: dict, frozen_a: dict, frozen_b: dict) -> dict:
    tables = {}
    for name in ("A", "B"):
        path = (ROOT / "outputs" / "experiments" / protocol["runs"][name]["run_id"] /
                "validation_analysis" / "study_predictions.csv")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 859:
            raise ValueError(f"{name} official validation does not contain 859 Studies")
        keys = [(row["Subject_id"], row["Study_id"]) for row in rows]
        if len(set(keys)) != 859:
            raise ValueError(f"{name} duplicate official validation Study")
        tables[name] = {key: row for key, row in zip(keys, rows)}
    if set(tables["A"]) != set(tables["B"]):
        raise ValueError("Official validation Study keys differ")
    keys = sorted(tables["A"])
    a, b, y = [], [], []
    for key in keys:
        one, two = tables["A"][key], tables["B"][key]
        target_a = [int(one[f"target_{j}"]) for j in range(1, 11)]
        target_b = [int(two[f"target_{j}"]) for j in range(1, 11)]
        if target_a != target_b:
            raise ValueError(f"Official validation labels differ: {key}")
        a.append([float(one[f"probability_{j}"]) for j in range(1, 11)])
        b.append([float(two[f"probability_{j}"]) for j in range(1, 11)])
        y.append(target_a)
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite official validation probability")

    def macro(p: np.ndarray) -> float:
        return float(np.mean([binary_auc(y[:, j], p[:, j]) for j in range(10)]))

    if abs(macro(a) - frozen_a["official_val_macro_auc"]) > 1e-8 or \
            abs(macro(b) - frozen_b["official_val_macro_auc"]) > 1e-8:
        raise ValueError("Independent official validation metrics differ from frozen predictions")
    return {"study_count": 859, "patient_count": len({subject for subject, _ in keys}),
            "A_macro_auc": macro(a), "B_macro_auc": macro(b),
            "fixed_fusion_macro_auc": macro(0.5 * a + 0.5 * b),
            "fusion_weights": {"A": 0.5, "B": 0.5}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--p15a-pid", type=int, required=True,
                        help="PID of the already-running approved P15A process")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if protocol["phase"] != 15 or protocol["fixed_fusion"] != {
            "p15a_densenet121": 0.5, "p15b_convnext_tiny": 0.5,
            "level": "study_mean_probability"}:
        raise ValueError("Phase 15 fixed protocol differs from approval")
    audit = args.output_dir.resolve()
    if audit.exists():
        raise FileExistsError(audit)
    audit.mkdir(parents=True)
    run_a = ROOT / "outputs" / "experiments" / protocol["runs"]["A"]["run_id"]
    run_b = ROOT / "outputs" / "experiments" / protocol["runs"]["B"]["run_id"]
    if not run_a.is_dir() or run_b.exists():
        raise ValueError("Expected running P15A and absent P15B output")
    print("WAIT_P15A", flush=True)
    require_complete(run_a, pid=args.p15a_pid)
    frozen_a = verify_and_freeze("A", protocol["runs"]["A"], audit)
    print(f"P15A_FROZEN epoch={frozen_a['best_epoch']} auc={frozen_a['official_val_macro_auc']:.5f}", flush=True)
    run_b.mkdir()
    print("START_P15B", flush=True)
    run_command(["scripts/train.py", "--config", protocol["runs"]["B"]["config"],
                 "--run-id", protocol["runs"]["B"]["run_id"]],
                run_b / "train_stdout.log", run_b / "train_stderr.log")
    require_complete(run_b)
    frozen_b = verify_and_freeze("B", protocol["runs"]["B"], audit)
    print(f"P15B_FROZEN epoch={frozen_b['best_epoch']} auc={frozen_b['official_val_macro_auc']:.5f}", flush=True)
    official_fusion = locked_official_fusion(protocol, frozen_a, frozen_b)
    write_new_json(audit / "official_validation_fixed_fusion.json", official_fusion)
    lock = {
        "phase": 15, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(args.config.resolve()),
        "split_sha256": load_config(ROOT / protocol["runs"]["A"]["config"])["data"]["train_subset"]["sha256"],
        "models": {"A": frozen_a, "B": frozen_b},
        "official_validation_fixed_fusion": official_fusion,
        "fusion": {"A": 0.5, "B": 0.5}, "aggregation": "mean_image_sigmoid_probability",
        "threshold": 0.5, "holdout_use": "one_final_evaluation_only",
    }
    write_new_json(audit / "selection_lock.json", lock)
    (audit / "selection_lock.json").chmod(stat.S_IREAD)
    print("SELECTION_LOCKED; START_HOLDOUT_ONCE", flush=True)
    run_command(["scripts/evaluate_phase15_holdout.py", "--config", str(args.config.resolve()),
                 "--lock", str(audit / "selection_lock.json"),
                 "--output-dir", str(audit / "holdout")],
                audit / "holdout_stdout.log", audit / "holdout_stderr.log")
    result = json.loads((audit / "holdout" / "result.json").read_text(encoding="utf-8"))
    scores, intervals = result["scores"], result["patient_bootstrap"]
    lines = ["# Phase 15 independent-patient generalization", "",
             "## Internal patient Holdout: one final evaluation", "",
             f"{result['holdout_patients']} patients, {result['holdout_studies']} Studies, "
             f"{result['holdout_images']} images. These patients were excluded from both new training runs.",
             "", "| Model | Holdout Macro-AUC | 95% patient bootstrap CI | Macro-AP | Macro-F1@0.5 |",
             "|---|---:|---:|---:|---:|"]
    for name in ("A", "B", "fusion"):
        row = scores[name]
        ci = intervals["individual_macro_auc_ci95"][name]
        lines.append(f"| {name} | {row['macro_auc']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] | "
                     f"{row['macro_ap']:.5f} | {row['macro_f1_at_0_5']:.5f} |")
    lines.extend(["", f"Bootstrap: {intervals['valid_replicates']} valid paired patient resamples; "
                  f"{intervals['invalid_resamples']} excluded invalid draws; seed {intervals['seed']}.", ""])
    for comparison, point in result["paired_point_differences"].items():
        ci = intervals["paired_difference_ci95"][comparison]
        lines.append(f"- {comparison}: {point:+.5f}; 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}].")
    lines.extend(["", "Per-class Holdout AUC/AP and positive patient support for all ten labels, "
                  "plus single-image and multi-image Study results, are in `holdout/result.json`. "
                  "Group metrics are descriptive because label composition may differ.", "",
                  "## Official validation: checkpoint selection", "",
                  f"A best epoch {frozen_a['best_epoch']}, Macro-AUC {frozen_a['official_val_macro_auc']:.5f}; "
                  f"B best epoch {frozen_b['best_epoch']}, Macro-AUC {frozen_b['official_val_macro_auc']:.5f}; "
                  f"fixed fusion Macro-AUC {official_fusion['fixed_fusion_macro_auc']:.5f} "
                  "on the same 859 official validation Studies. These are selection-set results, "
                  "not independent-patient Holdout estimates.", "",
                  "## Existing leaderboard result", "",
                  "E01 previously received displayed official score 75.51. No Phase 15 model was run "
                  "on hidden test data or submitted. The displayed leaderboard score is separate "
                  "from both local validation and internal Holdout metrics.", ""])
    with (audit / "final_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    write_new_json(audit / "phase15_complete.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_result_sha256": digest(audit / "holdout" / "result.json"),
        "selection_lock_sha256": digest(audit / "selection_lock.json"),
        "final_report_sha256": digest(audit / "final_report.md"),
    })
    for path in (audit / "holdout").iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)
    print("PHASE15_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
