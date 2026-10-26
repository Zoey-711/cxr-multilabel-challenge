"""Audit frozen Phase 15 artifacts, E03 history, and N01/M02 differences."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.config)
    control = load_config(ROOT / "outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated/config.yaml")
    differences = []
    for section in control:
        for key in sorted(set(control[section]) | set(config[section])):
            if control[section].get(key) != config[section].get(key):
                differences.append({"section": section, "key": key,
                                    "m02": control[section].get(key), "n01": config[section].get(key)})
    if {(d["section"], d["key"]) for d in differences} != {
            ("experiment", "name"), ("model", "name"), ("model", "pretrained_sha256"),
            ("training", "ema")}:
        raise ValueError("N01 includes unapproved differences from M02")
    phase15 = ROOT / "outputs/analysis/phase15_formal_20261018/pipeline"
    holdout = phase15 / "holdout"
    holdout_manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    if digest(phase15 / "selection_lock.json") != holdout_manifest["selection_lock_sha256"]:
        raise ValueError("Frozen Phase 15 selection lock hash mismatch")
    for name, expected in holdout_manifest["files_sha256"].items():
        if digest(holdout / name) != expected:
            raise ValueError(f"Frozen Holdout result changed: {name}")
    phase15_models = {}
    for label, run_id in (("A", "p15a_densenet121_internal_train_384"),
                          ("B", "p15b_convnext_tiny_internal_train_384")):
        run = ROOT / "outputs/experiments" / run_id
        manifest = json.loads((run / "phase15_freeze_manifest.json").read_text(encoding="utf-8"))
        for filename, key in (("best_model.pt", "checkpoint_sha256"),
                              ("config.yaml", "config_sha256"),
                              ("validation_analysis/study_predictions.csv", "validation_predictions_sha256")):
            if digest(run / filename) != manifest[key]:
                raise ValueError(f"Frozen P15{label} hash mismatch: {filename}")
        phase15_models[label] = {"run_id": run_id, "checkpoint_sha256": manifest["checkpoint_sha256"]}
    e03 = ROOT / "outputs/submissions/E03_20261017"
    e03_manifest = json.loads((e03 / "manifest.json").read_text(encoding="utf-8"))
    csv_hash = digest(e03 / "submission.csv")
    if csv_hash != e03_manifest["csv_sha256"]:
        raise ValueError("E03 CSV differs from frozen generated manifest")
    with (ROOT / "outputs/submissions/submission_history.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    entries = [entry for entry in history if entry["version"] == "E03_20261017"]
    if len(entries) != 1 or entries[0]["csv_sha256"] != csv_hash:
        raise ValueError("E03 history row does not match generated CSV")
    preflight = json.loads((ROOT / "outputs/experiments/n01_efficientnet_v2_s_384_preflight_20261018/status.json").read_text(encoding="utf-8"))
    if preflight["state"] != "preflight_passed" or preflight["optimizer_steps"] != 5 or \
            preflight["amp_skipped_steps"] != 0 or preflight["ema_updates"] != 5:
        raise ValueError("N01 AMP/EMA GPU preflight did not pass")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "phase15_frozen": {"models": phase15_models, "holdout_files_verified": len(holdout_manifest["files_sha256"]),
                                 "holdout_reuse_policy": "No new model selection, fusion search, or hyperparameter tuning"},
              "e03": {"candidate_csv_sha256": csv_hash, "history_status": entries[0]["upload_status"],
                      "manifest_status": e03_manifest["upload_status"],
                      "user_reported_leaderboard_display_score": "76.45",
                      "verified_uploaded_csv_sha256": None,
                      "upload_evidence": "No platform upload record, timestamp/status, or platform-side file hash was supplied; keep generated"},
              "n01_control_differences": differences,
              "n01_preflight": preflight,
              "n01_pretrained_source_sha256": config["model"]["pretrained_sha256"]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase15_hashes_valid": True, "e03_uploaded_status": entries[0]["upload_status"],
                      "n01_control_differences": [d["key"] for d in differences]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
