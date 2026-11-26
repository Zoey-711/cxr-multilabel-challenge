"""Verify both one-step GPU preflights without running formal retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from cxr.utils.config import load_config


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--a-run-dir", required=True, type=Path)
    parser.add_argument("--b-run-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(protocol["audit_directory"]) / "gpu_preflight.json"
    if output.exists():
        raise FileExistsError(output)
    prior = json.loads((output.parent / "preflight.json").read_text(encoding="utf-8"))
    if prior["status"] != "prepared_not_trained":
        raise ValueError("Phase 15 audit has not passed")
    records = {}
    for label, path in (("A", args.a_run_dir), ("B", args.b_run_dir)):
        settings = protocol["runs"][label]
        if not path.is_dir() or path.resolve() == Path(prior["runs"][label]["planned_output"]).resolve():
            raise ValueError(f"{label} preflight must be separate from formal output")
        config = Path(settings["config"])
        if load_config(path / "config.yaml") != load_config(config):
            raise ValueError(f"{label} preflight used a different config")
        if sha256(config) != prior["runs"][label]["retrain_config_sha256"]:
            raise ValueError(f"{label} config hash changed since audit")
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        provenance = json.loads((path / "pretrained_source.json").read_text(encoding="utf-8"))
        if status["state"] != "preflight_passed" or status["steps"] != 1 or \
                not 0 < status["train_loss"] < 100 or status["gpu_peak_mib"] <= 0 or \
                provenance["sha256"] != settings["pretrained_sha256"]:
            raise ValueError(f"{label} one-step GPU preflight failed")
        if any((path / filename).exists() for filename in ("best_model.pt", "last_training.pt", "metrics.jsonl")):
            raise ValueError(f"{label} preflight unexpectedly saved training results")
        records[label] = {"run_dir": str(path.resolve()), "train_images": prior["runs"][label]["images"],
                          "optimizer_groups_per_epoch": prior["runs"][label]["optimizer_groups_per_epoch"],
                          "micro_batch": prior["runs"][label]["micro_batch"],
                          "accumulation": prior["runs"][label]["accumulation"],
                          "effective_batch": prior["runs"][label]["effective_batch"],
                          "gpu_peak_allocated_mib": status["gpu_peak_mib"],
                          "first_group_train_loss": status["train_loss"],
                          "pretrained_sha256": provenance["sha256"], "best_checkpoint_saved": False}
    output.write_text(json.dumps({"scope": "one optimizer group per model; no formal retraining",
                                  "gpu_memory_total_mib": prior["gpu"]["total_memory_mib"],
                                  "runs": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({label: row["gpu_peak_allocated_mib"] for label, row in records.items()}))


if __name__ == "__main__":
    main()
