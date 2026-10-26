"""Create ConvNeXt and Swin fold configs from actual frozen runtime configs."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from cxr.data.patient_folds import INDEX_FILES, sha256
from cxr.utils.config import load_config


CONTROLS = {
    "C17": ("convnext_tiny", "m02_convnext_tiny_384_aug_20261016_elevated", 6),
    "S17": ("swin_t", "t01_swin_tiny_384_20261018", 6),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Project config used to locate dataset")
    parser.add_argument("--fold-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    folds_dir = args.fold_dir.resolve()
    report = json.loads((folds_dir / "fold_report.json").read_text(encoding="utf-8"))
    hashes = {name: sha256(folds_dir / name) for name in INDEX_FILES}
    if hashes != report["index_sha256"]:
        raise ValueError("Fold index SHA-256 mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_reference = folds_dir.relative_to(root).as_posix()
    generated = []
    for prefix, (model, run_id, patience) in CONTROLS.items():
        run_dir = root / "outputs" / "experiments" / run_id
        control_path = run_dir / "config.yaml"
        baseline = load_config(control_path)
        provenance = json.loads((run_dir / "pretrained_source.json").read_text(encoding="utf-8"))
        if (baseline["model"]["name"] != model or baseline["training"]["early_stopping_patience"] != patience or
                baseline["training"]["batch_size"] != 8 or
                baseline["training"].get("gradient_accumulation_steps") != 2):
            raise ValueError(f"{prefix} actual control differs from approved configuration")
        for fold in range(1, 6):
            config = copy.deepcopy(baseline)
            config["experiment"]["name"] = f"{prefix.lower()}_fold{fold}_384"
            config["model"]["pretrained_sha256"] = provenance["sha256"]
            oof = {key: report["folds"][str(fold)][key] for key in ("patients", "studies", "images")}
            train = {key: report["source_counts"][key] - oof[key] for key in oof}
            config["data"]["train_subset"] = {
                "kind": "patient_fold_v1", "directory": fold_reference, "fold": fold,
                "sha256": hashes, "expected_counts": {"train": train, "oof": oof}}
            compared = copy.deepcopy(config)
            compared["experiment"]["name"] = baseline["experiment"]["name"]
            del compared["model"]["pretrained_sha256"]
            del compared["data"]["train_subset"]
            if compared != baseline:
                raise ValueError(f"{prefix}-F{fold} has an unapproved control difference")
            path = output / f"{prefix.lower()}_fold{fold}_384.yaml"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            generated.append({"name": f"{prefix}-F{fold}", "path": str(path),
                              "sha256": sha256(path), "control_config_sha256": sha256(control_path),
                              "pretrained_weight_sha256": provenance["sha256"],
                              "train": train, "oof": oof})
    audit = {"fold_index_sha256": hashes, "generated": generated,
             "approved_differences": ["experiment.name", "data.train_subset", "model.pretrained_sha256 assertion"],
             "resume_behavior": "Phase17 uses nonpersistent workers and saves main/DataLoader RNG states each completed epoch"}
    (folds_dir / "joint_config_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated_count": len(generated), "fold_index_sha256": hashes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
