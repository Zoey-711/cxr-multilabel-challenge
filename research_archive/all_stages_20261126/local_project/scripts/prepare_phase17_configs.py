"""Freeze five DenseNet fold configs from B01's actual saved runtime config."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from cxr.data.patient_folds import INDEX_FILES, sha256
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="B01 saved runtime config.yaml")
    parser.add_argument("--fold-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline = load_config(args.config)
    fold_dir = args.fold_dir.resolve()
    report = json.loads((fold_dir / "fold_report.json").read_text(encoding="utf-8"))
    hashes = {name: sha256(fold_dir / name) for name in INDEX_FILES}
    if hashes != report["index_sha256"]:
        raise ValueError("Fold indices differ from frozen report")
    if baseline["model"]["name"] != "densenet121" or baseline["training"]["batch_size"] != 16 or \
            baseline["training"].get("gradient_accumulation_steps", 1) != 1 or \
            baseline["training"]["early_stopping_patience"] != 8:
        raise ValueError("B01 actual config differs from approved control")
    provenance = json.loads((args.config.parent / "pretrained_source.json").read_text(encoding="utf-8"))
    if provenance["sha256"] != "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395":
        raise ValueError("B01 ImageNet source hash changed")
    root = Path(__file__).resolve().parents[1]
    fold_reference = fold_dir.relative_to(root).as_posix()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    total = report["source_counts"]
    for fold in range(1, 6):
        config = copy.deepcopy(baseline)
        config["experiment"]["name"] = f"f17_densenet121_fold{fold}_384"
        config["model"]["pretrained_sha256"] = provenance["sha256"]
        oof = {key: report["folds"][str(fold)][key] for key in ("patients", "studies", "images")}
        training = {key: total[key] - oof[key] for key in oof}
        config["data"]["train_subset"] = {
            "kind": "patient_fold_v1", "directory": fold_reference, "fold": fold,
            "sha256": hashes, "expected_counts": {"train": training, "oof": oof}}
        check = copy.deepcopy(config)
        check["experiment"]["name"] = baseline["experiment"]["name"]
        del check["model"]["pretrained_sha256"]
        del check["data"]["train_subset"]
        if check != baseline:
            raise ValueError(f"Fold {fold} config has an unapproved B01 difference")
        path = output / f"f17_densenet121_fold{fold}_384.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        generated.append({"fold": fold, "config": str(path), "config_sha256": sha256(path),
                          "train": training, "oof": oof})
    audit = {"b01_runtime_config_sha256": sha256(args.config),
             "pretrained_weight_sha256": provenance["sha256"],
             "fold_index_sha256": hashes,
             "approved_differences": ["experiment.name", "data.train_subset", "model.pretrained_sha256 assertion"],
             "folds": generated,
             "official_val_role": "best checkpoint selection only",
             "oof_role": "final out-of-fold evaluation only; no early stopping"}
    (fold_dir / "config_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
