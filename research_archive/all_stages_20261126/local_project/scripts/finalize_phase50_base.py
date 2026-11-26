"""Fixed epoch-3/4 SWA for the isolated H50-BASE trajectory."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import update_bn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.engine.metrics import binary_auc
from cxr.phase34_aux import digest, save_torch, write_csv, write_json
from cxr.phase39_run import infer
from run_phase50_base import ARCHIVE, OUT, RUN, model_new, setup


def read_probabilities(path, keys):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = {(row["Subject_id"], row["Study_id"]):
             [float(row[f"Prob_{i}"]) for i in range(1, 11)] for row in rows}
    if len(table) != len(rows) or set(table) != set(keys):
        raise ValueError(f"Invalid Study coverage: {path}")
    output = np.asarray([table[key] for key in keys])
    if output.shape != (len(keys), 10) or not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
        raise ValueError(f"Invalid probability values: {path}")
    return output


def paired_analysis(targets, base, swa):
    keys = [(target.subject_id, target.study_id) for target in targets]
    y = np.asarray([target.labels for target in targets], dtype=np.uint8)
    per_class = []
    for class_index in range(10):
        original = binary_auc(y[:, class_index], base[:, class_index])
        averaged = binary_auc(y[:, class_index], swa[:, class_index])
        per_class.append(dict(class_id=class_index + 1, positive_studies=int(y[:, class_index].sum()),
                              base_auc=original, swa_auc=averaged, delta=averaged - original))
    groups = {}
    for index, (patient, _) in enumerate(keys):
        groups.setdefault(patient, []).append(index)
    patient_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(5001)
    draws = []
    base_draws = []
    invalid = 0
    for _ in range(2000):
        sample = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        indices = np.fromiter((index for patient in sample for index in groups[patient]), dtype=np.int64)
        try:
            base_auc = [binary_auc(y[indices, j], base[indices, j]) for j in range(10)]
            swa_auc = [binary_auc(y[indices, j], swa[indices, j]) for j in range(10)]
            base_draws.append(float(np.mean(base_auc)))
            draws.append(float(np.mean(np.asarray(swa_auc) - np.asarray(base_auc))))
        except ValueError:
            invalid += 1
    if len(draws) < 1900:
        raise ValueError("Too many invalid patient bootstrap samples")
    base_auc = float(np.mean([item["base_auc"] for item in per_class]))
    swa_auc = float(np.mean([item["swa_auc"] for item in per_class]))
    return per_class, dict(base_oof_auc=base_auc, swa_oof_auc=swa_auc, delta=swa_auc-base_auc,
                           ci95=np.percentile(draws, [2.5, 97.5]).tolist(), valid=len(draws),
                           base_ci95=np.percentile(base_draws, [2.5, 97.5]).tolist(),
                           invalid=invalid, seed=5001, unit="patient")


def main():
    if not (RUN / "DONE.json").is_file():
        raise ValueError("BASE has not completed; SWA requires its frozen trajectory")
    cfg, train_rows, heldout = setup()
    expected = cfg["swa_raw_epochs"]
    if expected != [3, 4]:
        raise ValueError("SWA epoch rule changed")
    inputs = [RUN / f"raw_epoch_{epoch:02d}.pth" for epoch in expected]
    if not all(path.is_file() for path in inputs):
        write_json(OUT / "swa_status.json", dict(state="BLOCKED", reason="fixed epoch 3/4 checkpoint missing"))
        (OUT / "final_decision.md").write_text(
            "# Phase50 Local B\n\nBASE completed; SWA BLOCKED because a fixed epoch-3/4 checkpoint is missing. "
            "KD remains BLOCKED pending outer-Fold1-isolated inner teachers.\n", encoding="utf-8")
        return
    output = OUT / "H50_SWA_F1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE.json").exists():
        raise FileExistsError("Completed SWA is frozen")
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in inputs]
    if states[0].keys() != states[1].keys():
        raise ValueError("SWA state keys differ")
    merged = {}
    for key in states[0]:
        a, b = states[0][key], states[1][key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"SWA tensor mismatch: {key}")
        merged[key] = (a.float().add(b.float()).mul_(0.5).to(a.dtype) if a.is_floating_point() else b.clone())
    model = model_new(False).cuda()
    model.load_state_dict(merged)
    bn_loader = make_dataloader(ARCHIVE, "train", rows=train_rows, batch_size=cfg["micro_batch"],
                                num_workers=0, shuffle=False, seed=42,
                                preprocessing=cfg["preprocessing"], input_size=cfg["input_size"])
    with torch.no_grad():
        update_bn((batch["image"] for batch in bn_loader), model, device="cuda")
    model.eval()
    save_torch(output / "swa_model.pth", model.state_dict())
    eval_loader = make_dataloader(ARCHIVE, "train", rows=heldout, batch_size=cfg["micro_batch"],
                                  num_workers=0, shuffle=False, seed=43,
                                  preprocessing=cfg["preprocessing"], input_size=cfg["input_size"])
    targets = study_targets_from_rows(heldout)
    swa_metrics = infer(model, eval_loader, targets, output / "oof_predictions.csv", torch.device("cuda"))
    keys = [(target.subject_id, target.study_id) for target in targets]
    base = read_probabilities(RUN / "oof_predictions.csv", keys)
    swa = read_probabilities(output / "oof_predictions.csv", keys)
    per_class, paired = paired_analysis(targets, base, swa)
    write_csv(output / "per_class_auc.csv", per_class)
    write_json(output / "bootstrap.json", paired)
    write_json(output / "metrics.json", swa_metrics)
    write_json(output / "run_manifest.json", dict(source="H50_BASE_F1", source_checkpoint_epochs=expected,
        source_checkpoint_sha256={path.name:digest(path) for path in inputs},
        train_fold_bn_recalibration=True, fold1_patient_seen_during_training=False,
        validation_used_to_choose_epochs=False, model_type="same_trajectory_raw_weight_swa"))
    status = "EXPLORATORY_GAIN" if paired["ci95"][0] > 0 else "COMPLETED_NO_CONFIRMED_GAIN"
    (output / "final_decision.md").write_text(
        f"# H50 SWA\n\nStatus: {status}. Fixed raw epochs 3 and 4. "
        f"BASE OOF {paired['base_oof_auc']:.6f}; SWA OOF {paired['swa_oof_auc']:.6f}; "
        f"delta {paired['delta']:+.6f}; paired patient 95% CI {paired['ci95']}. "
        "Fold1 has participated in prior research; this is exploratory.\n", encoding="utf-8")
    hashes = {path.name:digest(path) for path in output.iterdir() if path.is_file() and path.name != "DONE.json"}
    write_json(output / "DONE.json", dict(state="DONE", sha256=hashes))
    base_metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    (OUT / "final_decision.md").write_text(
        f"# Phase50 Local B\n\nBASE DONE: best epoch {base_metrics['best_epoch']}, "
        f"Fold1 OOF AUC {paired['base_oof_auc']:.6f}, patient bootstrap 95% CI "
        f"{paired['base_ci95']}, peak allocated GPU memory "
        f"{base_metrics['peak_allocated_mib']:.1f} MiB. "
        f"SWA {status}: Fold1 OOF AUC {paired['swa_oof_auc']:.6f}, "
        f"delta {paired['delta']:+.6f}, paired 95% CI {paired['ci95']}. "
        "KD BLOCKED pending outer-Fold1-isolated inner teachers. "
        "Fold1 has participated in prior research, so these results are exploratory.\n",
        encoding="utf-8")
    print(json.dumps(paired))


if __name__ == "__main__":
    main()
