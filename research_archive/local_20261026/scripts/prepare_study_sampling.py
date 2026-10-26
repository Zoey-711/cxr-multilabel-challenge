"""Quantify image-count bias and preflight a study-balanced sampling control."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from cxr.data.archive import archive_path, parse_positive_classes, read_split_rows
from cxr.data.sampling import StudyBalancedSampler
from cxr.utils.config import load_config


def flatten(data, prefix=""):
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result.update(flatten(value, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Proposed S01 config")
    parser.add_argument("--baseline-config", required=True, type=Path, help="Frozen M02 runtime config")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    m02, s01 = load_config(args.baseline_config), load_config(args.config)
    before, after = flatten(m02), flatten(s01)
    differences = {key: {"M02": before.get(key), "S01": after.get(key)}
                   for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
    if set(differences) != {"experiment.name", "data.train_sampling"} or \
            differences["data.train_sampling"] != {"M02": None, "S01": "study_balanced_v1"}:
        raise ValueError(f"S01 differs from M02 beyond sampling and run name: {differences}")
    archive = archive_path(m02)
    rows = read_split_rows(archive, "train")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["Subject_id"], row["Study_id"])].append(index)
    sizes = Counter(len(images) for images in groups.values())
    multi = {key for key, indices in groups.items() if len(indices) > 1}
    sampler = StudyBalancedSampler(rows, seed=int(s01["experiment"]["seed"]))
    selected = list(sampler)
    drawn = Counter((rows[index]["Subject_id"], rows[index]["Study_id"]) for index in selected)
    if len(selected) != len(rows) or set(drawn) != set(groups) or \
            max(drawn.values()) - min(drawn.values()) > 1:
        raise ValueError("Study-balanced sampling coverage or epoch size differs")
    labels = {}
    for key, indices in groups.items():
        values = {rows[index]["Predict_class"] for index in indices}
        if len(values) != 1:
            raise ValueError(f"Study labels differ across its images: {key}")
        labels[key] = parse_positive_classes(values.pop())
    classes = []
    for class_id in range(1, 11):
        positive = [key for key in groups if class_id in labels[key]]
        positive_images = sum(len(groups[key]) for key in positive)
        classes.append({"class_id": class_id, "positive_studies": len(positive),
                        "positive_images": positive_images,
                        "study_positive_rate": len(positive) / len(groups),
                        "image_positive_rate": positive_images / len(rows)})
    split_root = Path("outputs/analysis/phase5_internal_split")
    split_report = json.loads((split_root / "split_report.json").read_text(encoding="utf-8"))
    with (split_root / "study_split.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    patient_split = defaultdict(set)
    for row in split_rows:
        patient_split[row["Subject_id"]].add(row["split"])
    if any(len(parts) != 1 for parts in patient_split.values()) or \
            split_report["patient_overlap"] != 0 or \
            len(split_rows) != len(groups):
        raise ValueError("Frozen internal patient split fails alignment")
    val_rows = read_split_rows(archive, "val")
    val_sizes = Counter((row["Subject_id"], row["Study_id"]) for row in val_rows)
    official_strata = {"single_image_studies": sum(n == 1 for n in val_sizes.values()),
                       "multi_image_studies": sum(n >= 2 for n in val_sizes.values())}
    internal_val_keys = {(row["Subject_id"], row["Study_id"]) for row in split_rows
                         if row["split"] == "internal_val"}
    internal_strata = {"single_image_studies": sum(len(groups[key]) == 1 for key in internal_val_keys),
                       "multi_image_studies": sum(len(groups[key]) >= 2 for key in internal_val_keys)}
    data = {"baseline": "frozen M02", "candidate": "S01: one or two uniformly selected images per Study per epoch",
            "config_differences": differences,
            "train": {"patients": len({key[0] for key in groups}), "studies": len(groups),
                      "images": len(rows), "image_count_distribution": dict(sorted(sizes.items())),
                      "multi_image_study_count": len(multi),
                      "multi_image_study_fraction": len(multi) / len(groups),
                      "multi_image_training_image_count": sum(len(groups[key]) for key in multi),
                      "multi_image_training_image_fraction": sum(len(groups[key]) for key in multi) / len(rows),
                      "baseline_study_exposure_min": min(sizes), "baseline_study_exposure_max": max(sizes),
                      "candidate_study_exposure_distribution": dict(sorted(Counter(drawn.values()).items())),
                      "epoch_images_unchanged": len(selected) == len(rows),
                      "optimizer_steps_per_epoch": math.ceil(math.ceil(len(rows) / int(s01["training"]["batch_size"])) /
                                                           int(s01["training"]["gradient_accumulation_steps"]))},
            "class_prevalence": classes,
            "internal_split": {"source": str(split_root), "counts": split_report["counts"],
                               "patient_overlap": split_report["patient_overlap"],
                               "all_ten_classes_both_signs": split_report["all_ten_classes_have_positive_and_negative_studies_in_both_splits"]},
            "evaluation_strata": {"official_val": official_strata, "internal_val": internal_strata}}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "sampling_audit.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# S01 Study-balanced sampling plan", "",
             f"M02 trained on {len(rows)} images from {len(groups)} Studies and {data['train']['patients']} patients.",
             f"Multi-image Studies: {len(multi)} ({len(multi)/len(groups):.1%}) of Studies, but "
             f"{data['train']['multi_image_training_image_fraction']:.1%} of image-level training exposures.",
             f"Original Study exposures per epoch range from {min(sizes)} to {max(sizes)} images. "
             f"Proposed S01 exposures: {data['train']['candidate_study_exposure_distribution']}; "
             f"epoch still has {len(rows)} image draws and "
             f"{data['train']['optimizer_steps_per_epoch']} optimizer steps.", "",
             "The sampler draws every Study once, samples distinct Studies for the remaining draws, "
             "and chooses one source image uniformly on each draw. It uses seed+epoch so resume is reproducible. "
             "Subjects with multiple Studies still contribute once per Study; patient balancing is a separate question.",
             "", "Only algorithmic config change from frozen M02: `data.train_sampling: study_balanced_v1`. "
             "Run name is bookkeeping. BCE, ConvNeXt-Tiny, LR, augmentation, batch/accumulation, seed, "
             "384px preprocessing, official validation and Mean aggregation remain the same.", "",
             f"Validation groups: one image vs two or more images per Study. Official val: {official_strata}; "
             f"internal val: {internal_strata}. Report group counts, "
             "ten per-class AUCs, Macro-AUC only if each class has both signs, and paired patient "
             "bootstrap of S01-M02 within each group. Keep 0.5 F1 as secondary. "
             "The official 859-Study validation remains the model selection set.", "",
             "Generalization check: reuse the frozen patient split index (9949/1106 patients; "
             "16667/1846 Studies), with no Subject_id overlap. A future internal rerun must train "
             "both M02 control and S01 candidate only on internal_train; the frozen full-train M02 "
             "checkpoint cannot evaluate internal_val without leakage. No such rerun is authorized now.", "",
             "Training is **not started**. Proposed command after approval: "
             "`.\\.venv\\Scripts\\python.exe scripts\\train.py --config configs\\s01_convnext_tiny_study_balanced_384.yaml`."]
    (args.output_dir / "experiment_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "studies": len(groups), "images": len(rows),
                      "multi_studies": len(multi), "new_exposures": dict(Counter(drawn.values()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
