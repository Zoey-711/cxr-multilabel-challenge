"""Reaggregate frozen M01 image probabilities and audit validation views."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cxr.data.archive import archive_path, read_split_rows
from cxr.engine.b01_analysis import binary_auc, read_study_predictions
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES


def metrics(probabilities, targets):
    aucs = [binary_auc(targets[:, j], probabilities[:, j]) for j in range(10)]
    return {"study_count": len(targets),
            "macro_auc": float(np.mean(aucs)) if all(value is not None for value in aucs) else None,
            "per_class_auc": aucs,
            "classes_with_defined_auc": sum(value is not None for value in aucs)}


def paired_bootstrap(schemes, targets, subjects, *, replicates, seed):
    patients = defaultdict(list)
    for index, subject in enumerate(subjects):
        patients[subject].append(index)
    groups = [np.asarray(indices) for indices in patients.values()]
    contrasts = {"Max_minus_Mean": [], "HalfMeanHalfMax_minus_Mean": [],
                 "HalfMeanHalfMax_minus_Max": []}
    rng = np.random.default_rng(seed)
    invalid = 0
    for _ in range(replicates):
        draw = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in draw])
        sampled_targets = targets[indices]
        if any(sampled_targets[:, j].min() == sampled_targets[:, j].max() for j in range(10)):
            invalid += 1
            continue
        auc = {name: float(np.mean([binary_auc(sampled_targets[:, j], p[indices, j]) for j in range(10)]))
               for name, p in schemes.items()}
        contrasts["Max_minus_Mean"].append(auc["Max"] - auc["Mean"])
        contrasts["HalfMeanHalfMax_minus_Mean"].append(auc["HalfMeanHalfMax"] - auc["Mean"])
        contrasts["HalfMeanHalfMax_minus_Max"].append(auc["HalfMeanHalfMax"] - auc["Max"])
    if replicates - invalid < 100:
        raise ValueError("Too few valid patient-level bootstrap draws")
    return {"patient_count": len(groups), "seed": seed, "requested_replicates": replicates,
            "valid_replicates": replicates - invalid, "invalid_replicates_excluded": invalid,
            "invalid_rule": "Discard a draw if any of ten classes lacks positive or negative Studies",
            "method": "paired patient-cluster percentile bootstrap using identical patient draws",
            "contrasts": {name: {"point_difference": None,
                                 "ci_95_lower": float(np.quantile(values, 0.025)),
                                 "ci_95_upper": float(np.quantile(values, 0.975))}
                          for name, values in contrasts.items()}}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m01-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run = args.m01_run_dir.resolve()
    output = args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("Phase 5 analysis must not modify the frozen M01 experiment")
    if load_config(args.config) != load_config(run / "config.yaml"):
        parser.error("Analysis config differs from frozen M01 runtime config")
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        parser.error("M01 must be complete")
    with (run / "validation_analysis" / "image_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        image_rows = list(csv.DictReader(handle))
    metadata_rows = read_split_rows(archive_path(load_config(args.config)), "val")
    meta = {row["Dicom_id"]: row for row in metadata_rows}
    if len(meta) != len(metadata_rows) or len(image_rows) != len(metadata_rows):
        raise ValueError("Image predictions and validation metadata have different or duplicate image counts")
    predictions = defaultdict(list)
    views = defaultdict(list)
    seen_images = set()
    for row in image_rows:
        image_id, study_id, subject_id = row["Image_id"], row["Study_id"], row["Subject_id"]
        if image_id in seen_images or image_id not in meta:
            raise ValueError(f"Duplicate or unknown validation image: {image_id}")
        seen_images.add(image_id)
        source = meta[image_id]
        if (subject_id, study_id) != (source["Subject_id"], source["Study_id"]):
            raise ValueError(f"Prediction/metadata Study mismatch: {image_id}")
        probability = np.asarray([float(row[f"probability_{j}"]) for j in range(1, 11)])
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid image probabilities: {image_id}")
        predictions[study_id].append(probability)
        views[study_id].append(source["ViewPosition"].strip() or "<blank>")
    if seen_images != set(meta):
        raise ValueError("Some validation images lack M01 probabilities")
    saved, targets, subjects, studies = read_study_predictions(
        run / "validation_analysis" / "study_predictions.csv")
    if len(studies) != 859 or len(set(studies)) != len(studies) or set(studies) != set(predictions):
        raise ValueError("Study_id alignment failed")
    for index, study_id in enumerate(studies):
        if any(meta[row["Image_id"]]["Subject_id"] != subjects[index]
               for row in image_rows if row["Study_id"] == study_id):
            raise ValueError(f"Study_id has conflicting Subject_id: {study_id}")
    mean = np.stack([np.mean(predictions[study], axis=0) for study in studies])
    maximum = np.stack([np.max(predictions[study], axis=0) for study in studies])
    if not np.array_equal(mean, saved):
        raise ValueError("Recomputed Mean differs from frozen M01 Study probabilities")
    schemes = {"Mean": mean, "Max": maximum, "HalfMeanHalfMax": (mean + maximum) / 2}
    scores = {name: metrics(p, targets) for name, p in schemes.items()}
    if any(value["classes_with_defined_auc"] != 10 for value in scores.values()):
        raise ValueError("Validation data lack positive/negative Studies for a class")
    counts = np.asarray([len(predictions[study]) for study in studies])
    subgroup = {label: {name: metrics(p[mask], targets[mask]) for name, p in schemes.items()}
                for label, mask in (("single_image", counts == 1), ("multi_image", counts > 1))}
    bootstrap = paired_bootstrap(schemes, targets, subjects,
                                 replicates=args.bootstrap_replicates, seed=args.seed)
    differences = {"Max_minus_Mean": scores["Max"]["macro_auc"] - scores["Mean"]["macro_auc"],
                   "HalfMeanHalfMax_minus_Mean": scores["HalfMeanHalfMax"]["macro_auc"] - scores["Mean"]["macro_auc"],
                   "HalfMeanHalfMax_minus_Max": scores["HalfMeanHalfMax"]["macro_auc"] - scores["Max"]["macro_auc"]}
    for name, difference in differences.items():
        bootstrap["contrasts"][name]["point_difference"] = difference
    view_by_image = Counter(view for values in views.values() for view in values)
    view_by_study = Counter(" + ".join(sorted(set(views[study]))) for study in studies)
    result = {"image_count": len(image_rows), "study_count": len(studies), "patient_count": len(set(subjects)),
              "image_count_per_study": dict(sorted(Counter(map(int, counts)).items())),
              "distinct_view_count_per_study": dict(sorted(Counter(len(set(views[study])) for study in studies).items())),
              "view_position_image_counts": dict(sorted(view_by_image.items())),
              "view_combination_study_counts": dict(view_by_study.most_common()),
              "scores": scores, "subgroups": subgroup, "paired_bootstrap": bootstrap,
              "frozen_mean_exact_match": True}
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiview_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "per_class_auc.csv", [
        {"class_id": j + 1, "class_name": name, "positive_studies": int(targets[:, j].sum()),
         "negative_studies": int((targets[:, j] == 0).sum()),
         **{f"{scheme}_auc": scores[scheme]["per_class_auc"][j] for scheme in schemes}}
        for j, name in enumerate(CLASS_NAMES)])
    write_csv(output / "study_views.csv", [
        {"Subject_id": subjects[i], "Study_id": study, "image_count": int(counts[i]),
         "distinct_view_count": len(set(views[study])), "views": ";".join(views[study])}
        for i, study in enumerate(studies)])
    study_image_count = {study: int(counts[i]) for i, study in enumerate(studies)}
    write_csv(output / "view_distribution.csv", [
        {"ViewPosition": name, "image_count": count,
         "single_image_study_images": sum(1 for study in studies if study_image_count[study] == 1
                                          for view in views[study] if view == name),
         "multi_image_study_images": sum(1 for study in studies if study_image_count[study] > 1
                                         for view in views[study] if view == name)}
        for name, count in sorted(view_by_image.items())])
    lines = ["# M01 multiview aggregation", "",
             f"Aligned {len(image_rows)} saved image probabilities to {len(studies)} validation Studies and {len(set(subjects))} patients.",
             "The reconstructed Mean exactly matches frozen M01 Study probabilities. No model inference or weight change was made.",
             "", "| Aggregation | Macro-AUC | Delta vs Mean | Paired patient-bootstrap 95% CI |",
             "|---|---:|---:|---:|"]
    for name in schemes:
        contrast = {"Max": "Max_minus_Mean", "HalfMeanHalfMax": "HalfMeanHalfMax_minus_Mean"}.get(name)
        ci = bootstrap["contrasts"].get(contrast) if contrast else None
        lines.append(f"| {name} | {scores[name]['macro_auc']:.5f} | " +
                     (f"{ci['point_difference']:+.5f} | [{ci['ci_95_lower']:+.5f}, {ci['ci_95_upper']:+.5f}] |"
                      if ci else "+0.00000 | reference |"))
    lines += ["", f"Bootstrap: {bootstrap['valid_replicates']}/{bootstrap['requested_replicates']} valid patient draws; "
              f"{bootstrap['invalid_replicates_excluded']} excluded.", "",
              "| Subgroup | Studies | Mean AUC | Max AUC | HalfMeanHalfMax AUC |",
              "|---|---:|---:|---:|---:|"]
    for label in subgroup:
        group = subgroup[label]
        lines.append(f"| {label} | {group['Mean']['study_count']} | " + " | ".join(
            "NA" if group[name]["macro_auc"] is None else f"{group[name]['macro_auc']:.5f}"
            for name in schemes) + " |")
    lines += ["", "| Class | Mean AUC | Max AUC | HalfMeanHalfMax AUC |", "|---|---:|---:|---:|"]
    for j, label in enumerate(CLASS_NAMES):
        lines.append(f"| {j + 1} {label} | " + " | ".join(
            f"{scores[name]['per_class_auc'][j]:.5f}" for name in schemes) + " |")
    lines += ["", "ViewPosition counts are image-level in `view_distribution.csv`; per-Study combinations are in the JSON report."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"scores": {key: value["macro_auc"] for key, value in scores.items()},
                      "subgroups": {group: {name: value["macro_auc"] for name, value in schemes.items()}
                                    for group, schemes in subgroup.items()},
                      "bootstrap": bootstrap["contrasts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
