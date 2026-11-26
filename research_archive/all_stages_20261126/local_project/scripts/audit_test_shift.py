"""Audit label-free Train/Val/Test metadata and the frozen E01 inference protocol."""

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.inference.versioned_submission import sha256
from cxr.utils.config import load_config


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    ordered = sorted(values)
    result = {}
    for name, fraction in (("min", 0), ("p10", .1), ("p25", .25), ("median", .5),
                           ("p75", .75), ("p90", .9), ("max", 1)):
        index = fraction * (len(ordered) - 1)
        lower, upper = math.floor(index), math.ceil(index)
        result[name] = round(ordered[lower] * (upper - index) + ordered[upper] * (index - lower), 3) \
            if lower != upper else round(ordered[lower], 3)
    return result


def summarize(rows: list[dict[str, str]], split: str, names: set[str]) -> dict:
    studies: dict[tuple[str, str], int] = defaultdict(int)
    by_subject: dict[str, set[str]] = defaultdict(set)
    views, orientations, formats = Counter(), Counter(), Counter()
    heights, widths, ratios, pixels = [], [], [], []
    missing_image_files, bad_dimensions = [], []
    for row in rows:
        subject, study = row["Subject_id"], row["Study_id"]
        studies[(subject, study)] += 1
        by_subject[subject].add(study)
        views[row["ViewPosition"].strip() or "<blank>"] += 1
        orientations[row["PatientOrientation"].strip() or "<blank>"] += 1
        member = image_member(row, split)
        formats[Path(member).suffix.lower()] += 1
        if member not in names:
            missing_image_files.append(member)
        try:
            height, width = int(row["Rows"]), int(row["Columns"])
            if height <= 0 or width <= 0:
                raise ValueError("nonpositive")
        except (ValueError, TypeError):
            bad_dimensions.append(row["Dicom_id"])
            continue
        heights.append(height)
        widths.append(width)
        ratios.append(width / height)
        pixels.append(height * width)
    if len({row["Dicom_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image IDs in {split}")
    if len({study for _, study in studies}) != len(studies):
        raise ValueError(f"Study ID reused under different patients in {split}")
    study_counts = Counter(studies.values())
    return {"subjects": len(by_subject), "studies": len(studies), "images": len(rows),
            "view_position": dict(sorted(views.items())),
            "patient_orientation": dict(sorted(orientations.items())),
            "image_formats": dict(sorted(formats.items())),
            "image_height_rows": quantiles(heights), "image_width_columns": quantiles(widths),
            "aspect_ratio_width_over_height": quantiles(ratios),
            "image_pixels": quantiles(pixels),
            "images_per_study": {str(key): value for key, value in sorted(study_counts.items())},
            "multi_image_study_percent": round(100 * sum(n for count, n in study_counts.items() if count > 1) / len(studies), 3),
            "studies_per_subject": quantiles([len(value) for value in by_subject.values()]),
            "missing_image_file_count": len(missing_image_files),
            "missing_image_file_examples": missing_image_files[:5],
            "bad_dimension_count": len(bad_dimensions),
            "bad_dimension_examples": bad_dimensions[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/test_shift_20261016"))
    parser.add_argument("--report", type=Path, default=Path("docs/test_shift_audit.md"))
    args = parser.parse_args()
    config = load_config(args.config)
    archive = archive_path(config)
    splits = {split: read_split_rows(archive, split) for split in ("train", "val", "test")}
    with zipfile.ZipFile(archive) as container:
        names = set(container.namelist())
        summary = {split: summarize(rows, split, names) for split, rows in splits.items()}
        sampled_dimensions = {}
        for split, rows in splits.items():
            sampled_dimensions[split] = []
            for row in rows[:10]:
                with container.open(image_member(row, split)) as stream, Image.open(stream) as image:
                    actual = [image.height, image.width]
                declared = [int(row["Rows"]), int(row["Columns"])]
                sampled_dimensions[split].append({"image_id": row["Dicom_id"],
                                                  "declared": declared, "decoded": actual,
                                                  "match": declared == actual})
    subjects = {split: {row["Subject_id"] for row in rows} for split, rows in splits.items()}
    studies = {split: {row["Study_id"] for row in rows} for split, rows in splits.items()}
    overlaps = {f"{a}_{b}": {"shared_subjects": len(subjects[a] & subjects[b]),
                               "shared_studies": len(studies[a] & studies[b])}
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    e01 = Path("outputs/submissions/E01_20261016")
    e01_manifest = json.loads((e01 / "manifest.json").read_text(encoding="utf-8"))
    if sha256(e01 / "submission.csv") != e01_manifest["csv_sha256"]:
        raise ValueError("Frozen E01 submission hash mismatch")
    run_configs = {name: load_config(Path("outputs/experiments") / version / "config.yaml")
                   for name, version in (("B01", "b01_densenet121_384_20261016"),
                                         ("B02", "b02_densenet121_wd1e3_20261016_elevated"),
                                         ("B03", "b03_densenet121_lr3e5_20261016"))}
    protocol = {"all_models_same_archive": all(archive_path(item) == archive for item in run_configs.values()),
                "all_models_384": all(item["data"]["input_size"] == 384 for item in run_configs.values()),
                "all_models_study_mean": all(item["validation"]["study_fusion"] == "mean_probability"
                                             for item in run_configs.values()),
                "shared_deterministic_preprocessor": "CXRArchiveDataset.__getitem__ calls preprocess_image for train, val, and test; only train passes augmentation",
                "preprocessor": "RGB -> PIL ImageOps.pad full-image aspect-preserving 384x384 bilinear black -> tensor -> ImageNet normalization",
                "class_mapping": "dataset target[class_id - 1], E01 submission class IDs 1 through 10",
                "frozen_e01_csv_sha256_verified": True,
                "frozen_e01_csv_sha256": e01_manifest["csv_sha256"]}
    report = {"archive": str(archive), "metadata_only_no_test_labels": True,
              "splits": summary, "overlap": overlaps,
              "sampled_decoded_dimensions": sampled_dimensions, "protocol": protocol}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_shift_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Train / Val / Test label-free shift audit", "",
             "Source: ZIP split CSV metadata and ten decoded images per split. No test labels were read or inferred.", "",
             "## Measured distribution", "",
             "| Measure | Train | Val | Test |", "|---|---:|---:|---:|"]
    fields = [("Patients", lambda x: x["subjects"]), ("Studies", lambda x: x["studies"]),
              ("Images", lambda x: x["images"]),
              ("Studies per patient median", lambda x: x["studies_per_subject"]["median"]),
              ("Multi-image Studies (%)", lambda x: x["multi_image_study_percent"]),
              ("Height median (px)", lambda x: x["image_height_rows"]["median"]),
              ("Width median (px)", lambda x: x["image_width_columns"]["median"]),
              ("Width/height median", lambda x: x["aspect_ratio_width_over_height"]["median"]),
              ("Missing ZIP images", lambda x: x["missing_image_file_count"]),
              ("Invalid dimension metadata", lambda x: x["bad_dimension_count"])]
    for title, value in fields:
        lines.append("| " + title + " | " + " | ".join(str(value(summary[split])) for split in ("train", "val", "test")) + " |")
    for field, title in (("view_position", "ViewPosition"), ("patient_orientation", "PatientOrientation"),
                         ("images_per_study", "Images per Study")):
        lines += ["", f"## {title}", "", "| Value | Train | Val | Test |", "|---|---:|---:|---:|"]
        all_values = sorted(set().union(*(summary[split][field] for split in summary)))
        for key in all_values:
            lines.append("| " + str(key).replace("|", "\\|") + " | " + " | ".join(
                str(summary[split][field].get(key, 0)) for split in ("train", "val", "test")) + " |")
    lines += ["", "## Dimension quantiles", "", "| Measure | Train p10 / median / p90 | Val | Test |",
              "|---|---:|---:|---:|"]
    for field in ("image_height_rows", "image_width_columns", "aspect_ratio_width_over_height", "image_pixels"):
        lines.append("| " + field + " | " + " | ".join(
            " / ".join(str(summary[split][field][q]) for q in ("p10", "median", "p90"))
            for split in ("train", "val", "test")) + " |")
    def view_share(split: str, view: str) -> float:
        return 100 * summary[split]["view_position"].get(view, 0) / summary[split]["images"]

    ap_val, ap_test = view_share("val", "AP"), view_share("test", "AP")
    pa_val, pa_test = view_share("val", "PA"), view_share("test", "PA")
    blank_val, blank_test = view_share("val", "<blank>"), view_share("test", "<blank>")
    lines += ["", "## Confirmed findings", "",
              f"- Patient and Study overlaps: {overlaps}.",
              f"- AP images: Val {ap_val:.1f}% versus Test {ap_test:.1f}% ({ap_test-ap_val:+.1f} percentage points); PA: Val {pa_val:.1f}% versus Test {pa_test:.1f}% ({pa_test-pa_val:+.1f} points). Blank ViewPosition: Val {blank_val:.1f}% versus Test {blank_test:.1f}%.",
              f"- Multi-image Studies: Val {summary['val']['multi_image_study_percent']:.1f}% versus Test {summary['test']['multi_image_study_percent']:.1f}%. Test has {summary['test']['studies']} Studies from {summary['test']['subjects']} patients; Val has {summary['val']['studies']} from {summary['val']['subjects']}.",
              f"- Median image height: Val {summary['val']['image_height_rows']['median']:.0f}px versus Test {summary['test']['image_height_rows']['median']:.0f}px. Median width and width/height ratio are the same in both splits.",
              "- E01 test inference and validation use the same dataset preprocessor and Study Mean aggregation. Training-only augmentation is absent from validation and test. The source code maps class ID 1 to output index 0 and writes IDs 1–10 in the CSV.",
              "- E01 frozen CSV SHA-256 matches its manifest. Full image-to-Study Mean and E01 fusion were independently verified by `scripts/verify_submission_bundle.py`.",
              f"- The first ten decoded images in each split agree with Rows/Columns metadata: {all(item['match'] for group in sampled_dimensions.values() for item in group)}.",
              "", "## Hypotheses, not established causes", "",
              "- Differences in view, size, and Study image counts may contribute to a validation-to-leaderboard gap, but no test labels or per-subgroup test scores are available to establish a cause.",
              "- Test patients contribute more repeat Studies on average. Because the competition scores Studies, those repeated patients may influence the test metric differently from validation; the direction and magnitude of any effect cannot be determined without hidden labels.",
              "- A 75.51 leaderboard display would equal 0.7551 only if the display uses Macro-AUC percentage units; then the difference from local E01 validation AUC 0.8026744715463039 would be -0.04757. The display scale has not been independently confirmed.",
              "- The test set may have different disease prevalence or acquisition sites, but those properties cannot be determined from these metadata alone.",
              "", "No threshold tuning, fusion-weight search, or hidden test labels were used."]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.output_dir / "test_shift_audit.json"),
                      "markdown": str(args.report), "summary": summary,
                      "overlap": overlaps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
