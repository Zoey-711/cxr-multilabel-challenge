"""Same model loader/inference entry as predict.py. Reference timings are not T4 proof."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from predict import load_models, predict_image
HERE = Path(__file__).resolve().parent

def validate_submission(path: Path) -> dict:
    keys=set(); studies={}
    with path.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames != ["Subject_id","Study_id","Predict_class","Probability"]:
            raise ValueError("Submission columns/order do not match official four-column format")
        for row in reader:
            key=(row["Subject_id"],row["Study_id"])
            cls=int(row["Predict_class"])
            prob=float(row["Probability"])
            if not key[0] or not key[1] or cls not in range(1,11) or not math.isfinite(prob) or not 0<=prob<=1:
                raise ValueError("Invalid submission ID/class/probability")
            if (key,cls) in keys:
                raise ValueError("Duplicate Study/class submission row")
            keys.add((key,cls));studies.setdefault(key,set()).add(cls)
    if len(keys)!=14140 or len(studies)!=1414 or any(classes!=set(range(1,11)) for classes in studies.values()):
        raise ValueError("Submission must contain 1414 Studies and all 10 classes each")
    return {"rows":len(keys),"studies":len(studies),"format_status":"PASS"}

def list_images(folder: Path, expected: int) -> list[Path]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != expected:
        raise ValueError(f"{folder}: expected {expected} images, got {len(paths)}")
    return paths

def run(candidate: str, measured: Path, warmup: Path, *, reference: Path | None = None,
        study_index: Path | None = None, study_reference: Path | None = None,
        official_sample_attested: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for cost measurement")
    images, warm_images = list_images(measured, 100), list_images(warmup, 5)
    models = load_models(candidate, device="cuda")
    for image in warm_images:
        predict_image(image, models, candidate=candidate)
        torch.cuda.synchronize()
    output = HERE / f"{candidate}_cost_predictions.csv"
    timings = {str(p.relative_to(measured)): [] for p in images}
    final = {}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "image", *[f"Prob_{i}" for i in range(1, 11)]])
        for round_id in range(1, 4):
            for path in images:
                torch.cuda.synchronize()
                start = time.perf_counter()
                probabilities = predict_image(path, models, candidate=candidate)
                writer.writerow([round_id, str(path.relative_to(measured)), *[format(float(v), ".17g") for v in probabilities]])
                handle.flush()
                torch.cuda.synchronize()
                timings[str(path.relative_to(measured))].append((time.perf_counter() - start) * 1000)
                final[str(path.relative_to(measured))] = probabilities
    per_image = [statistics.mean(v) for v in timings.values()]
    report = {"candidate": candidate, "gpu": torch.cuda.get_device_name(), "batch": 1,
              "warmup_images": 5, "measured_images": 100, "rounds": 3,
              "scope": "image file read through final probability CSV write, models preloaded; CUDA synchronized",
              "precision": "FP16 autocast", "timer": "time.perf_counter wall clock with CUDA synchronization",
              "mean_ms": statistics.mean(per_image), "p50_ms": float(np.percentile(per_image, 50)),
              "p95_ms": float(np.percentile(per_image, 95)), "max_ms": max(per_image),
              "latency_status": "T4_UNVERIFIED" if "T4" not in torch.cuda.get_device_name() else "TARGET_MEASURED",
              "official_sample_attested": official_sample_attested,
              "limit_ms": 100.0, "latency_under_limit_on_measured_device": statistics.mean(per_image) <= 100.0,
              "submission_equivalence_status": "UNVERIFIED_NO_STUDY_INDEX",
              "per_image_three_round_mean_ms": {k: statistics.mean(v) for k, v in timings.items()},
              "predictions_file": str(output)}
    if reference:
        # Reference file is image-level and must map the exact same 100 filenames.
        with reference.open(encoding="utf-8-sig", newline="") as f:
            expected = {r["image"]: np.array([float(r[f"Prob_{i}"]) for i in range(1,11)])
                        for r in csv.DictReader(f)}
        if set(expected) != set(final):
            raise ValueError("Reference image IDs differ; refusing false submission equivalence claim")
        report["reference_max_abs_probability_diff"] = max(float(np.max(np.abs(final[k]-expected[k]))) for k in expected)
        report["reference_tolerance"] = 1e-4
        report["reference_status"] = "PASS" if report["reference_max_abs_probability_diff"] <= 1e-4 else "FAIL"
    if study_index or study_reference:
        if not study_index or not study_reference:
            raise ValueError("Study index and Study reference must both be supplied")
        with study_index.open(encoding="utf-8-sig", newline="") as f:
            index_rows=list(csv.DictReader(f))
        if len(index_rows)!=100 or {r["image"] for r in index_rows} != set(final):
            raise ValueError("Cost sample Study index does not cover exactly the 100 images")
        grouped={}
        for r in index_rows:
            key=(r["Subject_id"],r["Study_id"])
            grouped.setdefault(key,[]).append(final[r["image"]])
        complete={key:np.mean(v,axis=0) for key,v in grouped.items() if len(v)==int(next(
            r["study_image_count"] for r in index_rows if (r["Subject_id"],r["Study_id"])==key))}
        with study_reference.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        if not rows:
            raise ValueError("Empty Study reference")
        if "Prob_1" in rows[0] or "probability_1" in rows[0]:
            col="Prob_" if "Prob_1" in rows[0] else "probability_"
            truth={(r["Subject_id"],r["Study_id"]):np.array([float(r[f"{col}{i}"]) for i in range(1,11)]) for r in rows}
        else:
            truth={}
            for r in rows:
                key=(r["Subject_id"],r["Study_id"])
                truth.setdefault(key,np.full(10,np.nan))[int(r["Predict_class"])-1]=float(r["Probability"])
        if not complete or not set(complete).issubset(truth):
            raise ValueError("No fully represented Study for comparison with submission")
        delta=max(float(np.max(np.abs(prob-truth[key]))) for key,prob in complete.items())
        report["complete_studies_compared_to_submission"] = len(complete)
        report["submission_max_abs_probability_diff"] = delta
        report["submission_tolerance"] = 1e-4
        report["submission_equivalence_status"] = "PASS" if delta <= 1e-4 else "FAIL"
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_csv", type=Path, help="Frozen Study-level submission; retained for provenance")
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--candidate", choices=["E08_EMA", "H17_02"], default="E08_EMA")
    parser.add_argument("--image-reference", type=Path)
    parser.add_argument("--study-index", type=Path)
    parser.add_argument("--study-reference", type=Path, help="Defaults to positional submission_csv when --study-index is set")
    parser.add_argument("--official-sample", action="store_true",
                        help="Attest pictures_5/100 are the organizer's fixed timing sample")
    args = parser.parse_args()
    if not args.submission_csv.is_file():
        parser.error("Frozen submission CSV missing")
    submission_format=validate_submission(args.submission_csv)
    if args.study_reference and not args.study_index:
        parser.error("--study-reference requires --study-index")
    report=run(args.candidate, args.pictures_100, args.pictures_5,
               reference=args.image_reference, study_index=args.study_index,
               study_reference=(args.study_reference or args.submission_csv) if args.study_index else None,
               official_sample_attested=args.official_sample)
    report["submission_csv"] = str(args.submission_csv)
    report["submission_format"] = submission_format
    report["deployment_status"] = "LATENCY_UNVERIFIED" if report["latency_status"] == "T4_UNVERIFIED" else (
        "OFFICIAL_SAMPLE_UNVERIFIED" if not report["official_sample_attested"] else (
        "LATENCY_FAIL" if not report["latency_under_limit_on_measured_device"] else (
            "PREDICTION_UNVERIFIED" if report["submission_equivalence_status"] != "PASS" else "LATENCY_PASS_PREDICTION_PASS")))
    out=HERE / f"{args.candidate}_cost_report.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report":str(out),"mean_ms":report["mean_ms"],
                      "deployment_status":report["deployment_status"]},ensure_ascii=False))
