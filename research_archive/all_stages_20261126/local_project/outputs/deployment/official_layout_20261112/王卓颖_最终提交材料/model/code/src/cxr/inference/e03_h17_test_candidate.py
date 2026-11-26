"""Generate a fixed E03/H17 Test candidate from original images, with a closed audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, __version__ as pillow_version

from cxr.data.archive import image_member, read_split_rows, split_archive_path
from cxr.inference.e03_h17_validation import NAMES, fuse, load_models, predict_image, sha

EXPECTED_CLASSES = tuple(range(1, 11))
SUBMISSION_FIELDS = ["Subject_id", "Study_id", "Predict_class", "Probability"]


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_submission(path: Path, expected_keys: set[tuple[str, str]]) -> tuple[list[tuple[str, str, int]], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError(f"E08 submission columns differ: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 14_140:
        raise ValueError(f"E08 submission rows: {len(rows)}")
    order: list[tuple[str, str, int]] = []
    grouped: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        study = (r["Subject_id"], r["Study_id"])
        cls = int(r["Predict_class"])
        prob = float(r["Probability"])
        if cls not in EXPECTED_CLASSES or not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError(f"Invalid E08 row: {r}")
        if cls in grouped[study]:
            raise ValueError(f"Duplicate E08 key: {(study, cls)}")
        grouped[study][cls] = prob
        order.append((*study, cls))
    if set(grouped) != expected_keys or len(grouped) != 1414 or any(set(d) != set(EXPECTED_CLASSES) for d in grouped.values()):
        raise ValueError("E08 Study/class keys differ from official Test metadata")
    return order, {key: np.array([grouped[key][i] for i in EXPECTED_CLASSES], dtype=np.float64) for key in grouped}


def verify_source(cfg: dict, test_keys: set[tuple[str, str]]) -> tuple[dict, list, dict]:
    source_path = Path(cfg["candidate_manifest"])
    source = checked_json(source_path)
    expected_formula = {"B01": 1 / 12, "B02": 1 / 12, "B03": 1 / 12, "M02": .25, "H17_BATCH1": .5}
    if list(source["formula"]) != NAMES or any(abs(source["formula"][k] - v) > 1e-15 for k, v in expected_formula.items()):
        raise ValueError("Frozen candidate formula changed")
    if [w["model"] for w in source["weights"]] != NAMES:
        raise ValueError("Frozen checkpoint order changed")
    all_assets = source["weights"] + source["source_configs"]
    for entry in all_assets:
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Source file size/SHA mismatch: {path}")
    if sum(x["bytes"] for x in all_assets) != source["checkpoint_and_source_config_bytes"]:
        raise ValueError("Source size ledger mismatch")
    val = checked_json(Path(cfg["val_report"]))
    if val["state"] != "VAL_REPRODUCTION_COMPLETED" or not val["self_repeat_pass"] or val["studies"] != 859:
        raise ValueError("The batch=1 Val reproduction gate failed")
    val_audit = checked_json(Path(cfg["val_source_audit"]))
    if val_audit["candidate_manifest_sha256"] != sha(source_path):
        raise ValueError("Val gate used a different candidate manifest")
    old_entry = Path(cfg["val_executed_source"])
    current_entry = Path(cfg["inference_entry"])
    if sha(old_entry) != val_audit["code_sha256"]:
        raise ValueError("Val inference source snapshot changed")
    # The only source change after Val was Python int conversion for JSON reporting.
    old_text = old_entry.read_text(encoding="utf-8")
    if current_entry.read_text(encoding="utf-8") != old_text.replace("worst_class=j+1)", "worst_class=int(j+1))"):
        raise ValueError("Current inference entry differs from Val-validated forward logic")
    for key in ["archive", "candidate_manifest", "val_report", "val_source_audit", "val_executed_source", "inference_entry"]:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    e08_dir = Path(cfg["e08_submission"]).parent
    e08_manifest = checked_json(e08_dir / "manifest.json")
    if e08_manifest["files"]["submission.csv"]["sha256"] != sha(Path(cfg["e08_submission"])):
        raise ValueError("Official E08 submission does not match its frozen manifest")
    for item in source["weights"][:4]:
        frozen = e08_manifest["models"][item["model"]]
        if frozen["checkpoint_sha256"] != item["sha256"] or frozen["preprocessing"] != "black_pad_v1" or frozen["input_size"] != 384:
            raise ValueError("E03 source differs from frozen E08")
    h17_config = checked_json(Path(cfg["h17_deployment_config"]))
    if any((h17_config.get(k) != v) for k, v in {
        "model_id": "H17_DISTILL_02", "class_order": list(EXPECTED_CLASSES), "input_size": 384,
        "preprocessing": "black_pad_v1", "aggregation": "study_mean_of_image_probabilities",
        "inference_precision": "fp16_autocast", "batch_size": 1,
    }.items()) or h17_config["model_sha256"] != source["weights"][-1]["sha256"]:
        raise ValueError("H17 frozen deployment config differs")
    old_order, old_studies = read_submission(Path(cfg["e08_submission"]), test_keys)
    return source, old_order, old_studies


def rank_positions(prob: np.ndarray) -> np.ndarray:
    order = np.argsort(-prob, kind="stable")
    ranks = np.empty(10, dtype=np.int64)
    ranks[order] = np.arange(1, 11)
    return ranks


def compare_rows(old: dict, new: dict, order: list) -> tuple[list, list, dict]:
    records = []
    by_class = {k: {"abs": [], "binary_changed": 0, "rank_changed": 0, "top_changed": 0} for k in EXPECTED_CLASSES}
    studies_top_changed = set()
    for subject, study, cls in order:
        key = (subject, study)
        a = old[key]
        b = new[key]
        rank_a = rank_positions(a)
        rank_b = rank_positions(b)
        i = cls - 1
        delta = float(b[i] - a[i])
        old_bin = int(a[i] >= .5)
        new_bin = int(b[i] >= .5)
        top_changed = int(np.argmax(a) != np.argmax(b))
        if top_changed:
            studies_top_changed.add(key)
        stat = by_class[cls]
        stat["abs"].append(abs(delta))
        stat["binary_changed"] += int(old_bin != new_bin)
        stat["rank_changed"] += int(rank_a[i] != rank_b[i])
        stat["top_changed"] += top_changed
        records.append([subject, study, cls, format(float(a[i]), ".17g"), format(float(b[i]), ".17g"),
                        format(delta, ".17g"), old_bin, new_bin, int(rank_a[i]), int(rank_b[i]), top_changed])
    summary = []
    for cls, st in by_class.items():
        vals = np.array(st["abs"])
        summary.append([cls, float(vals.mean()), float(np.median(vals)), float(np.quantile(vals, .95)),
                        float(vals.max()), st["binary_changed"], st["rank_changed"], st["top_changed"]])
    totals = {"binary_changes": sum(st["binary_changed"] for st in by_class.values()),
              "class_rank_changes": sum(st["rank_changed"] for st in by_class.values()),
              "studies_with_changed_top_class": len(studies_top_changed)}
    return records, summary, totals


def write_csv(path: Path, fields: list[str], records: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(records)


def validate_outputs(out: Path, keys: set, old_order: list, expected: dict, images: list) -> None:
    candidate = out / "test_candidate.csv"
    with candidate.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError("Candidate columns differ from official E08")
        submission = list(reader)
    if len(submission) != 14_140:
        raise ValueError("Candidate row count invalid")
    actual_order = [(r["Subject_id"], r["Study_id"], int(r["Predict_class"])) for r in submission]
    if actual_order != old_order or len(set(actual_order)) != 14_140:
        raise ValueError("Candidate order, class mapping or duplicate keys invalid")
    if any(not np.isfinite(float(r["Probability"])) or not 0 <= float(r["Probability"]) <= 1 for r in submission):
        raise ValueError("Candidate probability invalid")
    wide = out / "test_predictions.csv"
    with wide.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["Subject_id", "Study_id"] + [f"Prob_{i}" for i in EXPECTED_CLASSES]:
            raise ValueError("Wide prediction columns invalid")
        rows = list(reader)
    if len(rows) != 1414:
        raise ValueError("Wide prediction Study count invalid")
    wmap = {(r["Subject_id"], r["Study_id"]): np.array([float(r[f"Prob_{i}"]) for i in EXPECTED_CLASSES]) for r in rows}
    if set(wmap) != keys or len(wmap) != len(rows):
        raise ValueError("Wide prediction Study keys invalid")
    for key in keys:
        if not np.array_equal(wmap[key], expected[key]):
            raise ValueError("Saved wide predictions differ from actual inference")
    for r in submission:
        key = (r["Subject_id"], r["Study_id"])
        if float(r["Probability"]) != wmap[key][int(r["Predict_class"]) - 1]:
            raise ValueError("Saved candidate differs from actual inference")
    with (out / "image_predictions.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        saved_images = list(reader)
    if len(saved_images) != 2218 or len({r["Dicom_id"] for r in saved_images}) != 2218:
        raise ValueError("Image prediction coverage invalid")
    regroup = defaultdict(list)
    for r in saved_images:
        regroup[(r["Subject_id"], r["Study_id"])].append(np.array([float(r[f"Prob_{i}"]) for i in EXPECTED_CLASSES]))
    if set(regroup) != keys:
        raise ValueError("Image prediction Study mapping invalid")
    for key in keys:
        if not np.allclose(np.mean(regroup[key], axis=0), wmap[key], rtol=0, atol=2e-16):
            raise ValueError("Stored image predictions do not aggregate to Study predictions")
    if [(r["Subject_id"], r["Study_id"], r["Dicom_id"]) for r in saved_images] != images:
        raise ValueError("Saved image order or ID mapping invalid")


def run(cfg: dict, out: Path, cfg_path: Path) -> None:
    started = time.perf_counter()
    archive = Path(cfg["archive"])
    rows = read_split_rows(archive, "test")
    if len(rows) != 2218 or len({r["Dicom_id"] for r in rows}) != 2218:
        raise ValueError("Official Test image coverage mismatch")
    grouped_rows = defaultdict(list)
    for r in rows:
        grouped_rows[(r["Subject_id"], r["Study_id"])].append(r["Dicom_id"])
    if len(grouped_rows) != 1414 or len({s for _, s in grouped_rows}) != 1414:
        raise ValueError("Official Test Study coverage mismatch")
    test_keys = set(grouped_rows)
    source, old_order, old_studies = verify_source(cfg, test_keys)
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed batch=1 FP16 CUDA path unavailable")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    models = load_models(source)
    torch.cuda.reset_peak_memory_stats()
    grouped_prob = defaultdict(list)
    images = []
    image_rows = []
    with zipfile.ZipFile(split_archive_path(archive, "test")) as z:
        for index, row in enumerate(rows, 1):
            member = image_member(row, "test")
            raw = z.read(member)
            with Image.open(io.BytesIO(raw)) as image:
                probabilities = predict_image(image, models)
            key = (row["Subject_id"], row["Study_id"])
            grouped_prob[key].append(probabilities["E03_H17"])
            images.append((*key, row["Dicom_id"]))
            image_rows.append([*key, row["Dicom_id"], hashlib.sha256(raw).hexdigest(),
                               *[format(float(x), ".17g") for x in probabilities["E03_H17"]]])
            if index % 100 == 0 or index == len(rows):
                print(json.dumps({"test_images_completed": index, "total": len(rows),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    studies = {key: np.mean(vals, axis=0) for key, vals in grouped_prob.items()}
    if set(studies) != test_keys or any(len(grouped_prob[k]) != len(grouped_rows[k]) for k in test_keys):
        raise ValueError("Study Mean image mapping incomplete")
    if any(v.shape != (10,) or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in studies.values()):
        raise ValueError("Invalid Study probabilities")
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    comparisons, per_class, totals = compare_rows(old_studies, studies, old_order)
    if len(comparisons) != 14_140:
        raise ValueError("Comparison coverage invalid")
    # Immutable inputs are checked again before any final file becomes visible.
    for asset in source["weights"] + source["source_configs"]:
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError(f"Source changed during inference: {asset['path']}")
    if sha(Path(cfg["e08_submission"])) != checked_json(Path(cfg["e08_submission"]).parent / "manifest.json")["files"]["submission.csv"]["sha256"]:
        raise ValueError("Frozen E08 changed during inference")

    write_csv(out / "test_candidate.csv", SUBMISSION_FIELDS,
              [[s, t, i, format(float(studies[(s, t)][i - 1]), ".17g")] for s, t, i in old_order])
    write_csv(out / "test_predictions.csv", ["Subject_id", "Study_id"] + [f"Prob_{i}" for i in EXPECTED_CLASSES],
              [[*key, *[format(float(x), ".17g") for x in studies[key]]] for key in sorted(test_keys)])
    write_csv(out / "image_predictions.csv", ["Subject_id", "Study_id", "Dicom_id", "image_sha256"] +
              [f"Prob_{i}" for i in EXPECTED_CLASSES], image_rows)
    write_csv(out / "comparison_vs_E08.csv",
              ["Subject_id", "Study_id", "Predict_class", "E08_Probability", "Candidate_Probability", "Candidate_minus_E08",
               "E08_binary_at_0_5", "Candidate_binary_at_0_5", "E08_rank", "Candidate_rank", "Study_top_class_changed"], comparisons)
    write_csv(out / "per_class_comparison.csv",
              ["Predict_class", "mean_abs_diff", "median_abs_diff", "p95_abs_diff", "max_abs_diff",
               "binary_changes", "rank_changes", "studies_with_top_change"], per_class)
    validate_outputs(out, test_keys, old_order, studies, images)

    module_paths = {
        "candidate_entry": Path(__file__),
        "validated_inference_entry": Path(cfg["inference_entry"]),
        "preprocessing": Path(cfg["preprocessing_source"]),
        "data_mapping": Path(cfg["archive_source"]),
        "model_factory": Path(cfg["model_factory_source"]),
        "checkpoint_loader": Path(cfg["checkpoint_source"]),
    }
    with zipfile.ZipFile(split_archive_path(archive, "test")) as z:
        with z.open("cxr_first/test.csv") as f:
            metadata_hash = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "candidate": "E03_50_H17_BATCH1_50", "status": "TEST_CSV_VALIDATED_READY_FOR_T4_PERFORMANCE_VERIFICATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "0.5*(0.5*(B01+B02+B03)/3 + 0.5*M02) + 0.5*H17_BATCH1",
        "inference": {"batch": 1, "precision": "FP16 CUDA autocast", "preprocessing": "black_pad_v1 384",
                      "study_aggregation": "mean of image sigmoid probabilities", "class_order": list(EXPECTED_CLASSES),
                      "binary_comparison_threshold": .5, "ranking_tiebreak": "ascending class ID", "models": NAMES},
        "coverage": {"test_images": len(rows), "studies": len(studies), "candidate_rows": len(old_order),
                     "per_study_10_classes": True, "original_metadata_sha256": metadata_hash,
                     "official_e08_exact_key_and_order_match": True},
        "comparison_vs_e08": totals,
        "source": {"candidate_manifest_sha256": sha(cfg["candidate_manifest"]),
                   "val_report_sha256": sha(cfg["val_report"]), "val_source_audit_sha256": sha(cfg["val_source_audit"]),
                   "e08_submission_sha256": sha(cfg["e08_submission"]),
                   "checkpoints": source["weights"], "configs": source["source_configs"],
                   "code_sha256": {name: sha(path) for name, path in module_paths.items()},
                   "run_config_sha256": sha(cfg_path),
                   "git_commit": cfg.get("git_commit", "UNAVAILABLE_NO_LOCAL_GIT_METADATA")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "pillow": pillow_version, "cuda_runtime": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "cudnn": torch.backends.cudnn.version()},
        "resource": {"total_wall_seconds_including_load_and_test": inference_seconds,
                     "peak_cuda_allocated_bytes": int(peak_allocated), "peak_cuda_reserved_bytes": int(peak_reserved),
                     "t4_status": "UNVERIFIED", "official_100ms_status": "UNVERIFIED",
                     "checkpoint_and_source_config_bytes": source["checkpoint_and_source_config_bytes"],
                     "local_timing_is_not_official_t4_benchmark": True},
        "safety": {"hidden_test_labels_read": False, "test_parameters_fitted": False,
                   "frozen_e08_modified": False, "uploaded_to_official_platform": False, "test_auc": None},
        "file_sha256": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in
                        ["test_candidate.csv", "test_predictions.csv", "image_predictions.csv",
                         "comparison_vs_E08.csv", "per_class_comparison.csv"]},
    }
    message = (
        "# E03＋H17 batch=1 Test候选审计\n\n"
        f"真实推理：{len(rows)}张Test图片，{len(studies)}个Study，{len(old_order)}行；全覆盖、每Study 10类、概率及ID核验PASS。"
        "按冻结E08官方文件顺序保存；Test标签未读取，Test AUC不计算。\n\n"
        "固定公式：0.5×[0.5×(B01+B02+B03)/3＋0.5×M02]＋0.5×H17_02_BATCH1。"
        "batch=1，FP16 CUDA，384 black_pad_v1，逐图sigmoid后Study Mean。"
        "全部五个checkpoint与来源配置在推理前后SHA-256均匹配；实际权重和必要来源配置"
        f"{source['checkpoint_and_source_config_bytes']:,}字节，低于500,000,000字节。\n\n"
        "旧E08的对照只用于描述预测变化，不据此调融合比例或阈值。"
        f"阈值0.5的二值变化{totals['binary_changes']}项，类别排名变化{totals['class_rank_changes']}项，"
        f"Top-1类别变化{totals['studies_with_changed_top_class']}个Study。逐类见per_class_comparison.csv。\n\n"
        f"4060本次处理含模型加载的总墙钟{inference_seconds:.1f}秒；峰值CUDA分配显存{peak_allocated/1048576:.1f}MiB。"
        "这不是T4官方单图测速，不能据此判断100ms合格。新组合标记T4_UNVERIFIED。\n\n"
        "提交前阶段结论：CSV及来源审计通过，具备进入T4真实性能验证的条件；"
        "目前没有官方成绩，不自动提交。Val batch=1 AUC=0.8130544037，"
        "与E08的患者级Bootstrap区间跨0，不能保证提分。\n\n"
        "所有输出独立保存，原E08提交和模型未修改。\n"
    )
    (out / "test_audit.md").write_text(message, encoding="utf-8")
    manifest["file_sha256"]["test_audit.md"] = {"sha256": sha(out / "test_audit.md"),
                                               "bytes": (out / "test_audit.md").stat().st_size}
    save_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "candidate_sha256": manifest["file_sha256"]["test_candidate.csv"]["sha256"],
                      "candidate_path": str(out / "test_candidate.csv"), "studies": len(studies),
                      "rows": len(old_order), "comparison": totals}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = checked_json(cfg_path)
    out = Path(cfg["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(cfg, out, cfg_path)
    except Exception as exc:
        save_json(out / "failed.json", {"status": "FAILED_NO_VALID_CANDIDATE", "reason": f"{type(exc).__name__}: {exc}",
                                        "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        for name in ["test_candidate.csv", "test_predictions.csv", "comparison_vs_E08.csv", "manifest.json"]:
            (out / name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
