"""Reproducible E03/E08 one-image full-path benchmark; target status stays unverified.

This reuses the exact Phase20 shared-input implementation and frozen checkpoints.
It does not export a new model or modify the production inference path.
"""

import argparse
import hashlib
import json
import os
import runpy
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHASE20 = ROOT / "outputs/analysis/phase20_inference_optimization"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", choices=("E03", "E08-EMA"), required=True)
    parser.add_argument("--mode", choices=("verify", "benchmark"), required=True)
    parser.add_argument("--archive", type=Path, help="Official ZIP; required only for benchmark")
    parser.add_argument("--output", type=Path, help="New benchmark JSON; never overwrite")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["source_manifest"]).read_text(encoding="utf-8"))
    names = config["models_e03" if args.candidate == "E03" else "models_e08"]
    if config["preprocessing"] != "black_pad_v1" or config["image_size"] != 384:
        raise ValueError("Different input protocol")
    total = 0
    for name in names:
        item = manifest["models"][name]
        path = ROOT / "outputs/experiments" / item["run_id"] / item["checkpoint"]
        if digest(path) != item["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {name}")
        total += path.stat().st_size
    equivalence = json.loads((ROOT / config["phase20_equivalence"]).read_text(encoding="utf-8"))
    result = {"candidate": args.candidate, "models": names, "weight_bytes": total,
              "size_status": "PASS" if total <= 500_000_000 else "SIZE_FAIL",
              "phase20_equivalence_sha256": digest(ROOT / config["phase20_equivalence"]),
              "official_protocol_status": config["official_latency_protocol_status"],
              "official_deployment_status": "LATENCY_UNVERIFIED",
              "note": "The official timer scope/statistic is not documented; this is a reproducible research measurement."}
    if args.mode == "verify":
        result["phase20_equivalence_available"] = bool(equivalence)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.archive is None or args.output is None:
        parser.error("--archive and --output are required for benchmark")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    import numpy as np
    import torch
    from cxr.data.archive import read_split_rows

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real benchmark")
    scope = runpy.run_path(str(PHASE20 / "run_phase20.py"))
    _, sources = scope["load_sources"]()
    if args.candidate == "E03":
        removed = sources.pop("N01_EMA")
        del removed
        torch.cuda.empty_cache()
    rows = scope["choose_images"](read_split_rows(args.archive, "val"))
    measured = []
    stages = defaultdict(list)
    with zipfile.ZipFile(args.archive) as archive:
        for _ in range(config["benchmark_warmup_rounds"]):
            for row in rows:
                scope["timed_single"](archive, row, sources, tuple(names), shared=True)
        for _ in range(config["benchmark_measured_rounds"]):
            for row in rows:
                _, _, parts = scope["timed_single"](archive, row, sources, tuple(names), shared=True)
                measured.extend(parts["full_end_to_end"])
                for key, values in parts.items():
                    stages[key].extend(values)
    result.update({"device": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                   "torch": torch.__version__, "batch_size": 1,
                   "warmup_rounds": config["benchmark_warmup_rounds"],
                   "measured_rounds": config["benchmark_measured_rounds"],
                   "images_per_round": len(rows), "samples": len(measured),
                   "scope": "ZIP read + RGB decode + resize/pad + normalization + H2D + all model forwards + fusion + one-image Study Mean; models preloaded",
                   "latency_ms": {"mean": statistics.mean(measured), "p50": float(np.percentile(measured, 50)),
                                  "p95": float(np.percentile(measured, 95)), "max": max(measured)},
                   "stages_mean_ms": {key: statistics.mean(values) for key, values in stages.items()},
                   "official_deployment_status": "SIZE_FAIL" if total > 500_000_000 else "LATENCY_UNVERIFIED"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["latency_ms"], ensure_ascii=False))


if __name__ == "__main__":
    main()
