"""Freeze SHA-256 and actual file bytes for the independent H17 package."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import PIL
import numpy
import torch
import torchvision

from predict import HERE, load_config, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config/deployment_config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    weight = HERE / config["model_file"]
    if sha256(weight) != config["model_sha256"]:
        raise ValueError("Frozen H17 weight SHA mismatch")
    resolved = HERE / "config/resolved_config.yaml"
    source_config_sha = "5c2bce4c1294c5d62bd096dacdc12871d3351f0fe2a737b95d98c62acf975717"
    if sha256(resolved) != source_config_sha:
        raise ValueError("Friend resolved training config SHA mismatch")
    skip = {"package_manifest.json", "SHA256SUMS.txt"}
    entries = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(HERE).as_posix()
        if relative in skip:
            continue
        entries[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    model_files = ["model/H17_02.pth", "config/deployment_config.json", "config/resolved_config.yaml"]
    actual_model_config_bytes = sum(entries[name]["bytes"] for name in model_files)
    package_without_results_bytes = sum(item["bytes"] for name, item in entries.items()
                                        if not name.startswith("results/"))
    internal = HERE / "results/internal_consistency.json"
    consistency = json.loads(internal.read_text(encoding="utf-8")) if internal.exists() else None
    candidate_status = "LOCAL_CONSISTENCY_PASS_T4_UNVERIFIED" if (
        consistency and consistency.get("status_1e-4") == "PASS" and
        consistency.get("wide_vs_submission", {}).get("status") == "PASS"
    ) else "LOCAL_CONSISTENCY_PENDING_OR_FAIL_T4_UNVERIFIED"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "candidate": "H17_DISTILL_02_BATCH1", "source_results_commit": config["source_results_commit"],
                "source_training_code_commit": config["source_training_code_commit"],
                "model_file": "model/H17_02.pth", "model_sha256": config["model_sha256"],
                "loaded_model_files": ["model/H17_02.pth"],
                "actual_model_plus_necessary_config_bytes": actual_model_config_bytes,
                "full_package_without_results_bytes": package_without_results_bytes,
                "size_limit_bytes": 500000000, "size_status": "PASS" if package_without_results_bytes <= 500000000 else "FAIL",
                "latency_status": "T4_UNVERIFIED", "candidate_status": candidate_status,
                "test_auc": None, "hidden_test_labels_used": False,
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                                "numpy": numpy.__version__, "Pillow": PIL.__version__},
                "files": entries}
    output = HERE / "package_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"{meta['sha256']}  {name}" for name, meta in entries.items()]
    lines.append(f"{sha256(output)}  package_manifest.json")
    (HERE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": candidate_status, "size_bytes": package_without_results_bytes,
                      "files": len(entries), "manifest_sha256": sha256(output)}))


if __name__ == "__main__":
    main()
