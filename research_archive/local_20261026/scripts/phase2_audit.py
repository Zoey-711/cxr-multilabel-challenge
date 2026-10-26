"""Audit the inspected cxr_first ZIP, then write requested Phase 2 tables."""

import argparse
import json
from pathlib import Path

from cxr.data.archive import archive_path
from cxr.data.phase2_audit import audit_archive
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audits"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    report = audit_archive(archive_path(load_config(args.config)), args.output_dir, workers=args.workers)
    print(json.dumps({"splits": {name: {k: value for k, value in data.items() if k in ("patients", "studies", "images_referenced")}
                                for name, data in report["splits"].items()}, "image_validation": report["image_validation"],
                      "cross_split": report["cross_split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
