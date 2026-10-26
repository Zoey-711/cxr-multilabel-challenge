"""Record a format-rule correction without editing sealed submission bundles."""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

from cxr.inference.versioned_submission import HISTORY_FIELDS
from cxr.utils.config import load_config


EVENT_FIELDS = ("version", "changed_at", "previous_format_status", "new_format_status", "reason")
BLOCKED = "blocked_pending_official_empty_study_rule"


def retract(root: Path, version: str, reason: str) -> None:
    if not reason.strip():
        raise ValueError("Correction reason is required")
    history = root / "submission_history.csv"
    with history.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise ValueError("Unexpected submission history format")
        rows = list(reader)
    targets = [row for row in rows if row["version"] == version]
    if len(targets) != 1:
        raise ValueError("Expected exactly one history entry for the version")
    target = targets[0]
    if target["format_status"] == BLOCKED:
        raise ValueError("Version is already marked blocked")
    if target["upload_status"] != "not_uploaded":
        raise ValueError("Upload status requires manual review before correction")
    previous = target["format_status"]
    target["format_status"] = BLOCKED
    event = {"version": version, "changed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
             "previous_format_status": previous, "new_format_status": BLOCKED,
             "reason": reason.strip()}
    events = root / "submission_status_events.csv"
    with events.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        if events.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(event)
    temporary = root / "submission_history.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/submissions"))
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    load_config(args.config)
    retract(args.output_root.resolve(), args.version, args.reason)
    print(f"Retracted format-ready status for {args.version}; sealed bundle unchanged")


if __name__ == "__main__":
    main()
