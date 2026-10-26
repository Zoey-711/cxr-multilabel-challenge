"""Write a factual Phase22 status snapshot without touching frozen experiments."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cxr.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RUNS = {"D17-F3": "d17_f3_20261021", "C17-F3": "c17_f3_20261021",
        "S17-F3": "s17_f3_20261021"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "configs/f17_densenet121_fold3_384.yaml").resolve():
        parser.error("Expected frozen Fold 3 configuration")
    if load_config(args.config)["data"]["train_subset"]["fold"] != 3:
        raise ValueError("Incorrect Fold")
    pipeline = ROOT / "outputs/analysis/phase17_fold3_pipeline_20261021/status.json"
    state = json.loads(pipeline.read_text(encoding="utf-8"))
    entries = {}
    for name, run_id in RUNS.items():
        run = ROOT / "outputs/experiments" / run_id
        status = json.loads((run / "status.json").read_text(encoding="utf-8")) if (
            run / "status.json").exists() else {}
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                   ] if (run / "metrics.jsonl").exists() else []
        manifest = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8")) if (
            run / "frozen_manifest.json").exists() else {}
        entries[name] = {"state": "frozen" if manifest else status.get("state", "not_started"),
                         "latest_completed_epoch": records[-1].get("epoch") if records else None,
                         "latest_val_auc": records[-1].get("val", {}).get("macro_auc") if records else None,
                         "best_epoch": manifest.get("best_epoch", status.get("best_epoch")),
                         "best_val_auc": manifest.get("official_val_macro_auc", status.get("best_auc")),
                         "oof_auc": manifest.get("oof_macro_auc"),
                         "checkpoint_sha256": manifest.get("best_checkpoint_sha256")}
    fusion_path = ROOT / "outputs/analysis/phase17_fold3_fixed_fusion_20261021/summary.json"
    fusion = json.loads(fusion_path.read_text(encoding="utf-8")) if fusion_path.exists() else None
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "pipeline": state,
              "models": entries,
              "fold3_fusion": {key: fusion[key]["macro_auc"] for key in
                               ("fold3_val", "fold3_oof", "partial_fold1_to_fold3_oof")}
              if fusion else None,
              "remote_26_tasks": "UNKNOWN: no authorized synchronized server status",
              "fold4_and_fold5": "NOT_STARTED: not authorized in this turn",
              "hidden_test_auc": None}
    out = ROOT / "outputs/analysis/phase22_status"
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Phase22 进度", "", f"记录时间 UTC：{result['checked_utc']}", "",
             f"Fold3 流水线：{state['state']}，当前步骤：{state.get('step')}", "",
             "| 模型 | 状态 | 最新完成 Epoch | 当前最佳 Val AUC | OOF AUC |",
             "| --- | --- | ---: | ---: | ---: |"]
    for name, item in entries.items():
        lines.append(f"| {name} | {item['state']} | {item['latest_completed_epoch'] or '—'} | "
                     f"{item['best_val_auc'] if item['best_val_auc'] is not None else '—'} | "
                     f"{item['oof_auc'] if item['oof_auc'] is not None else '—'} |")
    lines += ["", f"Fold3 固定融合：{result['fold3_fusion'] or '等待三模型完成'}。",
              "远程 26 项：UNKNOWN；没有朋友服务器的合法同步状态。",
              "Fold4–5：未启动。隐藏 Test 无标签，不计算 AUC。", ""]
    (out / "status.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
