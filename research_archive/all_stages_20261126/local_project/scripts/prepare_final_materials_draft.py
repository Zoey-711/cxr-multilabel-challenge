"""Build an evidence-backed, explicitly unapproved final-materials draft."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", NS)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source: Path, target: Path, expected: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if expected and digest(source) != expected:
        raise ValueError(f"Source SHA-256 differs: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise ValueError(f"Copy differs: {target}")


def fill_template(template: Path, destination: Path, values: dict[str, str]) -> None:
    with zipfile.ZipFile(template) as source, zipfile.ZipFile(destination, "x") as target:
        for entry in source.infolist():
            content = source.read(entry.filename)
            if entry.filename == "xl/worksheets/sheet1.xml":
                root = ET.fromstring(content)
                changed = set()
                for row in root.findall(f".//{{{NS}}}sheetData/{{{NS}}}row"):
                    number = int(row.attrib["r"])
                    key = {2: "A1", 3: "A2", 4: "A3", 5: "A4", 6: "A5", 7: "A6",
                           8: "B1", 9: "B2", 10: "B3", 11: "C1", 12: "C2",
                           13: "D1", 14: "D2", 15: "D3", 16: "D4", 17: "D5",
                           18: "E1", 19: "E2", 20: "E3", 21: "F1", 22: "F2", 23: "F3"}.get(number)
                    if not key:
                        continue
                    cell = next((c for c in row.findall(f"{{{NS}}}c") if c.attrib.get("r") == f"E{number}"), None)
                    if cell is None:
                        cell = ET.SubElement(row, f"{{{NS}}}c", {"r": f"E{number}"})
                    cell.clear()
                    cell.set("r", f"E{number}")
                    cell.set("t", "inlineStr")
                    text = ET.SubElement(ET.SubElement(cell, f"{{{NS}}}is"), f"{{{NS}}}t")
                    text.text = values[key]
                    changed.add(key)
                if changed != set(values):
                    raise ValueError(f"Template rows missing: {set(values) - changed}")
                content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(entry, content)
    with zipfile.ZipFile(destination) as workbook:
        xml = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        fields = [cell for cell in xml.iter(f"{{{NS}}}c") if cell.attrib.get("r", "").startswith("E")]
        if len(fields) < len(values):
            raise ValueError("Filled workbook lost declaration cells")


def build(project: Path, template: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    runtime = project / "outputs/deployment/e08_s17_t4_batch1_full_audit_v2_20261110"
    runtime_manifest = json.loads((runtime / "manifest_sha256.json").read_text(encoding="utf-8"))
    for relative, expected in runtime_manifest.items():
        copy_file(runtime / relative, output / relative, expected)
    copy_file(runtime / "manifest_sha256.json", output / "manifest_sha256.json")
    (output / "README.md").write_text(
        "# 王卓颖六模型部署入口：待目标 T4 验收\n\n"
        "请先阅读 `提交前必读.md`。本目录提供历史 78.81 提交对应的六模型候选入口，"
        "并未取得目标 T4 的完整预测一致性与官方计时通过证明。\n\n"
        "在目标 T4 的独立环境安装 `requirements-lock.txt`，运行 `python check_environment.py`；"
        "然后按 `提交前必读.md` 的命令先做完整 Test 一致性，再做官方固定100张测速。\n",
        encoding="utf-8")
    runtime_manifest["README.md"] = digest(output / "README.md")
    (output / "manifest_sha256.json").write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    historical = project / "outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv"
    copy_file(historical, output / "submission.csv",
              "8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c")
    # Keep the internal runtime reference with the exact hash checked by cost.py.
    if digest(output / "submission_original_78_81.csv") != digest(output / "submission.csv"):
        raise ValueError("Runtime reference and highest-scoring submission differ")

    for relative in ("scripts/train.py", "scripts/batch_train.py", "scripts/predict.py",
                     "scripts/generate_e03_submission.py", "scripts/generate_e08_ema_candidate.py",
                     "scripts/generate_e08_s17_f3_candidate.py"):
        copy_file(project / relative, output / "code" / relative)
    for source in sorted((project / "src/cxr").rglob("*.py")):
        copy_file(source, output / "code" / source.relative_to(project))

    e08 = json.loads((runtime / "phase65_source/frozen_e08_manifest.json").read_text(encoding="utf-8"))
    runs = [(name, info["run_id"]) for name, info in e08["models"].items()]
    runs.append(("S17_F3", "s17_f3_20261021"))
    for name, run_id in runs:
        folder = project / "outputs/experiments" / run_id
        for filename in ("config.yaml", "pretrained_source.json", "frozen_manifest.json",
                         "status.json", "environment.json", "best_metrics.json"):
            source = folder / filename
            if source.is_file():
                copy_file(source, output / "code/train_records" / name / filename)

    copy_file(project / "outputs/deployment/phase53_t4_benchmark/licenses/torchvision_LICENSE_BSD3.txt",
              output / "licenses/torchvision_LICENSE_BSD3.txt")
    copy_file(project / "outputs/submissions/E08_075_S17_F3_025_20261110/manifest.json",
              output / "materials/historical_submission_manifest.json")
    copy_file(project / "outputs/analysis/e08_s17_historical_val_recovery_20261110/historical_test_provenance/report.json",
              output / "materials/historical_test_provenance.json")
    copy_file(project / "outputs/analysis/e08_s17_consistency_standardization_20261110/test_audit/audit_report.json",
              output / "materials/batch1_4060_full_test_audit.json")
    copy_file(project / "outputs/analysis/e08_s17_historical_val_recovery_20261110/rules_package_audit_20261110.md",
              output / "materials/rules_package_audit.md")

    pretraining = {}
    for name, _ in runs:
        source = output / "code/train_records" / name / "pretrained_source.json"
        if source.is_file():
            pretraining[name] = json.loads(source.read_text(encoding="utf-8"))
    unique = {entry["url"]: entry for entry in pretraining.values()}
    (output / "pretrain_model").mkdir()
    (output / "pretrain_model/recorded_sources.json").write_text(
        json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "pretrain_model/README_PENDING.md").write_text(
        "# 预训练原始权重待核验\n\n训练记录提供了 torchvision ImageNet 权重的公开 URL、SHA-256 和大小。"
        "本机未找到四种原始 ImageNet 权重文件，故没有把推理 checkpoint 冒充预训练权重。"
        "正式递交前须下载原始公开权重、逐一验 SHA-256，并核实各权重的再分发许可及上游数据说明。\n",
        encoding="utf-8")

    values = {
        "A1": "DenseNet121（B01/B02/B03）、ConvNeXt-Tiny（M02）、EfficientNetV2-S（N01 EMA）、Swin-Tiny（S17-F3）；均为 torchvision ImageNet 初始化，训练记录见 code/train_records。",
        "A2": "四种公开权重的 URL 见 pretrain_model/recorded_sources.json；正式递交前核对公网可访问性。",
        "A3": "待核验：附有 torchvision 软件 BSD-3-Clause 全文，不能据此认定四份预训练权重的再分发许可。",
        "A4": "训练记录显示 ImageNet-1K；四种权重的官方 model card/训练说明尚待逐项附齐。",
        "A5": "训练记录中的四种 SHA-256 见 pretrain_model/recorded_sources.json；本机未取得原始文件逐字节复核，待核验。",
        "A6": "历史提交由 PyTorch FP16 batch=8/16 生成；候选部署为 B01/B03/N01/S17 PyTorch FP16 CUDA Graph，B02/M02 TensorRT FP32；两者一致性待目标 T4 全量复核。",
        "B1": "训练记录指向官方训练集及官方验证集；未发现外部医学数据或隐藏 Test 训练记录；最终数据来源仍需队伍签署确认。",
        "B2": "官方 Val 曾用于早停、checkpoint 选择与候选分析；未并入这些模型的训练。历史 Val 反复使用，结论属探索性。",
        "B3": "各模型的数据增强与训练参数不同，逐项以 code/train_records 中的原始 config.yaml 为准；推理入口关闭随机增强。",
        "C1": "这些六模型的训练记录未申报官方 Train 自监督续训；需队伍最终确认。",
        "C2": "这些六模型的训练记录未申报额外外部无标签数据；需队伍最终确认。",
        "D1": "六个推理模型未使用知识蒸馏教师。",
        "D2": "不适用：无蒸馏教师。",
        "D3": "不适用：无蒸馏教师。",
        "D4": "不适用：无蒸馏教师参与推理。",
        "D5": "不适用：无知识蒸馏或软标签生成。",
        "E1": "已有代码及审计未显示 Test 训练/统计拟合；Test 用于生成提交并核验来源。具体数据使用由队伍确认。",
        "E2": "部署入口使用冻结参数，PyTorch 模型 eval，Study Mean 为固定聚合；T4 TensorRT 路径待验收。",
        "E3": "六模型固定融合：0.375×E03+0.375×N01 EMA+0.25×S17-F3，E03=0.5×E01+0.5×M02，E01为B01/B02/B03均值；每 Study 多图概率算术平均；无 TTA。",
        "F1": "待验：当前六模型混合入口尚无真实 T4 官方100张×3轮结果。本地4060纯PyTorch入口53.79ms不适用。",
        "F2": "待验：4060纯PyTorch batch=1全 Test 相比历史提交有10个 Study 排序变化；T4 混合入口尚未全量运行。",
        "F3": "实际加载四个 PyTorch checkpoint、两个 TensorRT 引擎及必要配置，合计392800139字节，低于500000000字节；见 build_report.json。",
    }
    fill_template(template, output / "模型核验材料与数据申明清单.xlsx", values)
    (output / "materials/declaration_values.json").write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = """# 王卓颖最终提交材料：待验收草稿

历史最高分 CSV 为根目录 `submission.csv`（用户确认官方78.81），SHA-256 为
`8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c`。
模型固定为 0.75×E08 EMA + 0.25×S17-F3，共六个骨干。

## 当前状态

**不可标为正式合规提交**。历史 batch=8/16 路径精确复算了 CSV，但官方第7条要求
batch=1。4060 纯 PyTorch batch=1 全 Test 有10个 Study 类别排序变化；本包 T4
混合 TensorRT/PyTorch 路径尚未在目标 T4 完成全 Test 一致性或官方100张测速。
主办方要求100张样本逐张判定和类别排序一致；T4 未实测不得填写合格速度。

## 目录

- `submission.csv`：历史最高分文件；`submission_original_78_81.csv` 为同字节推理参考。
- `predict.py` / `cost.py` / `full_test_audit.py`：同一六模型部署入口与核验工具。
- `phase65_source/model/` 和 `tensorrt_engines/`：实际加载的四份 checkpoint 与两个引擎。
- `code/`：训练、生成、分析源码和六次训练的冻结配置/来源记录。
- `pretrain_model/`：四种 ImageNet 原始权重来源记录，原始文件及许可待补。
- `licenses/`：已获得的 torchvision 软件许可证，权重许可待核。
- `模型核验材料与数据申明清单.xlsx`：按模板填写已证实项，其余明确为待核验。
- `materials/`：证据与 SHA-256 清单。

## 目标 T4 检验

依赖及命令见 `README.md`。该 README 原有解压文件名过时，须以当前草稿包名为准。
先将官方 Test 原图合法放在私有路径：

```bash
python check_environment.py
python full_test_audit.py --images-root /private/test_images --output /private/reports/full_test
python cost.py --config config.json --pictures-100 /private/pictures_100 --pictures-5 /private/pictures_5 --image-index /private/timed_image_index.csv --submission submission.csv --output /private/reports/cost_01 --official-sample
```

`full_test_audit.py` 需要2218张原始 Test 图的完整目录，必须获得 `PASS`，包括0个排序、判定和概率超差 Study。`cost.py` 的官方固定100张还需要逐样本核验；如样本只覆盖某 Study 的部分图，当前程序明确返回 `UNVERIFIED_PARTIAL_STUDY`，须按主办方接口确定核对方式。不可为了通过检查改历史 CSV、融合权重或图片抽样。

## 正式递交前缺项

1. 在目标 T4 跑完整 Test 一致性及官方100张测速，保留原始报告和环境信息。
2. 补齐四种预训练权重原件、SHA-256 实测、许可证全文及上游训练数据说明；确认再分发权利。
3. 核对主办方的 Gate/Pred_Label 定义及多图 Study 逐样本核验方式。
4. 队伍核对并签署数据使用和模型来源声明。只有全部通过后，才能重新生成不带“待验收”的正式材料包。
"""
    (output / "提交前必读.md").write_text(readme, encoding="utf-8")
    # Keep runtime manifest exactly as verified; this manifest covers all added materials.
    all_files = {str(path.relative_to(output)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": digest(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"}
    (output / "materials/all_files_sha256.json").write_text(json.dumps(all_files, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "DRAFT_BLOCKED", "team": "王卓颖", "files": len(all_files),
            "submission_sha256": digest(output / "submission.csv"),
            "runtime_assets_and_config_bytes": json.loads((output / "build_report.json").read_text(encoding="utf-8"))["loaded_and_config_bytes"],
            "missing": ["T4 six-model full Test consistency", "official fixed-100 latency and per-image consistency",
                        "four original pretrained weights and redistribution-license checks", "team declaration sign-off"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.project, args.template, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
