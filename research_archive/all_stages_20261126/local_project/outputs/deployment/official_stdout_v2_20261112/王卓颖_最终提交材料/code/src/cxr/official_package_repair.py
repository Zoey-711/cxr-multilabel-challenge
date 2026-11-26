"""Create an isolated official-container build kit without changing frozen assets."""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from cxr.official_deployment_tools import IMAGE, SUBMISSION_SHA, lock_manifest, sha256, write_json


README = r'''# 王卓颖：官方 compete:3.0 六模型构建与验收包

当前状态：PREPARED_PENDING_T4_BUILD。本包不是已经通过官方验收的成品。
历史78.81的 submission.csv 字节不变。固定融合仍为 0.75×E08+0.25×S17-F3；
E08=0.5×E03+0.5×N01_EMA，E03=0.5×E01+0.5×M02，E01=(B01+B02+B03)/3。
输入384、batch=1、原预处理及Study Mean不变。未训练、未改融合权重、未修改概率。

## 本次修订

- 不再安装旧 requirements-lock.txt。官方镜像预装框架是唯一目标；代码记录并检查实际版本。
- 排除旧 TensorRT 11.4 引擎，提供原始FP32 ONNX、checkpoint与SHA来源，
  在官方TensorRT 10.9/T4中构建两个FP32引擎。重建不改变训练权重，但可能改变数值，必须复验。
- 四个PyTorch模型使用原FP16 CUDA Graph；B02/M02使用FP32引擎；没有CPU回退或备用精度。
- cost.py、完整Test审计直接导入根目录predict.py，使用完全相同load_models/predict_image。
- 构建目录与推理目录分离。正式推理目录没有构建用ONNX及多余checkpoint。
- 所有命令拒绝覆盖已有输出。依赖检查不以“发现T4”代替版本/驱动检查。

## 环境

主办方镜像（用户提供）：
`crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`

Linux、Python3.12、PyTorch2.10.0 CUDA12.8/cuDNN9、TensorRT10.9.0.34、
ONNX Runtime GPU1.26.0、单张T4 16GB、驱动>=570.117、8CPU、32GB内存、>=100GB SSD。
TensorRT模块若只显示10.9.0，报告保留该值；镜像摘要和pip包记录用于核实构建版本。
没有在本机运行Docker/T4；Windows CPU测试不等于Linux或GPU验收。
requirements.txt为无安装操作的说明文件。禁止pip升级/降级镜像内框架。

## 在Linux T4服务器执行

假设已将本ZIP解压，目录命名为 `/work/official_kit`。另建输出目录：

```bash
mkdir -p /work/official_runs
docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0
docker image inspect crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0 > /work/official_runs/docker_image_inspect.json
docker run --rm --gpus '"device=0"' --cpus=8 --memory=32g --shm-size=2g \
  --entrypoint /bin/bash \
  -v /work/official_kit:/kit:ro -v /work/official_runs:/runs \
  -v /private/test_images:/images:ro \
  -v /private/official_100:/pictures_100:ro -v /private/official_5:/pictures_5:ro \
  crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0
```

`/private/...`必须替换为服务器实际目录，图片不在包内。官方100图和5张预热图不可自行替换。
容器内按顺序执行（路径均为绝对路径，可以从任意工作目录执行）：

```bash
python /kit/check_environment.py --config /kit/config.json --output /runs/environment.json
python -m pip freeze > /runs/pip_freeze.txt
python /kit/build_engines.py --config /kit/config.json --output /runs/candidate_trt109
python /runs/candidate_trt109/check_environment.py --config /runs/candidate_trt109/config.json --load-models --output /runs/model_load.json
python /runs/candidate_trt109/full_test_audit.py --images-root /images --output /runs/full_test
python /runs/candidate_trt109/cost.py /runs/candidate_trt109/submission.csv /pictures_100 /pictures_5 --output /runs/cost_01 --official-sample
```

构建只在第一次运行，耗时待T4实测，不计入推理计时。两项顺序构建，workspace每项2GiB；
关闭FP16/INT8/TF32构建标志。原有四模型FP16设置保持不变。
构建失败保留BUILD_FAILED.json，必须另选新输出目录重试，不能把半成品当成候选。
首次构建的候选目录会生成新的manifest及真实文件大小报告。

## 一致性与计时

完整Test审计针对冻结的2218图、1414 Study、14140概率，校验原图哈希。
它不修改submission.csv。cost每图重新执行六模型，5独立预热、100图连续3轮，
CUDA同步并计入读取、预处理、推理、写出/flush，以及可完成Study的Mean和写出。
先计算每图3轮均值，再计算100图均值；模型加载和预热不计时。

cost支持官方示例三个位置参数，也支持--submission/--pictures-100/--pictures-5。
默认依据包内真实完整Test映射自动生成100图索引；Val模拟需另给--image-index，
但Val模拟不能与Test submission核验，不能冒充正式验收。
如果官方100张未包含某Study全部影像，保持UNVERIFIED_PARTIAL_STUDY，
不缓存或提前推理缺失影像来加速计时，不擅自改变主办方样本。
须向主办方确认逐图片核验与Study Mean的接口关系。
0.5是沿用的内部二值检查约定，须确认官方Pred_Label/Gate定义；未确认不得宣布官方合规。
主办方样本级判定/排序不一致数必须为0，且概率误差<=1e-3。

## 验收顺序与禁止事项

环境检查通过→新引擎构建和实际加载通过→全Test一致性通过→官方100图测速与逐样本一致性。
如果全Test失败，停止将此版本作为78.81提交的部署对应物，不修改概率、阈值或融合权重补救。
新引擎未生成前不报告最终模型大小；生成后实际加载资产及必要代码/配置合计<=500000000字节。
本包包含构建源，因此体积不代表正式推理目录大小。不能把本ZIP直接当最终验收成品。
没有T4结果前：T4_UNVERIFIED；没有一致性结果前：HISTORICAL_SUBMISSION_UNVERIFIED。

## 仍需补齐

原始ImageNet预训练文件、再分发许可证明和上游数据说明；队伍签署声明；
主办方更新版Excel模板与当前模板差异核对（本地目前只有旧模板）；
官方Gate及部分Study核验口径；T4最终报告。历史证据在materials/history，只作历史记录。
训练代码、模型配置和历史来源仍随包提供。包内无原始医学图像、无Test标签、无凭据。
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Unexpected source structure: {old[:100]}')
    return text.replace(old, new, 1)


def function_dump(text, name):
    return [ast.dump(n, include_attributes=False) for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == name]


def repair(project, source, output):
    project, source, output = map(Path, (project, source, output))
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source/'materials/all_files_sha256.json').read_text(encoding='utf-8'))
    for name, record in original.items():
        if sha256(source/name) != record['sha256']:
            raise ValueError(f'Original draft modified: {name}')
    output.mkdir(parents=True)
    for path in source.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(source)
        if rel.parts[0] in ('materials', 'tensorrt_engines') or path.name in (
            'requirements-lock.txt', 'build_report.json', 'manifest_sha256.json',
            '模型核验材料与数据申明清单.xlsx'):
            continue
        target = output/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    history = output/'materials/history'
    shutil.copytree(source/'materials', history)
    shutil.copy2(source/'模型核验材料与数据申明清单.xlsx', history/'旧版声明清单_历史待验收.xlsx')
    shutil.copy2(source/'requirements-lock.txt', history/'requirements_old_do_not_install.txt')
    (output/'build_sources').mkdir()
    export_dir = project/'outputs/analysis/phase44_e08_acceleration'
    exports = json.loads((export_dir/'onnx_export_manifest.json').read_text(encoding='utf-8'))
    e08 = json.loads((output/'phase65_source/frozen_e08_manifest.json').read_text(encoding='utf-8'))
    provenance = {}
    for name in ('B02', 'M02'):
        info = e08['models'][name]
        checkpoint = project/'outputs/experiments'/info['run_id']/info['checkpoint']
        graph = export_dir/f'onnx_models/{name}_fp32.onnx'
        if sha256(checkpoint) != info['checkpoint_sha256'] or exports[name]['source_sha256'] != info['checkpoint_sha256']:
            raise ValueError(f'Frozen checkpoint provenance mismatch: {name}')
        if sha256(graph) != exports[name]['fp32_sha256']:
            raise ValueError(f'Frozen ONNX hash mismatch: {name}')
        shutil.copy2(checkpoint, output/f'build_sources/{name}.pt')
        shutil.copy2(graph, output/f'build_sources/{name}_fp32.onnx')
        provenance[name] = {'checkpoint_sha256': sha256(checkpoint), 'onnx_sha256': sha256(graph),
                            'checkpoint_run_id': info['run_id'], 'opset': exports[name]['opset'],
                            'input_shape': [1,3,384,384], 'input_name': 'image', 'output_name': 'logits',
                            'source': 'Frozen Phase44 FP32 export from exact frozen E08 checkpoint',
                            'build_input_rename': 'image -> input (name only; no tensor/value transform)'}
    write_json(output/'build_sources/provenance.json', provenance)
    tools_source = project/'src/cxr/official_deployment_tools.py'
    shutil.copy2(tools_source, output/'phase65_source/src/cxr/official_deployment_tools.py')
    for action, filename in (('check','check_environment.py'), ('build','build_engines.py')):
        (output/filename).write_text(
            'from pathlib import Path\nimport sys\nsys.dont_write_bytecode = True\n'
            'ROOT = Path(__file__).resolve().parent\n'
            'sys.path.insert(0, str(ROOT / "phase65_source/src"))\n'
            'from cxr.official_deployment_tools import main\n'
            f'if __name__ == "__main__": main("{action}", ROOT)\n', encoding='utf-8')
    predict_path = output/'phase65_source/predict.py'
    before = predict_path.read_text(encoding='utf-8')
    after = replace_once(before, 'def load_models():\n',
        'def load_models():\n'
        '    from cxr.official_deployment_tools import require_environment, require_engines\n'
        '    import random\n'
        '    require_environment()\n'
        '    require_engines(Path(__file__).resolve().parent.parent)\n'
        '    random.seed(42)\n    np.random.seed(42)\n    torch.manual_seed(42)\n'
        '    torch.cuda.manual_seed_all(42)\n    torch.backends.cudnn.benchmark = False\n')
    after = replace_once(after, 'PRECISION = "PyTorch FP16 autocast + CUDA Graph"',
        'PRECISION = "B01/B03/N01_EMA/S17_F3 PyTorch FP16 CUDA Graph; B02/M02 TensorRT FP32"')
    # Warmup has no input data or predictions: initialize rather than read uninitialized GPU memory.
    after = replace_once(after, 'static_x = torch.empty(', 'static_x = torch.zeros(')
    for name in ('gray_first_preprocess', 'predict_image'):
        if function_dump(before, name) != function_dump(after, name):
            raise ValueError(f'Prediction arithmetic changed: {name}')
    predict_path.write_text(after, encoding='utf-8')
    for filename in ('e08_s17_t4_cost.py', 'e08_s17_t4_full_audit.py'):
        path = output/'phase65_source/src/cxr'/filename
        text = path.read_text(encoding='utf-8')
        text = replace_once(text, 'from phase65_source import predict', 'sys.path.insert(0, str(ROOT))\nimport predict')
        if filename == 'e08_s17_t4_cost.py':
            text = replace_once(text, "ap.add_argument('--config',type=Path,required=True)",
                "ap.add_argument('submission_pos',type=Path,nargs='?')\n"
                "    ap.add_argument('pictures_100_pos',type=Path,nargs='?')\n"
                "    ap.add_argument('pictures_5_pos',type=Path,nargs='?')\n"
                "    ap.add_argument('--config',type=Path,default=ROOT/'config.json')")
            for flag in ('pictures-100','pictures-5','image-index','submission'):
                text = replace_once(text, f"ap.add_argument('--{flag}',type=Path,required=True)",
                                     f"ap.add_argument('--{flag}',type=Path)")
            text = replace_once(text, "ap.add_argument('--output',type=Path,required=True)",
                "ap.add_argument('--output',type=Path,default=ROOT.parent/('cost_'+str(time.time_ns())))")
            text = replace_once(text, 'args=ap.parse_args()',
                "args=ap.parse_args()\n"
                "    for key in ('submission','pictures_100','pictures_5'):\n"
                "        flag=getattr(args,key); pos=getattr(args,key+'_pos')\n"
                "        if flag and pos: ap.error('Use positional or named argument, not both: '+key)\n"
                "        setattr(args,key,flag or pos)\n"
                "        if not getattr(args,key): ap.error('Missing '+key)\n"
                "    if args.config.resolve() != (ROOT/'config.json').resolve(): raise ValueError('Use frozen config')\n"
                "    if args.output.resolve().is_relative_to(ROOT): raise ValueError('Output must be outside package')")
            text = replace_once(text, "grouped,incomplete=sample_index(args.image_index,measured)",
                "if args.image_index:\n"
                "        grouped,incomplete=sample_index(args.image_index,measured)\n"
                "    else:\n"
                "        from cxr.prediction_consistency import read_index\n"
                "        full=read_index(ROOT/'test_image_index.csv')\n"
                "        wanted={p.name for p in measured}\n"
                "        rows=[r for r in full if r['image'] in wanted]\n"
                "        if len(rows)!=100: raise ValueError('Timed images absent from frozen Test mapping')\n"
                "        grouped={}; totals={}\n"
                "        for row in rows:\n"
                "            key=row['Subject_id'],row['Study_id']\n"
                "            grouped.setdefault(key,[]).append(row['image'])\n"
                "            totals[key]=int(row['study_image_count'])\n"
                "        incomplete={k:(len(v),totals[k]) for k,v in grouped.items() if len(v)!=totals[k]}")
        path.write_text(text, encoding='utf-8')
    # Entry wrappers must not create cache files in read-only container bind mounts.
    for name in ('predict.py','cost.py','full_test_audit.py'):
        p = output/name
        text = p.read_text(encoding='utf-8').replace('import sys\n', 'import sys\nsys.dont_write_bytecode = True\n')
        p.write_text(text, encoding='utf-8')
    (output/'README.md').write_text(README, encoding='utf-8')
    (output/'提交前必读.md').write_text(
        '# 王卓颖：官方环境修订包，待T4构建验收\n\n详见README.md。旧引擎已排除，'
        '本包须在官方镜像内构建后使用。历史78.81 Submission未变；'
        '没有宣称预测已一致、T4速度合格或许可材料齐全。\n', encoding='utf-8')
    (output/'requirements.txt').write_text(
        '# Do NOT pip install/upgrade the official container framework.\n'
        '# Base image: '+IMAGE+'\n# Python 3.12; torch 2.10.0 CUDA12.8/cuDNN9; TensorRT10.9.0.34; ORT GPU1.26.0.\n'
        '# No additional dependency installation is performed by this package.\n'
        '# check_environment.py reports missing imports and stops.\n', encoding='utf-8')
    write_json(output/'official_environment.json', {'source': 'User-provided organizer server specification 2026-11-10',
        'image': IMAGE, 'python': '3.12', 'torch': '2.10.0', 'cuda': '12.8', 'cudnn_major': 9,
        'tensorrt': '10.9.0.34', 'onnxruntime_gpu': '1.26.0', 'driver_min': '570.117',
        'gpu': 'NVIDIA T4 16GB x1', 'cpu_cores': 8, 'ram_gb': 32, 'ssd_gb_min': 100,
        'image_digest': 'PENDING_TARGET_DOCKER_INSPECT'})
    write_json(output/'package_status.json', {'status': 'PREPARED_PENDING_T4_BUILD',
        'team': '王卓颖', 'submission_sha256': SUBMISSION_SHA,
        't4': 'UNVERIFIED', 'engines': 'NOT_BUILT_FOR_TRT_10_9', 'source_draft_unchanged': True})
    write_json(output/'repair_report.json', {
        'preprocessing_function_ast_equal': True, 'both_predict_image_function_asts_equal': True,
        'fusion_formula_unchanged': True, 'historical_submission_byte_identical': sha256(output/'submission.csv') == SUBMISSION_SHA,
        'init_change': 'Zero-initialized synthetic CUDA Graph warmup tensor; no image or output transformation',
        'environment': 'Official versions strictly checked; no dependency installs',
        'old_trt_engines_shipped': False, 'test_image_bytes_shipped': False,
        'numerical_consistency': 'PENDING_TARGET_INFERENCE',
        'known_blockers': ['No local Docker/T4/TensorRT10.9', 'Historical Test batch1 ordering unverified',
                           'Official Gate/partial-Study interface', 'Pretrained redistribution evidence', 'Updated XLSX not provided']})
    values = json.loads((source/'materials/declaration_values.json').read_text(encoding='utf-8'))
    values['A6'] = '固定四模型PyTorch FP16 CUDA Graph、B02/M02 TensorRT FP32；仅在官方TensorRT10.9镜像内构建；尚未完成历史78.81预测一致性。'
    values['F1'] = '待验：官方compete:3.0 T4环境，5张预热、100张3轮；构建包没有T4实测。'
    values['F2'] = '待验：历史CSV字节冻结，新引擎完成后须按同一predict/cost入口核对；0.5 Gate只是内部约定，官方口径待确认。'
    values['F3'] = '待目标引擎构建后实际统计；四个checkpoint加两个TensorRT10.9引擎及必要文件须<=500000000字节。构建源不用于推理，输出候选目录会将其排除。'
    write_json(output/'materials/declaration_values.json', values)
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    keys = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2','D1','D2','D3','D4','D5','E1','E2','E3','F1','F2','F3']
    with zipfile.ZipFile(source/'模型核验材料与数据申明清单.xlsx') as inp, zipfile.ZipFile(output/'模型核验材料与数据申明清单.xlsx','x') as out:
        for entry in inp.infolist():
            data = inp.read(entry.filename)
            if entry.filename == 'xl/worksheets/sheet1.xml':
                tree = ET.fromstring(data)
                for row, key in enumerate(keys, 2):
                    cell = next(c for c in tree.iter(f'{{{ns}}}c') if c.get('r') == f'E{row}')
                    cell.clear(); cell.set('r', f'E{row}'); cell.set('t','inlineStr')
                    ET.SubElement(ET.SubElement(cell,f'{{{ns}}}is'),f'{{{ns}}}t').text = values[key]
                data = ET.tostring(tree, encoding='utf-8', xml_declaration=True)
            out.writestr(entry, data)
    lock_manifest(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    print(repair(cfg['project'], cfg['source'], cfg['output']))


if __name__ == '__main__':
    main()
