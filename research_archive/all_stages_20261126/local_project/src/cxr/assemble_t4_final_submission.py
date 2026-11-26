"""Assemble returned immutable T4 runtime and truthful compliance materials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
import shutil
import statistics
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import openpyxl

SUBMISSION_SHA = '8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def safe_path(root, name):
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe path: '+name)
    target = root.joinpath(*part.parts).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError('Escaping path: '+name)
    return target


def extract(archive, output):
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        names = set()
        total = 0
        for member in members:
            safe_path(output, member.name)
            if member.name.split('/')[0] not in ('candidate_trt109', 't4_val_reference_01'):
                raise ValueError('Unexpected archive root')
            if not member.isfile() and not member.isdir():
                raise ValueError('Links/special files prohibited')
            key = member.name.rstrip('/').casefold()
            if key in names:
                raise ValueError('Duplicate archive path')
            names.add(key)
            total += member.size
        if total > 2_000_000_000:
            raise ValueError('Unexpected archive size')
        output.mkdir(parents=True)
        for member in members:
            target = safe_path(output, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)


def check_manifest(root, manifest):
    for name, value in manifest.items():
        path = safe_path(root, name)
        expected = value['sha256'] if isinstance(value, dict) else value
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Hash mismatch: '+name)
        if isinstance(value, dict) and path.stat().st_size != value['bytes']:
            raise ValueError('Size mismatch: '+name)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def verify_return(candidate, reference):
    manifest = json.loads((candidate/'manifest_sha256.json').read_text(encoding='utf-8'))
    check_manifest(candidate, manifest)
    report = json.loads((reference/'report.json').read_text(encoding='utf-8'))
    checks = [(candidate/'manifest_sha256.json', 'source_package_manifest_sha256'),
              (candidate/'phase65_source/predict.py', 'prediction_code_sha256'),
              (reference/'timings_300.csv', 'timings_sha256'),
              (reference/'image_predictions_300.csv', 'predictions_sha256')]
    for path, key in checks:
        if sha(path) != report[key]:
            raise ValueError('Returned benchmark provenance differs: '+key)
    for name in ('submission.csv', 'submission_original_78_81.csv'):
        if sha(candidate/name) != SUBMISSION_SHA:
            raise ValueError('Historical submission differs')
    with (candidate/'submission.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['Subject_id', 'Study_id', 'Predict_class', 'Probability']:
            raise ValueError('Submission schema differs')
        rows = list(reader)
    keys = set()
    studies = {}
    for row in rows:
        key = (row['Subject_id'], row['Study_id'])
        cls, prob = int(row['Predict_class']), float(row['Probability'])
        if (key, cls) in keys or not all(key) or cls not in range(1,11) or not 0 <= prob <= 1:
            raise ValueError('Invalid submission row')
        keys.add((key, cls))
        studies.setdefault(key, set()).add(cls)
    if len(rows) != 14140 or len(studies) != 1414 or any(len(v)!=10 for v in studies.values()):
        raise ValueError('Submission coverage differs')
    timing = {}
    with (reference/'timings_300.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = row['image'], int(row['round'])
            ms = float(row['elapsed_ms'])
            if key in timing or key[1] not in (1,2,3) or not math.isfinite(ms) or ms <= 0:
                raise ValueError('Invalid timing row')
            timing[key] = ms
    names = {key[0] for key in timing}
    if len(timing)!=300 or len(names)!=100 or any((n,r) not in timing for n in names for r in (1,2,3)):
        raise ValueError('Incomplete timing records')
    values = [statistics.mean(timing[n,r] for r in (1,2,3)) for n in sorted(names)]
    recalc = {'mean_ms':statistics.mean(values), 'p50_ms':percentile(values,.5),
              'p95_ms':percentile(values,.95), 'max_ms':max(values)}
    if any(abs(report[k]-v)>1e-6 for k,v in recalc.items()):
        raise ValueError('Reported timing disagrees with raw records')
    predictions = {}
    with (reference/'image_predictions_300.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = row['image'], int(row['round'])
            vector = [float(row[f'Prob_{i}']) for i in range(1,11)]
            if key in predictions or key not in timing or any(not 0<=v<=1 for v in vector):
                raise ValueError('Invalid predictions')
            predictions[key] = vector
    if set(predictions)!=set(timing):
        raise ValueError('Prediction/timing coverage differs')
    repeat = max(abs(a-b) for n in names for r in (2,3)
                 for a,b in zip(predictions[n,1],predictions[n,r]))
    build = json.loads((candidate/'engine_build_report.json').read_text(encoding='utf-8'))
    assets = {}
    for name in ('B01','B03','N01_EMA','S17_F3'):
        path = candidate/'phase65_source/model'/f'{name}.pt'
        assets[path.relative_to(candidate).as_posix()] = {'bytes':path.stat().st_size,'sha256':sha(path)}
    for name in ('B02','M02'):
        path = candidate/'tensorrt_engines'/f'{name}_fp32.engine'
        if sha(path)!=build['engines'][name]['sha256'] or path.stat().st_size!=build['engines'][name]['bytes']:
            raise ValueError('Engine provenance mismatch')
        assets[path.relative_to(candidate).as_posix()] = {'bytes':path.stat().st_size,'sha256':sha(path)}
    return {'source_manifest_files_verified':len(manifest), 'timing_records':300,
            'timing_recomputed':recalc,'internal_three_round_max_probability_difference':repeat,
            'submission_rows':len(rows),'submission_studies':len(studies),
            'historical_submission_sha256':SUBMISSION_SHA,'inference_assets':assets,
            'inference_weights_bytes':sum(v['bytes'] for v in assets.values()),
            'environment':build['environment'], 'historical_test_consistency':'NOT_TESTED',
            'official_fixed_100_consistency':'NOT_TESTED','official_compliance':'UNVERIFIED'}


def predict_wrapper(parent):
    return f'''"""Delegate to the unchanged T4 benchmark runtime."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True
ROOT = {parent}
RUNTIME = ROOT / "model"
sys.path.insert(0, str(RUNTIME))
_spec = importlib.util.spec_from_file_location("_frozen_t4_predict", RUNTIME / "predict.py")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)
load_models = _entry.load_models
predict_image = _entry.predict_image
PRECISION = _entry.PRECISION
'''


def assemble(cfg):
    archive, material, work, output = (Path(cfg[k]).resolve() for k in ('archive','materials','work','output'))
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Final output already exists')
    extract(archive, work)
    candidate, reference = work/'candidate_trt109', work/'t4_val_reference_01'
    audit = verify_return(candidate, reference)
    material_manifest = json.loads((material/'manifest_sha256.json').read_text(encoding='utf-8'))
    check_manifest(material, material_manifest)
    output.mkdir(parents=True)
    shutil.copytree(candidate, output/'model')
    shutil.copytree(candidate/'code', output/'code')
    for folder in ('licenses','pretrain_model','materials'):
        shutil.copytree(material/folder, output/folder)
    shutil.copytree(reference, output/'materials/t4_val_reference_01')
    shutil.copy2(candidate/'submission.csv', output/'submission.csv')
    (output/'predict.py').write_text(predict_wrapper('Path(__file__).resolve().parent'), encoding='utf-8')
    (output/'code/predict.py').write_text(predict_wrapper('Path(__file__).resolve().parents[1]'), encoding='utf-8')
    (output/'cost.py').write_text('''"""Run the unchanged cost implementation beside the frozen runtime predict.py."""
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True
RUNTIME = Path(__file__).resolve().parent / "model"
sys.path.insert(0, str(RUNTIME))
if __name__ == "__main__":
    runpy.run_path(str(RUNTIME / "cost.py"), run_name="__main__")
''', encoding='utf-8')
    requirements = '''# Python 3.12; CUDA12.8/cuDNN9. Use the organizer image when available.
# These are the observed T4 self-test versions, not proof of image identity.
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.10.0+cu128
torchvision==0.25.0+cu128
tensorrt-cu12==10.9.0.34
onnxruntime-gpu==1.26.0
numpy==2.5.4
Pillow==13.0.0
'''
    (output/'requirements.txt').write_text(requirements, encoding='utf-8')
    (output/'code/requirements.txt').write_text(requirements, encoding='utf-8')
    runtime_bytes = sum(p.stat().st_size for p in (output/'model').rglob('*') if p.is_file())
    adapters_bytes = sum((output/p).stat().st_size for p in ('cost.py','predict.py','code/predict.py'))
    audit['runtime_unchanged_after_relocation'] = True
    audit['conservative_runtime_and_adapter_bytes'] = runtime_bytes + adapters_bytes
    audit['size_limit_bytes'] = 500_000_000
    audit['size_status'] = 'PASS' if runtime_bytes+adapters_bytes<=500_000_000 else 'FAIL'
    if audit['size_status']!='PASS':
        raise ValueError('Runtime exceeds size budget')
    check_manifest(output/'model', json.loads((candidate/'manifest_sha256.json').read_text(encoding='utf-8')))
    audit['return_archive_sha256'] = sha(archive)
    audit['package_status'] = 'PACKAGED_WITH_DECLARED_OPEN_ITEMS'
    audit['full_test_required_by_supplied_rules'] = False
    audit['remaining_official_items'] = ['Official fixed-100 Test consistency / F2',
        'Pretrained weight redistribution evidence / A3',
        'Team confirmation of training/data declarations']
    audit['official_resource_quota_verified'] = False
    audit['notes'] = ['Reported 96 CPUs and host RAM are not proof of allocated CPU/RAM quota.',
                      'No evidence of an 8-CPU/32GB constrained run or official container digest.',
                      'No full-Test precondition is introduced; optional full-Test tools are preserved.',
                      'Probability repeatability on Val is not historical Test submission consistency.']
    dump(output/'materials/assembly_audit.json', audit)
    values = json.loads((material/'materials/declaration_values.json').read_text(encoding='utf-8'))
    values['F1'] = ('T4 Val参考自测原始报告及300条记录已回传并独立复算：均值84.46236722171307ms，'
        'P50=83.62817019224167ms，P95=94.34345836440723ms；另5张预热、100图×3轮，batch=1。'
        '精度见A6。包含读图至图像预测写出/CUDA同步；样本均为完整单图Study。'
        '该次为benchmark_six_model_self_t4.py调用同一predict入口的Val参考测速，非cost.py对官方100张Test核验；'
        '环境版本匹配但未记录官方镜像摘要和8核/32GB资源约束。见materials/t4_val_reference_01。')
    values['F3'] = (f"实际六份推理权重/引擎={audit['inference_weights_bytes']}字节；"
        f"保守将整个不可变运行目录及入口适配代码全部计入={audit['conservative_runtime_and_adapter_bytes']}字节，"
        '低于500000000字节。清单及SHA见materials/assembly_audit.json。'
        '外层pretrain_model原始初始化文件不参与最终推理，不计入该限额；ZIP总大小不是模型大小口径。')
    values['A6'] = ('四模型PyTorch FP16 CUDA Graph，B02/M02 TensorRT10.9.0.34 FP32；已在租用T4构建并加载。'
                    'batch=1，384输入，固定六模型/融合/Study Mean；指定100张Test一致性尚未核验。')
    values['B1'] = values['B1'].replace('签署确认', '确认')
    workbook = openpyxl.load_workbook(material/'模型核验材料与数据申明清单.xlsx')
    for row in workbook.active.iter_rows(min_row=2):
        if row[0].value in values:
            workbook.active.cell(row[0].row,5,values[row[0].value])
    workbook.save(output/'模型核验材料与数据申明清单.xlsx')
    dump(output/'materials/declaration_values.json', values)
    readme = f'''# 王卓颖：最终提交材料

打包完成；状态为 PACKAGED_WITH_DECLARED_OPEN_ITEMS，不能称为官方已验收。
历史最高78.81分 submission.csv 原字节保留，SHA256：{SUBMISSION_SHA}。
最终固定融合：0.75 E08 + 0.25 S17-F3；E08=0.5 E03+0.5 N01 EMA；
E03=0.5 E01+0.5 M02；E01=(B01+B02+B03)/3。逐图384、batch=1、Study Mean，无TTA。

目录：
- code/：训练代码、各模型原配置/记录，以及正式推理入口适配。
- model/：服务器原 candidate_trt109 完整不可变运行目录，含4份pt和2份T4引擎、原代码及原SHA清单。
- pretrain_model/：4份原始ImageNet初始化权重及已核验SHA，均不参与推理。
- licenses/：已取得的软件许可证；权重再分发许可依据仍需补齐，未宣称已解决。
- cost.py、predict.py：仅定位/转调 model/ 内原入口，不改变模型或预测计算。
- requirements.txt：T4自测实际核心依赖版本。
- 模型核验材料与数据申明清单.xlsx：依据用户提供更新模板填写。
- materials/：规则、来源、回传测速记录、打包审计。
- submission.csv：历史最高分原CSV；manifest_sha256.json为本包文件哈希。

请从任意目录以绝对路径运行（/work/package替换为解压目录）：
```bash
python /work/package/model/check_environment.py --config /work/package/model/config.json --output /work/environment.json
python /work/package/cost.py /work/package/submission.csv /data/pictures_100 /data/pictures_5 --output /work/cost_run --official-sample
```
只有使用主办方指定100张Test及5张预热图时才加 --official-sample。
cost.py支持 --config /work/package/model/config.json；默认即为该文件。
输出目录须新建且在model/之外。包内没有赛事原始图像或隐藏Test标签。
正常推理接口：`from predict import load_models, predict_image`；load_models一次，predict_image逐图返回10类概率。
在 code/ 中导入predict也转调同一model/predict.py。多图Study由正式cost逻辑按固定Mean处理。

官方规则第7条和补充九（三）要求100张逐样本一致性，不要求预先完成全2218张Test。
model/full_test_audit.py仅为可选自检，不是运行cost.py或提交材料的额外前置条件。
已有原运行目录的旧README、状态及历史材料保留作来源记录；当前打包状态以外层README/声明/审计为准。
不要编辑model/内文件或重建引擎来套用本次测速结果。

当前证据：T4均值84.46236722171307ms，P50=83.62817019224167ms，P95=94.34345836440723ms。
这是Val参考脚本调用同一入口的结果，不是cost.py对官方100张Test的一致性验收。
已复核300条计时、300条预测、环境和引擎哈希；全局模型小计{audit['inference_weights_bytes']}字节；
保守运行目录及适配代码合计{audit['conservative_runtime_and_adapter_bytes']}字节，低于500MB。
ZIP包含不参与推理的原始预训练文件，ZIP总大小可以大于500MB。
环境Python3.12.16、PyTorch2.10.0+cu128、cuDNN9.10.2、TensorRT10.9.0.34、ORT1.26.0。
宿主可见96核和约503GiB RAM；未记录8核/32GB配额，也没有官方镜像摘要，最终以主办方复测为准。

尚未完成：F2官方100张Test的判定/排序/概率误差核验，A3预训练权重再分发许可依据，队伍确认数据声明。
cost中0.5为既有二值审计约定，官方未在所给文档明确Gate阈值；不得擅自调阈值来通过。
若指定100张中某Study影像不完整，原cost会报告UNVERIFIED_PARTIAL_STUDY；须确认主办方核验口径，不能伪报PASS。
所有测试均为固定权重前向；没有训练、修改融合、概率修正、自动上传或排行榜提交。
'''
    (output/'README.md').write_text(readme, encoding='utf-8')
    (output/'code/TRAINING_README.md').write_text(
        '训练来源保留于scripts/train.py及scripts/batch_train.py、src/cxr/和train_records/。\n'
        '各模型原训练参数和路径见train_records；推理包不启动训练。\n'
        'code/predict.py为最终T4入口；scripts/predict.py和其他历史生成脚本只作训练/历史流程来源，'
        '不作为本次正式推理入口。\n', encoding='utf-8')
    manifest = {p.relative_to(output).as_posix(): {'bytes':p.stat().st_size,'sha256':sha(p)}
                for p in sorted(output.rglob('*')) if p.is_file()}
    dump(output/'manifest_sha256.json', manifest)
    check_manifest(output, manifest)
    zip_path = output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        expected = dict(manifest)
        expected['manifest_sha256.json'] = {'bytes':(output/'manifest_sha256.json').stat().st_size,'sha256':sha(output/'manifest_sha256.json')}
        if set(z.namelist()) != set(expected):
            raise ValueError('ZIP inventory differs')
        for name, record in expected.items():
            with z.open(name) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=record['sha256']:
                    raise ValueError('ZIP content differs: '+name)
    result = {'archive':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':sha(zip_path),
              'files':len(manifest)+1,'status':audit['package_status'],
              'size_status':audit['size_status'],'runtime_and_adapters_bytes':audit['conservative_runtime_and_adapter_bytes'],
              't4_val_reference_ms':audit['timing_recomputed']['mean_ms'],
              'official_100_test_consistency':'NOT_TESTED'}
    dump(zip_path.with_suffix('.zip.sha256.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(assemble(json.loads(args.config.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
