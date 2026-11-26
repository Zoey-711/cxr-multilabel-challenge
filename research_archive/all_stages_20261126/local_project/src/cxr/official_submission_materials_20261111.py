"""Prepare truthful official declaration materials without changing frozen runtime files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import fitz
import openpyxl
from openpyxl.styles import Alignment


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(config):
    project, rules, output = (Path(config[k]) for k in ('project', 'rules', 'output'))
    source = project / 'outputs/deployment/王卓颖_官方环境修订构建包_ASCII_20261111'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for sub in ('pretrain_model', 'licenses', 'materials/rules'):
        (output / sub).mkdir(parents=True)
    sources = json.loads((source/'pretrain_model/recorded_sources.json').read_text(encoding='utf-8'))
    for url, entry in sources.items():
        path = project/'.wheels/torch_cache/hub/checkpoints'/url.rsplit('/', 1)[-1]
        if path.stat().st_size != entry['size_bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Pretrained source mismatch: '+str(path))
        target = output/'pretrain_model'/path.name
        shutil.copy2(path, target)
        if digest(target) != entry['sha256']:
            raise ValueError('Copied pretrained source differs')
        entry['local_original_verified'] = True
        entry['package_path'] = target.relative_to(output).as_posix()
    (output/'pretrain_model/sources_sha256.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(source/'licenses/torchvision_LICENSE_BSD3.txt', output/'licenses/torchvision_LICENSE_BSD3.txt')
    for name in ('比赛参赛规则.pdf', '关于比赛参赛规则若干事项的补充说明 (1).pdf'):
        shutil.copy2(rules/name, output/'materials/rules'/name)
        with fitz.open(rules/name) as doc:
            text = '\n\n'.join(f'第{i+1}页\n{page.get_text()}' for i, page in enumerate(doc))
        (output/'materials/rules'/(Path(name).stem+'.txt')).write_text(text, encoding='utf-8')
    template = rules/'模型核验材料与数据申明清单模板 (2).xlsx'
    shutil.copy2(template, output/'materials/rules/原始声明模板.xlsx')
    values = json.loads((source/'materials/declaration_values.json').read_text(encoding='utf-8'))
    values.update({
        'A2': '四种公开权重URL见 pretrain_model/sources_sha256.json。对应原始文件随包提供；无需登录的来源为 download.pytorch.org/models。',
        'A3': '附 licenses/torchvision_LICENSE_BSD3.txt 软件许可证全文。权重可能另受训练数据条款约束，不能仅据软件许可证确认全部权重再分发许可；仍需补足该许可依据。',
        'A4': '四种权重为 torchvision 的 IMAGENET1K_V1。官方模型文档链接见 pretrain_model/上游说明.md；没有医学预训练来源。',
        'A5': '四份原始预训练权重已从本地缓存取得，逐字节SHA-256均与冻结来源记录一致；完整哈希及文件字节数见 pretrain_model/sources_sha256.json。',
        'A6': 'B01/B03/N01_EMA/S17_F3：PyTorch FP16 CUDA Graph；B02/M02：TensorRT 10.9.0.34 FP32，禁用FP16/INT8/TF32引擎构建。输入384，batch=1。',
        'E2': '沿用冻结推理入口，eval、固定随机种子、无随机增强，无训练/梯度更新；各图独立预处理，Study Mean固定。新T4入口对历史Submission的一致性尚未核验。',
        'F1': '用户回传T4 Val参考自测：batch=1，另5张预热、100张连续3轮，共300记录；每图三轮均值后总体均值84.46236722171307ms，P50=83.62817019224167ms，P95=94.34345836440723ms。含读图至预测写出及CUDA同步。精度见A6。使用自行配置的指定版本环境及Val样本，非主办方固定100张Test；原始报告待回传归档。',
        'F2': '尚未执行主办方指定100张Test的逐样本一致性核验，不填写0不一致或PASS。cost.py与predict.py共用入口。须填实测判定/排名不一致数及最大概率误差（≤1e-3且判定/排名完全相同）；当前0.5二值阈值为内部审计约定，官方Gate口径须确认。全Test检查为自愿额外自检，并非本项必需材料。',
        'F3': '用户回传的两个T4引擎合计144088744字节，加已核验四个PyTorch权重248998985字节，六个推理文件合计393087729字节；这是模型文件小计，必要配置另计，最终总量待回传T4文件后复核。原始预训练文件不参与推理，不计入500MB限制。',
    })
    workbook = openpyxl.load_workbook(template)
    sheet = workbook.active
    seen = set()
    for row in sheet.iter_rows(min_row=2):
        key = row[0].value
        if key in values:
            sheet.cell(row[0].row, 5, values[key]).alignment = Alignment(wrap_text=True, vertical='top')
            seen.add(key)
    if seen != set(values):
        raise ValueError('Template keys differ: '+str(set(values)-seen))
    workbook.save(output/'模型核验材料与数据申明清单.xlsx')
    (output/'materials/declaration_values.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    text = '''# 王卓颖：提交材料整理说明

依据：比赛参赛规则第7、8、10条及附录A；补充说明第九（三）；更新声明模板F1/F2/F3。

官方要求对主办方固定100张Test逐张核验并测速，另5张预热、batch=1、连续3轮。
没有要求参赛者必须在提交前完成全部2218张Test自检。全Test脚本可保留为可选工具，不能列为官方必交报告。
预测可复现和100张的一致性要求仍然有效；不得以未做全Test为由填写一致性PASS。

本目录是已整理的材料补充，尚不是可直接提交的完整运行包。
最终ZIP建议命名 王卓颖_最终提交材料.zip；包含历史最高78.81分原始submission.csv，
训练/推理代码及依赖说明、实际使用的四个PyTorch权重和两个T4 TensorRT引擎、必要配置、
原始预训练权重、许可证、来源/数据说明、更新声明Excel和文件哈希。
保持服务器candidate_trt109的推理逻辑/模型/融合/预处理不变，不用本地旧引擎替代T4引擎。

已完成：读取两份规则与更新模板；补齐四份本地原始预训练文件并验证来源SHA；更新声明。
已知T4速度：84.46236722171307ms，来源为用户粘贴的Val参考报告，非官方Test一致性验收。
待回传：candidate_trt109完整目录、t4_val_reference_01目录及实际环境/加载报告。
待完成的官方项目：F2指定100张Test一致性记录；权重再分发许可依据；队伍确认数据使用声明真实性。
模板/规则没有要求全Test报告或额外签署文件，因此不把这些作为新增强制门禁。
当前不能把材料补充或待验包标记为“官方验收全部通过”。

重要：原始预训练权重只用于来源声明，不参与最终前向，不应被500MB模型大小审计误计入。
现有runtime的size_report会保守统计目录全部文件，因此补充材料应放在最终外层材料目录，
不要直接塞进冻结candidate_trt109运行目录。最终整理时保留其独立目录与原manifest。
'''
    (output/'README.md').write_text(text, encoding='utf-8')
    lines = ['# 预训练来源与上游说明', '', '四份文件均来自本地已有缓存，SHA与冻结训练来源记录一致。',
             '采用 torchvision IMAGENET1K_V1，主预训练数据为 ImageNet-1K；无医学预训练。', '',
             '官方模型说明链接：']
    for name in ('densenet121', 'convnext_tiny', 'efficientnet_v2_s', 'swin_t'):
        lines.append(f'- https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.{name}.html')
    lines.extend(['', 'Torchvision公开说明预训练模型可能有独立于软件许可证的数据集条款。',
                  '参考 https://github.com/pytorch/vision#pre-trained-model-license 。',
                  '本材料未将BSD软件许可证冒充全部权重的无条件再分发许可。'])
    (output/'pretrain_model/上游说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    status = {'status': 'MATERIALS_PREPARED_PENDING_T4_ASSET_RETURN', 'team': '王卓颖',
              'full_test_required_by_supplied_rules': False, 'official_sample_count': 100,
              'full_test_optional': True, 'pretrain_files_verified': len(sources),
              'historical_submission_sha256': '8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c',
              'official_consistency': 'NOT_TESTED', 't4_reference_ms_user_reported': 84.46236722171307,
              'runtime_assets_received': False, 'ready_to_submit': False}
    (output/'material_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest_sha256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.config.read_text(encoding='utf-8'))), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
