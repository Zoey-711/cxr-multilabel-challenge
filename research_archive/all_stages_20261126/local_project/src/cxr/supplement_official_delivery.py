"""Archive public model provenance and incorporate the team's explicit declarations."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import openpyxl


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def run(cfg):
    source,output=Path(cfg['source']),Path(cfg['output'])
    if output.with_suffix('.zip').exists():
        raise FileExistsError(output)
    original=json.loads((source/'manifest_sha256.json').read_text(encoding='utf-8'))
    for n,r in original.items():
        if sha(source/n)!=r['sha256']:
            raise ValueError('Source changed: '+n)
    if output.exists():
        # Resume only an untouched copy left by a public-source download failure.
        for n,r in original.items():
            if not (output/n).is_file() or sha(output/n)!=r['sha256']:
                raise ValueError('Existing output is no longer an untouched source copy: '+n)
    else:
        shutil.copytree(source,output)
    evidence=output/'materials/public_sources'
    evidence.mkdir(exist_ok=True)
    urls={
        'torchvision_v0.23_LICENSE.txt':'https://raw.githubusercontent.com/pytorch/vision/v0.23.0/LICENSE',
        'torchvision_v0.23_README.md':'https://raw.githubusercontent.com/pytorch/vision/v0.23.0/README.md',
        'torchvision_models_0.23.html':'https://docs.pytorch.org/vision/0.23/models.html',
    }
    for name in ('densenet121','convnext_tiny','efficientnet_v2_s','swin_t'):
        urls[name+'_official.html']=f'https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.{name}.html'
    for name in ('densenet','convnext','efficientnet','swin_transformer'):
        urls[name+'_v0.23_source.py']=f'https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/models/{name}.py'

    def download(item):
        name,url=item
        request=urllib.request.Request(url,headers={'User-Agent':'CXR-submission-provenance-audit/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            data=response.read()
            final=response.url
        if not data:
            raise ValueError('Empty source '+url)
        path=evidence/name
        path.write_bytes(data)
        return name,{'url':url,'resolved_url':final,'bytes':len(data),'sha256':sha(path),
                     'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=dict(pool.map(download,urls.items()))
    write_json(evidence/'source_manifest.json',records)
    shutil.copy2(evidence/'torchvision_v0.23_LICENSE.txt',output/'licenses/torchvision_v0.23_LICENSE.txt')
    declared_sources=json.loads((output/'pretrain_model/sources_sha256.json').read_text(encoding='utf-8'))
    names=('densenet','convnext','efficientnet','swin_transformer')
    for url,srcname in zip(declared_sources,names):
        if url not in (evidence/(srcname+'_v0.23_source.py')).read_text(encoding='utf-8'):
            raise ValueError('Public source weight URL mismatch: '+url)
    confirmed={
        'scope':'B01/B02/B03/M02/N01_EMA/S17_F3',
        'source':'用户在本次会话明确回复“以上全部属实”',
        'official_train_only':True,'official_val_used_for_selection_not_training':True,
        'external_medical_data_used':False,'additional_unlabeled_data_used':False,
        'self_supervised_continued_training_used':False,'knowledge_distillation_used':False,
        'test_only_fixed_inference_and_audit':True,
        'official_100_test_and_5_warmup_received':False,
        'official_gate_and_partial_study_clarification_received':False,
        'note':'该记录是用户确认的文字声明，不伪造签名或主办方批准。'}
    write_json(output/'materials/team_data_confirmation.json',confirmed)
    values=json.loads((output/'materials/declaration_values.json').read_text(encoding='utf-8'))
    values.update({
        'A2':'四种torchvision官方公开权重URL见pretrain_model/sources_sha256.json；已用v0.23.0官方源代码逐项核对URL。四份原始文件随包附齐。',
        'A3':'附torchvision v0.23.0 BSD-3-Clause许可证全文及官方README/模型文档快照。软件许可允许再分发；官方另外提示预训练权重可能受数据集衍生条款约束。当前材料未找到独立、明确覆盖四份权重再分发的许可文本，申请主办方核验，未宣称权重许可已无条件通过。见licenses/权重许可证据说明.md。',
        'A4':'四种torchvision IMAGENET1K_V1权重，主预训练数据为ImageNet-1K，非医学预训练。四份官方模型说明HTML、四份固定版本官方模型源码及URL/SHA已附materials/public_sources。',
        'B1':'队伍确认：六个最终推理模型仅使用赛方官方Train进行训练，未使用任何外部医学数据或Test图像训练。对应代码和训练配置见code/train_records，确认记录见materials/team_data_confirmation.json。',
        'B2':'队伍确认：官方Val未并入六个模型训练；仅用于早停、checkpoint选择及候选分析。历史Val被多次使用，开发结论属于探索性。',
        'C1':'队伍确认：六个模型未进行自监督预训练/续训；从已申报的公开ImageNet权重进行官方Train有监督训练。',
        'C2':'队伍确认：除已申报ImageNet初始化权重外，未额外使用外部无标签数据训练、模型选择或统计拟合。',
        'D1':'队伍确认：六个最终推理模型未使用知识蒸馏。其他历史蒸馏实验不属于本次最终模型。',
        'E1':'队伍确认：Test仅用于固定推理与提交脚本核验；未用于预训练、微调、自监督、伪标签、归一化统计量、类别先验、阈值/后处理拟合或checkpoint选择。',
        'F2':'主办方指定100张Test及5张预热图尚未发放（用户确认）；该项待主办方发放样本后使用cost.py核验，不虚填0不一致或PASS。cost.py与predict.py共用冻结入口。当前0.5为内部Gate约定；若样本含不完整多图Study，输出UNVERIFIED_PARTIAL_STUDY。请主办方确认Study Mean映射与Gate口径。全Test自检不是官方强制提交项。',
    })
    write_json(output/'materials/declaration_values.json',values)
    w=openpyxl.load_workbook(output/'模型核验材料与数据申明清单.xlsx')
    for row in w.active.iter_rows(min_row=2):
        if row[0].value in values:
            w.active.cell(row[0].row,5,values[row[0].value])
    w.save(output/'模型核验材料与数据申明清单.xlsx')
    (output/'licenses/权重许可证据说明.md').write_text('''# 权重许可证据与适用边界

已附固定版本torchvision v0.23.0软件许可证全文（BSD-3-Clause），官方README、模型说明及模型源码。
四份权重可从download.pytorch.org公开获取，实际原始文件SHA均与训练来源记录一致。
四份官方源码中的下载URL与随包记录逐一匹配；模型为ImageNet-1K初始化。

软件许可证不自动等同于全部预训练权重的单独授权。torchvision官方模型页面明确提示：
预训练模型可能受其训练数据集衍生的独立许可或条件约束，用户需自行确定适用许可。
来源：https://docs.pytorch.org/vision/0.23/models.html （General information on pre-trained weights）
原文及检索时文件哈希在materials/public_sources。

未找到四份权重独立的、明确满足赛事再分发条款的完整许可文本，因此A3提交现有真实证据，
请求主办方确认torchvision官方ImageNet权重是否按本次赛事第1条接受。
本包不伪造授权文件，不把BSD软件许可证或原论文许可证冒充权重授权。
''',encoding='utf-8')
    (output/'materials/主办方待确认事项.md').write_text('''# 主办方待确认事项（草稿，尚未发送）

队名：王卓颖。最终六模型使用torchvision官方ImageNet-1K预训练权重，固定融合及Study Mean。

1. 官方100张Test与5张预热图尚未收到，请告知发放时间/获取方式。F2待提供样本后核验。
2. CSV为Study级，若指定100图仅含某Study部分影像，如何得到与CSV对应的完整Study Mean？
   是否按完整Study提供样本，或提供其他明确核验流程？当前不缓存缺失图像结果、不伪报一致。
3. Pred_Label/Gate的具体阈值、No Finding等类别的判定规则是什么？现有0.5仅内部审计约定。
4. 四份torchvision官方ImageNet权重的来源、原始SHA、公开下载链接、BSD软件许可全文及官方
   权重许可提醒已附。请确认此类权重是否满足第1条赛事使用/再分发要求，或需补何种证明。

我们已有T4 Val参考测速均值84.46ms；尚未把它称为官方100张Test验收结果。
''',encoding='utf-8')
    (output/'materials/规则要求完成情况.md').write_text('''# 官方要求完成情况

- 已完成：冻结六模型/历史78.81 CSV、推理入口与cost共用逻辑、文件SHA及大小、原始预训练文件、
  官方模型说明/源码/软件许可归档、更新Excel、队伍真实数据使用声明、T4 Val参考报告及300记录。
- 尚未具备外部条件：主办方100张Test和5张预热图尚未发放，因此F2待样本发放后核验。
- 尚待明确：多图Study/Gate核验口径；预训练权重再分发许可适用性。
- 不属于强制新增项：全部2218张Test自检、要求用户另签未规定的授权书。
- 本次没有更改任何权重、引擎、预测函数、融合参数或Submission。

状态为MATERIALS_SUPPLEMENTED_PENDING_ORGANIZER_ITEMS，不写ALL_PASS或READY_TO_SUBMIT。
''',encoding='utf-8')
    readme=(output/'README.md').read_text(encoding='utf-8')
    readme=readme.replace('PACKAGED_WITH_DECLARED_OPEN_ITEMS','MATERIALS_SUPPLEMENTED_PENDING_ORGANIZER_ITEMS')
    readme=readme.replace('尚未完成：F2官方100张Test的判定/排序/概率误差核验，A3预训练权重再分发许可依据，队伍确认数据声明。',
       '队伍数据使用声明已明确确认，官方模型说明/源码/软件许可已归档。官方100张Test和5张预热图尚未发放，F2待发放后核验；权重再分发许可适用性仍待主办方确认。')
    readme+='\n本次补充材料及待确认事项见materials/规则要求完成情况.md与materials/主办方待确认事项.md。\n'
    (output/'README.md').write_text(readme,encoding='utf-8')
    audit=json.loads((output/'materials/assembly_audit.json').read_text(encoding='utf-8'))
    audit['package_status']='MATERIALS_SUPPLEMENTED_PENDING_ORGANIZER_ITEMS'
    audit['remaining_official_items']=['F2: awaiting organizer 100 Test / 5 warmup images','Study/Gate interface clarification','A3: pretrained-weight redistribution applicability']
    audit['team_data_declaration_confirmed']=True
    audit['public_source_files_archived']=len(records)
    write_json(output/'materials/assembly_audit.json',audit)
    for n,r in original.items():
        if n.startswith('model/') or n=='submission.csv':
            if sha(output/n)!=r['sha256']:raise ValueError('Frozen file changed: '+n)
    manifest={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
              for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'manifest_sha256.json'}
    write_json(output/'manifest_sha256.json',manifest)
    zp=output.with_suffix('.zip')
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    with zipfile.ZipFile(zp) as z:
        if len(z.namelist())!=len(manifest)+1:raise ValueError('Zip inventory differs')
        for n,r in manifest.items():
            with z.open(n) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=r['sha256']:raise ValueError(n)
    summary={'path':str(zp),'bytes':zp.stat().st_size,'sha256':sha(zp),'files':len(manifest)+1,
             'status':audit['package_status'],'frozen_runtime_and_submission_unchanged':True,
             'team_data_declaration_confirmed':True,'official_sample_available':False,
             'weight_license_unconditionally_verified':False,'source_files_archived':len(records)}
    write_json(zp.with_suffix('.zip.sha256.json'),summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
