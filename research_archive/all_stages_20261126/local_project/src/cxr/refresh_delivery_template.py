"""Refresh only declaration materials in an independently versioned delivery."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from copy import copy
import openpyxl

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path)
    cfg=json.loads(ap.parse_args().config.read_text(encoding='utf-8'))
    source,template,out=map(Path,(cfg['source'],cfg['template'],cfg['output']))
    assert not out.exists() and not out.with_suffix('.zip').exists()
    original=json.loads((source/'manifest_sha256.json').read_text(encoding='utf-8'))
    for n,v in original.items():
        assert sha(source/n)==v['sha256'],n
    shutil.copytree(source,out)
    values=json.loads((source/'materials/declaration_values.json').read_text(encoding='utf-8'))
    weights=json.loads((source/'pretrain_model/sources_sha256.json').read_text(encoding='utf-8'))
    values['A2']='\n'.join(v['source']+'：'+url for url,v in weights.items())
    values['A5']='\n'.join(Path(v['package_path']).name+' SHA-256：'+v['sha256'] for v in weights.values())
    values['A6']='B01/B03/N01_EMA/S17_F3：PyTorch FP16 CUDA Graph；B02/M02：TensorRT 10.9.0.34 FP32。batch=1，384×384。六模型固定融合及Study Mean。非INT8。'
    values['B1']='训练阶段未使用测试集图像（无论是否带标签）及任何外部医学数据集测试集图像（无论是否带标签）。队伍确认六个模型仅以官方Train有监督训练，使用A1–A5公开ImageNet初始化；未使用其他外部医学数据。见materials/team_data_confirmation.json及code/train_records。'
    values['B3']='B01/B02/B03未配置随机图像增强。M02/N01_EMA/S17_F3使用conservative_cxr_v1：以0.5概率施加±3°旋转、水平/垂直±2%平移；以0.5概率施加亮度及对比度系数0.95–1.05。六模型均使用普通BCE；未配置Mixup/CutMix、随机遮挡、过采样/欠采样或类别加权损失。逐项依据code/train_records/*/config.yaml及训练代码。增强仅在训练中使用，推理时关闭全部随机增强算子，随机参数不进入推理。固定RGB、384等比缩放补黑及ImageNet归一化不属于随机增强。'
    values['D4']='否。未使用知识蒸馏，无教师模型参与最终推理。'
    values['D5']='—（未使用知识蒸馏）'
    values['E2']='测试阶段仅固定参数前向推理，无梯度更新、测试时训练或跨样本自适应。模型eval运行；含BN模型使用训练期保存统计量，LN使用冻结参数及其固有逐样本计算，不从Test估计或更新状态。固定随机种子，关闭随机增强。逐图预处理，不跨图汇总统计量。分类CSV输出10类概率；内部二值核验阈值0.5，非推理时调整。无分割/NMS。历史CSV与T4部署入口的官方指定样本判定/排序/概率核验仍待完成。'
    values['E3']='无TTA。固定六模型概率融合：0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3；E03=0.5×mean(B01,B02,B03)+0.5×M02。逐模型多图Study作算术Mean，Study和患者ID仅来自官方元数据。同一患者不同Study之间不平滑；无标签使用、参数拟合或Test总体统计估计。本包为历史78.81版本，不含LR_HIGH替换候选。'
    w=openpyxl.load_workbook(template);old=openpyxl.load_workbook(next(source.glob('*.xlsx')))
    differences=[]
    for row in w.active:
        for cell in row[:4]:
            if cell.value!=old.active[cell.coordinate].value:differences.append(cell.coordinate)
    ids=[]
    for row in w.active.iter_rows(min_row=2):
        key=row[0].value
        if key:
            assert key in values,key
            ids.append(key);cell=w.active.cell(row[0].row,5,values[key])
            alignment=copy(cell.alignment);alignment.wrap_text=True;alignment.vertical='top';cell.alignment=alignment
            w.active.row_dimensions[row[0].row].height=min(300,max(72,18*(len(values[key])//45+values[key].count('\n')+2)))
    assert len(ids)==22 and len(set(ids))==22
    w.active.column_dimensions['E'].width=95
    filename='模型核验材料与数据申明清单.xlsx'
    w.save(out/filename)
    reread=openpyxl.load_workbook(out/filename)
    for row in w.active:
        for c in row[:4]:assert reread.active[c.coordinate].value==c.value
    for row in reread.active.iter_rows(min_row=2):assert row[4].value==values[row[0].value]
    shutil.copy2(template,out/'materials/official_template_received_20261111.xlsx')
    dump(out/'materials/declaration_values.json',values)
    audit={'team':'王卓颖','template_source':str(template),'template_sha256':sha(template),
           'filled_items':len(ids),'template_field_differences_vs_prior':differences,
           'scope':'Historical 78.81 six-model package, NOT LR_HIGH',
           'pending_items':['F2 official sample consistency','Study/Gate clarification','Pretrained weight licence applicability'],
           'frozen_runtime_submission_unchanged':True}
    for n,v in original.items():
        if n.startswith(('model/','code/','pretrain_model/')) or n in ('submission.csv','predict.py','cost.py','requirements.txt'):
            assert sha(out/n)==v['sha256'],n
    dump(out/'materials/template_update_audit.json',audit)
    with (out/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-11-11 官方清单模板更新\n队名：王卓颖。根目录Excel使用本次提供的模板重新填写22项；保留官方A–D列内容。原始模板及核验记录见materials/official_template_received_20261111.xlsx与template_update_audit.json。模型、推理代码和历史78.81 Submission字节未变，未纳入LR_HIGH候选。F2等待官方样本，A3许可适用性及Study/Gate口径仍待明确。\n')
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest_sha256.json'}
    dump(out/'manifest_sha256.json',manifest)
    zp=out.with_suffix('.zip')
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(zp) as z:
        assert len(z.namelist())==len(manifest)+1
        for n,v in manifest.items():
            with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==v['sha256'],n
    result={'path':str(zp),'bytes':zp.stat().st_size,'sha256':sha(zp),'zip_all_files_verified':True,**audit}
    dump(zp.with_suffix('.zip.sha256.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
