"""Seal the Val-only fixed substitution audit; does not run inference."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from cxr.inference.e03_h17_validation import sha,save,read_probs,roc_auc_score,csv_write
from cxr.data.archive import read_split_rows,parse_positive_classes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'));out=Path(cfg['output']);project=Path(__file__).resolve().parents[3]
    r=json.loads((out/'report.json').read_text(encoding='utf-8'));source=json.loads(Path(cfg['candidate_manifest']).read_text(encoding='utf-8'))
    audit=json.loads((out/'source_audit.json').read_text(encoding='utf-8'))
    assert sha(out/'executed_validation_source.py.txt')==audit['code_sha256']
    executed=(out/'executed_validation_source.py.txt').read_text(encoding='utf-8')
    corrected=Path(__file__).with_name('e03_h17_validation.py').read_text(encoding='utf-8')
    assert corrected==executed.replace('worst_class=j+1)','worst_class=int(j+1))')
    assert all(sha(p['path'])==p['sha256'] for p in source['weights']+source['source_configs'])
    previous=json.loads((project/'outputs/analysis/e08_fixed_weight_assessment_20261104/audit.json').read_text(encoding='utf-8'))
    for path,digest in previous['source_sha256'].items():assert sha(project/path)==digest
    assert all(sha(p)==audit['reference_hashes'][n] for n,p in cfg['references'].items())
    orig=json.loads((project/'outputs/submissions/E08_EMA_20261020/manifest.json').read_text(encoding='utf-8'))
    for n in ['B01','B02','B03','M02']:
        m=orig['models'][n];assert m['preprocessing']=='black_pad_v1' and m['input_size']==384
        assert sha(cfg['references'][n])==m['val_predictions_sha256']
    with (out/'pass1_image_predictions.csv').open(encoding='utf-8',newline='') as f:im1=list(csv.DictReader(f))
    with (out/'pass2_image_predictions.csv').open(encoding='utf-8',newline='') as f:im2=list(csv.DictReader(f))
    assert len(im1)==len(im2)==1422
    for a,b in zip(im1,im2):
        for k in ['Subject_id','Study_id','Dicom_id','image_sha256']:assert a[k]==b[k]
    raw=read_split_rows(cfg['archive'],'val');truth={(x['Subject_id'],x['Study_id']):[int(j in parse_positive_classes(x['Predict_class'])) for j in range(1,11)] for x in raw}
    keys=sorted(truth);y=np.array([truth[k] for k in keys]);new=read_probs(out/'pass1_E03_H17_val.csv',keys);e08=read_probs(cfg['e08_reference'],keys)
    for n in ['B01','B02','B03','M02']:
        with Path(cfg['references'][n]).open(encoding='utf-8-sig',newline='') as f:
            for rec in csv.DictReader(f):
                assert [int(rec[f'target_{j}']) for j in range(1,11)]==truth[rec['Subject_id'],rec['Study_id']]
    pc=[]
    for j in range(10):
        a=roc_auc_score(y[:,j],new[:,j]);b=roc_auc_score(y[:,j],e08[:,j]);pc.append([j+1,a,b,a-b,int(y[:,j].sum())])
    csv_write(out/'per_class_vs_e08.csv',['class','new_auc','e08_auc','delta','positive_studies'],pc)
    c=r['historical_comparisons'][-1];boot=r['bootstrap_vs_e08'];diagpath=out/'batch_diagnosis.json'
    diag=json.loads(diagpath.read_text(encoding='utf-8')) if diagpath.exists() else None
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('e03_h17_validation.py'),Path(__file__).with_name('e03_h17_batch_diagnosis.py'),Path(__file__).with_name('e03_h17_saved_audit.py'),
        project/'src/cxr/data/dataset.py',project/'src/cxr/data/archive.py',project/'src/cxr/models/factory.py',project/'src/cxr/utils/checkpoint.py',
        project/'tests/test_e03_h17_validation.py']
    for p in files:
        dest=snapshot/p.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    table='\n'.join(f"|{a['name']}|{a['auc']:.10f}|{a['reference_auc']:.10f}|{a['max_abs']:.9g}|{a['mean_abs']:.9g}|{a['p99_abs']:.9g}|{'PASS' if a['strict_1e4_pass'] else 'FAIL'}|" for a in r['historical_comparisons'])
    text=f'''# E03 + H17 固定替换：真实Val复现验收

## 实际完成
原始Val图像1422张，859 Study，228患者；两遍完整CUDA推理。未读取Test、未训练、未生成Submission。
固定公式：0.5 × E03 + 0.5 × H17_BATCH1；E03 = 0.5 × (B01+B02+B03)/3 + 0.5 × M02。
最终入口：`cxr.inference.e03_h17_validation.predict_image`；batch=1，FP16 autocast，原FP32权重，384 black_pad_v1，Study Mean，无随机增强。

## 概率与AUC
真实入口Val Macro-AUC：**{c['auc']:.10f}**；历史缓存固定融合：{c['reference_auc']:.10f}。
相对冻结E08：{boot['delta']:+.10f}；2000次患者级配对Bootstrap 95% CI：[{boot['ci95'][0]:+.10f}, {boot['ci95'][1]:+.10f}]。
Bootstrap固定seed42，按患者重采样，任一类无正/负样本则丢弃整次抽样；有效{boot['valid']}次。
反复使用的官方Val仅属探索性开发证据，不是完整OOF或独立泛化确认。

|组成/融合|新AUC|缓存AUC|最大绝对差|平均绝对差|P99差|历史1e-4门槛|
|---|---:|---:|---:|---:|---:|---|
{table}

最终融合最大差所在：{c['worst_subject']} / {c['worst_study']} / 类别{c['worst_class']}。
全部1422图与冻结H17预处理逐张张量最大差：{r['input_tensor_max_abs']}。
新入口两遍全Val最大差：{max(r['self_repeat_max_abs'].values()):.10g}；事先固定内部门槛1e-6：**{'PASS' if r['self_repeat_pass'] else 'FAIL'}**。
Study Mean与按组成模型聚合后固定融合的交换一致性已检查（误差<1e-14）。类别固定1–10，概率有限且在[0,1]，ID覆盖与原始Val映射一致。

## 来源及差异解释
五个checkpoint和必要来源配置逐字节SHA-256均核验，推理结束后再次核验，原文件未改动。
E03四个模型来自冻结E08 manifest。H17来自冻结Phase31 H17_02_T4_candidate，使用相同纯state_dict与模型构造。
历史E03实际批量为DenseNet 16、ConvNeXt 8（Phase20记录），不能仅依据配置中的推理batch_size=16推断所有模型实际批量。
当前候选统一batch=1；历史比较失败时不得称与缓存数值等价，不放宽1e-4门槛、不人为修改概率。
历史批量诊断：{json.dumps(diag,ensure_ascii=False) if diag else '未执行；没有新增诊断证据。'}

## 决策
{'内部复现与来源/输入校验通过，可作为独立batch=1候选继续讨论；' if r['self_repeat_pass'] else '内部复现失败，阻止后续Test候选；'}历史缓存数值等价状态为 **{'PASS' if c['strict_1e4_pass'] else 'FAIL'}**。
不将历史缓存与新入口混用，不将概率数值变化解读为泛化提升。当前没有Test CSV，也没有READY_TO_SUBMIT标记。
原E08仍保留为官方77.75基准；新候选的Val点估计不能保证官方提分。

## 资源与部署边界
五个实际权重{source['total_checkpoint_bytes']:,}字节；权重加来源配置{source['checkpoint_and_source_config_bytes']:,}字节，低于500,000,000字节。
两次实际推理与输入比较耗时{r['elapsed_seconds']:.1f}秒，不作为官方单图测速。显存峰值未成功持久化，标记UNAVAILABLE。
两遍预测写盘后，报告保存遇到NumPy整数JSON序列化错误；仅将报告类别编号转为Python int后从已保存预测进行CPU复算，未改动任何预测。原执行源码快照与修复后的源码均保留并校验单行差异。
这不是官方测速，未给出100ms结论。用户报告原E08 T4通过保持不变；新替换候选仍T4_UNVERIFIED。
本机目录没有可用Git仓库元数据，不能编造新Git commit；以代码快照、配置及SHA-256固定本次入口。H17历史来源commit见原部署配置。
'''
    (out/'final_decision.md').write_text(text,encoding='utf-8')
    readme=f'''# Fixed E03/H17 Val reproduction

This directory is Val-only. No Test candidate was generated. Do not overwrite completed outputs.
Actual inference: `cxr.inference.e03_h17_validation.load_models` and `predict_image`.
Source snapshot is an audit snapshot, not a complete portable deployment package.

```powershell
$env:PYTHONPATH = "{project / 'src'}"
& "{project / '.venv/Scripts/python.exe'}" -m unittest discover -s "{project / 'tests'}" -p test_e03_h17_validation.py -v
& "{project / '.venv/Scripts/python.exe'}" -m cxr.inference.e03_h17_validation --config "{args.config}"
& "{project / '.venv/Scripts/python.exe'}" -m cxr.inference.e03_h17_batch_diagnosis --config "{args.config}"
& "{project / '.venv/Scripts/python.exe'}" -m cxr.inference.e03_h17_report --config "{args.config}"
```

For a new verification, copy config.json and set a new output directory. Existing reports are not overwritten by inference.
CPU tests: fixed fusion, Study Mean commutativity, weighted/tied AUROC vs explicit pairwise definition and replicated observations, macro aggregation.
No new dependencies were installed. CUDA execution uses the existing PyTorch environment recorded in source_audit.json.
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    save(out/'cpu_tests.json',dict(passed=5,failed=0,test_file_sha256=sha(project/'tests/test_e03_h17_validation.py')))
    save(out/'manifest_sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest_sha256.json','running.lock']})
    print(json.dumps(dict(auc=c['auc'],self_repeat_pass=r['self_repeat_pass'],historical_match=c['strict_1e4_pass'],report=str(out/'final_decision.md')),ensure_ascii=False))

if __name__=='__main__':main()
