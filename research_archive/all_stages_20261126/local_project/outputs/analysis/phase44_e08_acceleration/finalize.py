"""Produce reproducible Phase44 reports from measured files; no inference."""
import csv
import hashlib
import json
import re
import pandas as pd
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))

def main():
    old_text=(ROOT/'outputs/analysis/phase30_deployment/candidate_deployment/E08_EMA_cost_report.json').read_text(encoding='utf-8')
    # Legacy report has a malformed Windows path near its end; numeric header is intact.
    old={key:float(re.search(r'"'+key+r'"\s*:\s*([0-9.]+)',old_text).group(1)) for key in ('mean_ms','p50_ms','p95_ms')}
    fp16=read('onnx_cost_report.json')
    fp32=read('onnx_fp32_cost_report.json')
    audits={p:read(f'onnx_{p}_val_audit.json') for p in ('fp16','fp32')}
    exported=read('onnx_export_manifest.json')
    sources=json.loads((ROOT/'outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json').read_text())
    configs=[ROOT/'outputs/experiments'/meta['run_id']/'config.yaml' for meta in sources['models'].values()]
    required=[HERE/'predict.py',HERE/'cost.py',HERE/'requirements_inference.txt',
              ROOT/'src/cxr/data/archive.py',ROOT/'src/cxr/data/dataset.py',
              ROOT/'src/cxr/data/sampling.py',ROOT/'src/cxr/engine/validate.py',
              ROOT/'outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json',*configs]
    for path in required:
        if not path.exists():raise FileNotFoundError(path)
    config_and_code=sum(path.stat().st_size for path in required)
    models={p:sum(exported[n][f'{p}_bytes'] for n in exported) for p in ('fp16','fp32')}
    size={p:dict(graph_bytes=models[p],necessary_config_and_code_bytes=config_and_code,
                 deployment_total_bytes=models[p]+config_and_code,limit_bytes=500000000,
                 under_500mb=models[p]+config_and_code<=500000000,
                 graphs={n:exported[n][f'{p}_sha256'] for n in exported}) for p in ('fp16','fp32')}
    size['frozen_pytorch']={'weight_plus_config_bytes':278442617,'source':'Phase30 frozen model_size_and_hashes.json'}
    size['counting_rule']='One selected graph format per each of five models; do not sum both FP16 and FP32 alternatives. Includes inference Python, protocol manifest and required source/config files; excludes libraries and non-loaded experiment files.'
    (HERE/'model_size_report.json').write_text(json.dumps(size,ensure_ascii=False,indent=2),encoding='utf-8')
    per_class=[]
    for precision in ('fp16','fp32'):
        frame=pd.read_csv(HERE/f'probability_consistency_{precision}.csv')
        frame.insert(0,'candidate',f'ONNX {precision.upper()}')
        per_class.append(frame)
    pd.concat(per_class,ignore_index=True).to_csv(HERE/'probability_consistency.csv',index=False)
    rows=[]
    for name,report,audit,bytes_ in [('PyTorch historical',old,None,278442617),('ONNX FP16',fp16,audits['fp16'],size['fp16']['deployment_total_bytes']),('ONNX FP32',fp32,audits['fp32'],size['fp32']['deployment_total_bytes'])]:
        rows.append(dict(candidate=name,device='RTX 4060 Laptop',mean_ms=report['mean_ms'],p50_ms=report['p50_ms'],
                         p95_ms=report['p95_ms'],difference_from_historical_ms=report['mean_ms']-old['mean_ms'],
                         measured_images=100,warmup=5,rounds=3,batch=1,model_and_required_file_bytes=bytes_,
                         val_auc=audit['onnx_macro_auc'] if audit else 0.8116686461,
                         probability_status=audit['tolerance_1e_4_status'] if audit else 'FROZEN_REFERENCE',
                         target_hardware_status='T4_UNVERIFIED',
                         comparison_limit='Same fixed Val images and timing scope, separate runs; no paired timing attribution'))
    rows.append(dict(candidate='TensorRT FP16',target_hardware_status='BLOCKED_NO_TENSORRT_RUNTIME',comparison_limit='No trtexec or TensorRT Python runtime; no engine built or measured'))
    with (HERE/'benchmark_comparison.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    historical=ROOT/'outputs/analysis/phase30_deployment/latency_breakdown.csv'
    with (HERE/'latency_breakdown.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file);writer.writerow(['candidate','stage','mean_ms','p50_ms','p95_ms','scope'])
        with historical.open(encoding='utf-8-sig',newline='') as src:
            for row in csv.DictReader(src):
                if row['candidate']=='E08_EMA' and row['mode']=='single_rgb':
                    writer.writerow(['PyTorch Phase30',row['stage'],row['mean_ms'],row['p50_ms'],row['p95_ms'],
                                     'historical instrumented run; not same as official cost mean'])
        with (HERE/'latency_breakdown_fp16.csv').open(encoding='utf-8-sig',newline='') as src:
            for row in csv.DictReader(src):writer.writerow(['ONNX FP16',row['stage'],row['mean_ms'],row['p50_ms'],row['p95_ms'],row['scope']])
        writer.writerow(['ONNX FP16','H2D_transfer',None,None,None,'ORT session.run copies CPU input for each model; included in forward stages, cannot isolate without changing I/O path'])
        writer.writerow(['ONNX FP16','Study_mean_aggregation',None,None,None,'Single image timing sample; full 859 Study inference included aggregation but did not separately time it'])
    doc=f'''# Phase44部署候选\n\n原E08 PyTorch FP16在本机4060历史完整流程{old['mean_ms']:.2f}ms。新ONNX FP16 {fp16['mean_ms']:.2f}ms、FP32 {fp32['mean_ms']:.2f}ms，同一固定105张Val图片、5张预热、100张计时×3轮，batch1；运行不是严格时间配对。模型加载及预热不计时，读图至CSV写出计时。\n\nONNX FP16相对冻结Val概率最大/均值/P99差{audits['fp16']['max_abs_diff']:.6f}/{audits['fp16']['mean_abs_diff']:.6f}/{audits['fp16']['p99_abs_diff']:.6f}；859 Study Macro-AUC={audits['fp16']['onnx_macro_auc']:.6f}，配对患者Bootstrap 95% CI={audits['fp16']['bootstrap_ci95']}。FP32最大差{audits['fp32']['max_abs_diff']:.6f}，Macro-AUC={audits['fp32']['onnx_macro_auc']:.6f}。两者都超过预设1e-4概率一致性门槛，状态REJECTED_CONSISTENCY；不能当作冻结E08的等价加速版本，不能生成新Test Submission。\n\nFP16、FP32实际单格式图及所需小文件分别{size['fp16']['deployment_total_bytes']}和{size['fp32']['deployment_total_bytes']}字节，均低于500000000字节；独立导出目录同时存两种格式，不是正式部署包。TensorRT未安装、未构建，延迟BLOCKED，无数值可填。4060结果不是T4合规证明；官方固定100张图像和T4验收仍未完成。\n'''
    (HERE/'deployment_candidates.md').write_text(doc,encoding='utf-8')
    (HERE/'final_decision.md').write_text(doc+'\n结论：当前仍采用原冻结E08作为准确性基准；Phase44 ONNX版仅用于研究。ONNX FP16在本机约节省'+f"{old['mean_ms']-fp16['mean_ms']:.2f}"+'ms，但概率偏差明显且本地均值仍高于100ms。若未来获得T4环境，可对经重新验收的候选进行目标设备测量；TensorRT须在目标兼容环境构建且先做完整Val一致性。\n',encoding='utf-8')
    manifest={}
    relevant=[p for p in HERE.iterdir() if p.is_file() and p.name not in ('manifest_sha256.json','finalize.py')]
    relevant.extend((HERE/'onnx_models').glob('*.onnx'))
    for path in sorted(relevant):manifest[path.relative_to(HERE).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
    (HERE/'manifest_sha256.json').write_text(json.dumps(dict(files=manifest,source_checkpoint_sha256={n:x['source_sha256'] for n,x in exported.items()}),indent=2),encoding='utf-8')
    print(json.dumps(dict(fp16_ms=fp16['mean_ms'],fp32_ms=fp32['mean_ms'],fp16_auc=audits['fp16']['onnx_macro_auc'],fp16_size=size['fp16']['deployment_total_bytes'],files=len(manifest))))

if __name__=='__main__':main()
