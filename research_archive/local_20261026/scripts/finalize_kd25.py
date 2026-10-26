"""Independently evaluate frozen KD25-D01 best checkpoint and seal its outputs."""
import argparse
import csv
import io
import json
import platform
import shutil
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cxr.data.archive import archive_path, image_member, read_split_rows
from cxr.data.dataset import make_dataloader, preprocess_image
from cxr.engine.average_precision import average_precision
from cxr.engine.metrics import binary_auc, study_multilabel_metrics
from cxr.engine.validate import StudyTarget
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.utils.checkpoint import load_model_state
from cxr.utils.config import load_config
from cxr.utils.labels import CLASS_NAMES
from cxr.engine.kd25 import digest, read_csv


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(args.config)
    if cfg['experiment']['name']!='KD25_D01':
        ap.error('Use KD25-D01 config')
    run=root/'outputs/experiments/phase26_local_distillation/KD25_D01'
    if json.loads((run/'status.json').read_text(encoding='utf-8'))['state']!='complete':
        raise ValueError('Training not complete')
    if (run/'manifest.json').exists():
        raise FileExistsError('KD25-D01 already sealed')
    if digest(run/'config.yaml')!=digest(args.config):
        raise ValueError('Training config differs')
    shutil.copy2(run/'config.yaml',run/'resolved_config.yaml')
    model=load_model_state(run/'best_model.pt',device='cuda')
    model.eval()
    archive=archive_path(cfg)
    val_rows=read_split_rows(archive,'val')
    loader=make_dataloader(archive,'val',rows=val_rows,batch_size=cfg['inference']['batch_size'],
        num_workers=cfg['data']['num_workers'],unlisted_policy='negative',input_size=384,
        preprocessing='black_pad_v1',persistent_workers=False)
    image_predictions=[]
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast('cuda',dtype=torch.float16):
                p=torch.sigmoid(model(batch['image'].to('cuda',non_blocking=True))).float().cpu().numpy()
            for i,vector in enumerate(p):
                image_predictions.append(ImagePrediction(batch['subject_id'][i],batch['study_id'][i],
                    batch['image_id'][i],tuple(float(x) for x in vector)))
    studies=aggregate_mean_probabilities(image_predictions)
    labels={}
    for row in val_rows:
        k=(row['Subject_id'],row['Study_id'])
        y=tuple(int(x in row['Predict_class'].split(';')) for x in map(str,range(1,11)))
        if k in labels and labels[k]!=y:
            raise ValueError('Val Study labels conflict')
        labels[k]=y
    if len(studies)!=859 or len(labels)!=859 or len({s.subject_id for s in studies})!=228:
        raise ValueError('Val coverage differs')
    targets=[StudyTarget(k[0],k[1],v) for k,v in labels.items()]
    scored=study_multilabel_metrics(image_predictions,targets)
    keys=[(s.subject_id,s.study_id) for s in studies]
    p=np.asarray([s.probabilities for s in studies])
    y=np.asarray([labels[k] for k in keys])
    with (run/'val_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Subject_id','Study_id']+[f'Prob_{i}' for i in range(1,11)])
        for k,vector in zip(keys,p):
            writer.writerow([*k,*[repr(float(x)) for x in vector]])
    per=[]
    for i,record in enumerate(scored['per_class']):
        per.append({**record,'ap':average_precision(y[:,i],p[:,i]),
                    'positive_patients':len({keys[j][0] for j in range(len(keys)) if y[j,i]})})
    with (run/'per_class_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per[0]))
        writer.writeheader(); writer.writerows(per)
    baseline_path=root/'outputs/submissions/E08_EMA_20261020/val_predictions.csv'
    baseline_manifest=json.loads((root/'outputs/submissions/E08_EMA_20261020/manifest.json').read_text(encoding='utf-8'))
    # The exact frozen E08 probability artifact is selected by the existing manifest, never by row order.
    from analyze_phase17_three_fold_ensemble import read_e08
    if not baseline_path.is_file():
        raise FileNotFoundError('Frozen E08 Val probability artifact not found')
    expected=baseline_manifest['files']['val_predictions.csv']['sha256']
    if digest(baseline_path)!=expected:
        raise ValueError('Frozen E08 Val probability SHA mismatch')
    bkeys,bp=read_e08(baseline_path)
    lookup={k:i for i,k in enumerate(bkeys)}
    if len(lookup)!=859 or set(lookup)!=set(keys):
        raise ValueError('E08 IDs differ')
    bp=bp[[lookup[k] for k in keys]]
    baseline_auc=float(np.mean([binary_auc(y[:,i],bp[:,i]) for i in range(10)]))
    if abs(baseline_auc-0.8116686461316818)>1e-9:
        raise ValueError('E08 frozen AUC changed')
    teacher_path=root/'outputs/analysis/phase17_five_fold_final_20261023/e17_val_fivefold.csv'
    handoff=json.loads((root/'outputs/analysis/phase17_final_handoff_20261023/freeze_manifest.json').read_text(encoding='utf-8'))
    teacher_sha=handoff['source_outputs']['e17_val_fivefold.csv']['sha256']
    if digest(teacher_path)!=teacher_sha:
        raise ValueError('Frozen E17 fivefold Val probability SHA mismatch')
    with teacher_path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    teacher_lookup={(r['Subject_id'],r['Study_id']):r for r in rows}
    if len(rows)!=859 or len(teacher_lookup)!=859 or set(teacher_lookup)!=set(keys):
        raise ValueError('E17 fivefold Val IDs differ')
    tp=np.asarray([[float(teacher_lookup[k][f'probability_{i}']) for i in range(1,11)] for k in keys])
    ty=np.asarray([[int(teacher_lookup[k][f'target_{i}']) for i in range(1,11)] for k in keys])
    if not np.array_equal(ty,y) or not np.isfinite(tp).all() or (tp<0).any() or (tp>1).any():
        raise ValueError('E17 fivefold Val labels or probabilities differ')
    teacher_auc=float(np.mean([binary_auc(y[:,i],tp[:,i]) for i in range(10)]))
    if abs(teacher_auc-float(handoff['fivefold_val_auc']))>1e-9:
        raise ValueError('Frozen E17 fivefold Val AUC changed')
    from analyze_phase17_three_fold_ensemble import paired_bootstrap
    paired=paired_bootstrap(p,bp,y,keys,seed=42,valid_samples=2000)
    model_bytes=(run/'best_model.pt').stat().st_size
    shutil.copy2(run/'best_model.pt',run/'best_student.pth')
    if digest(run/'best_student.pth') != digest(run/'best_model.pt'):
        raise ValueError('Student export differs from selected best checkpoint')
    write_json(run/'model_size.json',{'model_file':'best_model.pt','bytes':model_bytes,
        'limit_bytes':500000000,'size_status':'PASS' if model_bytes<=500000000 else 'SIZE_FAIL'})
    # One image and complete read/decode -> pad/normalize -> host/device -> forward -> sigmoid/Study aggregation.
    measured=[]
    with zipfile.ZipFile(archive) as zf,torch.inference_mode():
        samples=[val_rows[i] for i in np.linspace(0,len(val_rows)-1,20,dtype=int)]
        for i in range(45):
            row=samples[i%len(samples)]
            torch.cuda.synchronize()
            start=time.perf_counter()
            with Image.open(io.BytesIO(zf.read(image_member(row,'val')))) as image:
                tensor=preprocess_image(image,size=384,preprocessing='black_pad_v1').unsqueeze(0)
            with torch.autocast('cuda',dtype=torch.float16):
                prob=torch.sigmoid(model(tensor.to('cuda',non_blocking=False))).float().cpu().flatten().tolist()
            aggregate_mean_probabilities([ImagePrediction(row['Subject_id'],row['Study_id'],row['Dicom_id'],tuple(prob))])
            torch.cuda.synchronize()
            if i>=5: measured.append((time.perf_counter()-start)*1000)
    latency={'device':torch.cuda.get_device_name(0),'scope':'ZIP read/decode + RGB resize-pad + normalization + H2D + model forward + sigmoid + CPU transfer + single-image Study mean',
             'images':20,'warmup':5,'measured':len(measured),'precision':'fp16 AMP',
             'mean_ms':statistics.mean(measured),'p50_ms':float(np.percentile(measured,50)),
             'p95_ms':float(np.percentile(measured,95)),'max_ms':max(measured),
             'target_T4_status':'LATENCY_UNVERIFIED'}
    write_json(run/'latency_benchmark.json',latency)
    history=[]
    for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        history.append(json.loads(line))
    with (run/'train_history.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['epoch','train_loss','hard_bce','soft_bce','val_macro_auc','val_macro_f1',
                'best_epoch','best_auc','stale','optimizer_steps','amp_skipped_steps','elapsed_seconds']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in history:
            writer.writerow({'epoch':r['epoch'],'train_loss':r['train_loss'],
                'hard_bce':r['hard_bce'],'soft_bce':r['soft_bce'],
                'val_macro_auc':r['val']['macro_auc'],'val_macro_f1':r['val']['macro_f1'],
                **{name:r.get(name) for name in fields[6:]}})
    best=json.loads((run/'best_metrics.json').read_text(encoding='utf-8'))
    if abs(scored['macro_auc']-best['val']['macro_auc'])>1e-9:
        raise ValueError('Independent best-checkpoint Val AUC differs from training record')
    metrics={'best_epoch':best['epoch'],'val_macro_auc':scored['macro_auc'],
             'val_macro_f1_at_0_5':scored['macro_f1'],
             'val_macro_ap':float(np.mean([r['ap'] for r in per])),
             'per_class':per,'e08_val_macro_auc':baseline_auc,
             'e17_fivefold_val_macro_auc':teacher_auc,
             'delta_vs_e17_fivefold':scored['macro_auc']-teacher_auc,
             'e17_fivefold_val_sha256':teacher_sha,
             'delta_vs_e08':scored['macro_auc']-baseline_auc,
             'paired_patient_bootstrap':paired,'official_val_is_reused_development_data':True,
             'student_training_predictions_are_not_oof':True,
             'gpu_peak_mib':max(r['gpu_peak_allocated_mib'] for r in history),
             'stop_epoch':history[-1]['epoch'],'stop_reason':history[-1]['stop_reason']}
    write_json(run/'metrics.json',metrics)
    print(json.dumps({'best_epoch':best['epoch'],'val_auc':scored['macro_auc'],
          'delta_vs_e08':metrics['delta_vs_e08'],'size_bytes':model_bytes,
          'mean_latency_ms_4060':latency['mean_ms']}),flush=True)
    files=['config.yaml','resolved_config.yaml','teacher_oof_audit.json','run_provenance.json',
           'environment.json','pretrained_source.json','seed.txt','status.json','best_metrics.json',
           'best_model.pt','best_student.pth','last_training.pt','metrics.jsonl','train_history.csv',
           'val_predictions.csv','per_class_metrics.csv','model_size.json','latency_benchmark.json',
           'metrics.json','training.log','training_error.log']
    if not (run/'training.log').is_file():
        raise FileNotFoundError('Complete training.log required before sealing')
    manifest={'experiment_id':'KD25_D01','status':'FROZEN','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              'config_sha256':digest(args.config),'teacher_oof_sha256':json.loads((run/'teacher_oof_audit.json').read_text())['teacher_probability_sha256'],
              'study_count':859,'patient_count':228,'test_auc':None,
              'files':{name:{'bytes':(run/name).stat().st_size,'sha256':digest(run/name)} for name in files}}
    write_json(run/'manifest.json',manifest)

if __name__=='__main__':
    main()
