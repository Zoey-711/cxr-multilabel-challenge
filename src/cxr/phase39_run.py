"""Isolated, resumable Phase39 paired Fold1 experiment."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.data.patient_folds import select_fold_rows
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.engine.metrics import binary_auc, study_multilabel_metrics
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.phase39_model import build_model, MultiStageEfficientNet
from cxr.utils.training_resume import capture_rng_state, restore_rng_state
from cxr.phase39_support import digest, write_json, write_csv, save_torch, seed_all, latency, lock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/phase39_multistage_fusion'
CONFIGS = ROOT / 'configs/phase39'
TASKS = ('P39_BASE_F1', 'P39_FUSION_F1')
FOLDS = ROOT / 'outputs/analysis/phase17_patient_folds'
PRETRAINED_ROOT = Path(os.environ.get('CXR_PRETRAINED_ROOT', ROOT / '.wheels/torch_cache'))
WEIGHT = PRETRAINED_ROOT / 'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth'
SOURCE = [ROOT / 'src/cxr/phase39_model.py', ROOT / 'src/cxr/phase39_run.py',
          ROOT / 'src/cxr/phase39_support.py',
          ROOT / 'src/cxr/data/dataset.py', ROOT / 'src/cxr/data/patient_folds.py',
          ROOT / 'src/cxr/engine/metrics.py', ROOT / 'src/cxr/utils/training_resume.py']


def config(task):
    path = CONFIGS / ('base_config.yaml' if task == TASKS[0] else 'fusion_config.yaml')
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert cfg['experiment_id'] == task and cfg['fold'] == 1 and cfg['input_size'] == 384
    assert cfg['micro_batch'] * cfg['accumulation'] == 16
    assert cfg['epochs'] <= 30 and cfg['patience'] == 6 and cfg['ema_decay'] == .999
    assert cfg['selection_metric'] == 'fold1_study_macro_auc'
    return cfg


def audit():
    configs = [config(t) for t in TASKS]
    shared = set(configs[0]) - {'experiment_id', 'architecture'}
    assert all(configs[0][key] == configs[1][key] for key in shared)
    if not WEIGHT.is_file() or digest(WEIGHT) != configs[0]['pretrained_sha256']:
        raise ValueError('ImageNet weight missing or SHA-256 mismatch')
    for name, expected in configs[0]['fold_sha256'].items():
        if digest(FOLDS / name) != expected:
            raise ValueError(f'Frozen fold file mismatch: {name}')
    archive_setting = os.environ.get('CXR_PHASE39_ARCHIVE')
    if not archive_setting:
        raise EnvironmentError('Set CXR_PHASE39_ARCHIVE to the authorized competition ZIP')
    archive = Path(archive_setting)
    if not archive.is_file():
        raise FileNotFoundError(f'Competition archive missing: {archive}')
    all_rows = read_split_rows(archive, 'train')
    fold = dict(kind='patient_fold_v1', directory=str(FOLDS), fold=1,
                sha256=configs[0]['fold_sha256'], expected_counts=dict(
                    train=dict(patients=8844, studies=14826, images=24332),
                    oof=dict(patients=2211, studies=3687, images=6012)))
    train = select_fold_rows(all_rows, fold, role='train')
    oof = select_fold_rows(all_rows, fold, role='oof')
    train_ids = {r['Subject_id'] for r in train}
    oof_ids = {r['Subject_id'] for r in oof}
    if train_ids & oof_ids or len(train_ids) != 8844 or len(oof_ids) != 2211:
        raise ValueError('Patient isolation/count failure')
    if len({(r['Subject_id'],r['Study_id']) for r in oof}) != 3687:
        raise ValueError('OOF Study count failure')
    record = dict(state='PASS', archive=str(archive), archive_bytes=archive.stat().st_size,
                  train_patients=len(train_ids), oof_patients=len(oof_ids), patient_overlap=0,
                  train_images=len(train), oof_images=len(oof), fold_sha256=configs[0]['fold_sha256'],
                  pretrained_sha256=digest(WEIGHT), code_sha256={str(p.relative_to(ROOT)): digest(p) for p in SOURCE},
                  configs={task:digest(CONFIGS/('base_config.yaml' if task==TASKS[0] else 'fusion_config.yaml')) for task in TASKS},
                  official_val_accessed=False, test_accessed=False)
    path = OUT / 'protocol_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != record:
        raise ValueError('Protocol changed after sealing')
    write_json(path, record)
    return archive, train, oof, record


def loaders(cfg, archive, train_rows, oof_rows):
    aug = dict(enabled=True, name='conservative_cxr_v1', affine_probability=.5,
               rotation_degrees=3., translation_fraction=.02, photometric_probability=.5,
               brightness_contrast_fraction=.05)
    # Windows named pipes may be denied in the local sandbox. Keep both paired
    # arms on identical single-process loading and preserve DataLoader RNG.
    shared=dict(batch_size=cfg['micro_batch'], num_workers=0,
                preprocessing='black_pad_v1', input_size=384, persistent_workers=False)
    tr = make_dataloader(archive,'train', rows=train_rows, shuffle=True, seed=42,
                         augmentation=aug, **shared)
    ev = make_dataloader(archive,'train', rows=oof_rows, shuffle=False, seed=43, **shared)
    return tr, ev


def build(cfg, *, pretrained=True):
    seed_all(cfg['seed'])
    os.environ['TORCH_HOME'] = str(PRETRAINED_ROOT)
    return build_model(cfg['experiment_id'], pretrained=pretrained)


def optimizer_scheduler(cfg, model, steps):
    if isinstance(model, MultiStageEfficientNet):
        head = list(model.projection.parameters()) + list(model.classifier.parameters())
    else:
        head = list(model.classifier.parameters())
    ids={id(p) for p in head}
    back=[p for p in model.parameters() if id(p) not in ids]
    assert len(head)+len(back)==len(list(model.parameters()))
    opt=torch.optim.AdamW([dict(params=back,lr=cfg['backbone_lr']),
                           dict(params=head,lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    warm=cfg['warmup_epochs']*steps
    total=cfg['epochs']*steps
    def factor(step):
        if step<warm: return (step+1)/max(1,warm)
        return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,factor)


def step_group(cfg,model,ema,micro,opt,sch,scaler,device):
    denominator=sum(int(b['valid_mask'].sum()) for b in micro)
    if denominator<=0: raise ValueError('All labels missing')
    opt.zero_grad(set_to_none=True)
    loss_sum=0.
    for b in micro:
        x=b['image'].to(device,non_blocking=True)
        y=b['target'].to(device,non_blocking=True)
        mask=b['valid_mask'].to(device,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(x)
            loss=(F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')*mask).sum()/denominator
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite train loss')
        scaler.scale(loss).backward()
        loss_sum+=float(loss.detach())
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradients')
    before=scaler.get_scale();scaler.step(opt);scaler.update()
    skipped=scaler.get_scale()<before
    if not skipped:
        sch.step();ema.update_parameters(model)
    return loss_sum,skipped


def infer(model,loader,targets,output,device):
    model.eval();pred=[]
    with torch.inference_mode():
        for b in loader:
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                prob=torch.sigmoid(model(b['image'].to(device,non_blocking=True))).float().cpu().numpy()
            for i,p in enumerate(prob):
                pred.append(ImagePrediction(b['subject_id'][i],b['study_id'][i],b['image_id'][i],tuple(map(float,p))))
    metrics=study_multilabel_metrics(pred,targets)
    study=aggregate_mean_probabilities(pred)
    if output is not None:
        rows=[dict(Subject_id=s.subject_id,Study_id=s.study_id,
                   **{f'Prob_{j+1}':v for j,v in enumerate(s.probabilities)}) for s in study]
        write_csv(output,rows)
    return metrics


def cpu_tests():
    torch.set_num_threads(2)
    cfg=config(TASKS[1]);seed_all(42)
    x=torch.randn(2,3,64,64)
    base=build_model(TASKS[0],pretrained=False)
    fusion=build_model(TASKS[1],pretrained=False)
    fusion.eval();y=fusion(x)
    if y.shape!=(2,10):raise ValueError('Fusion shape')
    loss=F.binary_cross_entropy_with_logits(y,torch.rand_like(y));loss.backward()
    for name,weight in [('projection',fusion.projection.weight),('classifier',fusion.classifier.weight),
                        ('backbone',fusion.backbone.features[0][0].weight)]:
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all()) or not bool(weight.grad.abs().sum()>0):
            raise ValueError('Missing gradient: '+name)
    ema=AveragedModel(fusion,multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
    ema.update_parameters(fusion)
    path=OUT/'cpu_roundtrip.pt';save_torch(path,fusion.state_dict())
    restored=build_model(TASKS[1],pretrained=False).eval()
    restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    with torch.no_grad(): error=float((fusion(x)-restored(x)).abs().max())
    path.unlink()
    if error!=0: raise ValueError('CPU restore differs')
    report=dict(state='PASS',base_parameters=sum(p.numel() for p in base.parameters()),
                fusion_parameters=sum(p.numel() for p in fusion.parameters()),
                mid_channels=fusion.mid_channels,final_channels=fusion.final_channels,
                output_shape=list(y.shape),projection_gradient=True,backbone_gradient=True,
                ema_update=True,restore_max_abs_error=error)
    write_json(OUT/'cpu_smoke_report.json',report)
    return report


def cuda_smoke(archive,train_rows,oof_rows):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    report=dict(state='RUNNING',gpu=torch.cuda.get_device_name(),models={})
    for task in TASKS:
        cfg=config(task); tr,ev=loaders(cfg,archive,train_rows,oof_rows)
        model=build(cfg).cuda()
        ema=AveragedModel(model,device='cuda',multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
        opt,sch=optimizer_scheduler(cfg,model,math.ceil(len(tr)/cfg['accumulation']))
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        micro=list(itertools.islice(iter(tr),cfg['accumulation']))
        torch.cuda.reset_peak_memory_stats();model.train()
        loss,skip=step_group(cfg,model,ema,micro,opt,sch,scaler,torch.device('cuda'))
        if skip:raise FloatingPointError(task+' smoke AMP skip')
        state=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                   scheduler=sch.state_dict(),scaler=scaler.state_dict(),**capture_rng_state(tr,ev))
        p=OUT/'preflight'/task;p.mkdir(parents=True,exist_ok=True)
        save_torch(p/'resume.pt',state)
        recovered=torch.load(p/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(recovered['model']);ema.load_state_dict(recovered['ema'])
        opt.load_state_dict(recovered['optimizer']);sch.load_state_dict(recovered['scheduler'])
        scaler.load_state_dict(recovered['scaler']);restore_rng_state(recovered,tr,ev)
        sample=next(iter(ev))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            pred=ema(sample['image'].to('cuda'))
        if not bool(torch.isfinite(pred).all()):raise FloatingPointError('Nonfinite eval')
        report['models'][task]=dict(loss=loss,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                                    ema_updates=int(ema.n_averaged),resume='PASS',eval_shape=list(pred.shape))
        del model,ema,opt,sch,scaler,tr,ev,micro,state,recovered
        torch.cuda.empty_cache()
    report['state']='PASS';report['protocol_sha256']=digest(OUT/'protocol_manifest.json')
    write_json(OUT/'cuda_smoke_report.json',report)
    return report


def run_task(task,archive,train_rows,oof_rows):
    cfg=config(task);run=OUT/task;run.mkdir(exist_ok=True)
    if (run/'DONE.json').exists():
        done=json.loads((run/'DONE.json').read_text())
        if all(digest(run/name)==sha for name,sha in done['sha256'].items()):return
        raise ValueError('Frozen task artifact hash mismatch')
    write_json(run/'resolved_config.json',cfg)
    tr,ev=loaders(cfg,archive,train_rows,oof_rows)
    targets=study_targets_from_rows(oof_rows)
    model=build(cfg).cuda()
    ema=AveragedModel(model,device='cuda',multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
    steps=math.ceil(len(tr)/cfg['accumulation'])
    opt,sch=optimizer_scheduler(cfg,model,steps)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    tracker=EarlyStopTracker(patience=cfg['patience'])
    epoch0=1;history=[]
    resume=run/'last_training.pt';protocol_hash=digest(OUT/'protocol_manifest.json')
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['protocol_sha256']!=protocol_hash or state['task']!=task:raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);restore_rng_state(state,tr,ev)
        tracker=EarlyStopTracker(**state['tracker'])
        history=state['history'];epoch0=state['epoch']+1
    with (run/'training.log').open('a',encoding='utf-8') as log:
        def emit(s):
            print(task,s,flush=True);log.write(s+'\n');log.flush()
        emit(f'START epoch={epoch0} steps={steps} micro={cfg["micro_batch"]} accum={cfg["accumulation"]}')
        for epoch in range(epoch0,cfg['epochs']+1):
            model.train();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            total=0.;skips=0;updates=0;iterator=iter(tr)
            for step in range(1,steps+1):
                micro=list(itertools.islice(iterator,cfg['accumulation']))
                value,skip=step_group(cfg,model,ema,micro,opt,sch,scaler,torch.device('cuda'))
                total+=value;skips+=skip;updates+=not skip
                if skips>=20:raise FloatingPointError('Excessive AMP skips')
                if step==1 or step%100==0 or step==steps:
                    write_json(run/'status.json',dict(state='RUNNING',task=task,epoch=epoch,step=step,
                        steps=steps,train_loss=total/step,best_epoch=tracker.best_epoch,best_auc=tracker.best_auc))
            auc=infer(ema,ev,targets,None,torch.device('cuda'))['macro_auc']
            decision=tracker.observe(auc,epoch)
            record=dict(epoch=epoch,train_loss=total/steps,fold1_ema_macro_auc=auc,
                        best_epoch=tracker.best_epoch,best_auc=tracker.best_auc,early_stopping_stale=tracker.stale,
                        optimizer_updates=updates,ema_updates=int(ema.n_averaged),amp_skips=skips,
                        lr_backbone=opt.param_groups[0]['lr'],lr_head=opt.param_groups[1]['lr'],
                        seconds=time.perf_counter()-start,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            history.append(record)
            if decision.new_best_checkpoint:
                save_torch(run/'ema_best.pth',ema.module.state_dict())
                save_torch(run/'raw_at_ema_best.pth',model.state_dict())
            snapshot=dict(task=task,protocol_sha256=protocol_hash,epoch=epoch,
                          model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                          scheduler=sch.state_dict(),scaler=scaler.state_dict(),history=history,
                          tracker=dict(patience=tracker.patience,min_delta=tracker.min_delta,
                                       best_auc=tracker.best_auc,best_epoch=tracker.best_epoch,
                                       patience_reference_auc=tracker.patience_reference_auc,
                                       stale=tracker.stale,last_epoch=tracker.last_epoch),
                          **capture_rng_state(tr,ev))
            save_torch(resume,snapshot)
            write_csv(run/'train_history.csv',history)
            emit(f'EPOCH_DONE {json.dumps(record)}')
            if decision.should_stop:
                emit(f'EARLY_STOP epoch={epoch} best={tracker.best_epoch}')
                break
        if not (run/'ema_best.pth').exists():raise ValueError('No best EMA checkpoint')
        best=build_model(task,pretrained=False).cuda()
        best.load_state_dict(torch.load(run/'ema_best.pth',map_location='cpu',weights_only=True))
        metrics=infer(best,ev,targets,run/'oof_predictions.csv',torch.device('cuda'))
        write_csv(run/'per_class_metrics.csv',metrics['per_class'])
        metrics.update(task=task,best_epoch=tracker.best_epoch,stop_epoch=tracker.last_epoch,
                       ema=True,oof_scope='PARTIAL_OOF_FOLD1',patients=2211,
                       gpu_peak_allocated_mib=max(r['peak_allocated_mib'] for r in history),
                       training_seconds=sum(r['seconds'] for r in history))
        write_json(run/'metrics.json',metrics)
        # Includes actual inference graph and required config; no resume optimizer state.
        size=(run/'ema_best.pth').stat().st_size+(run/'resolved_config.json').stat().st_size
        write_json(run/'model_size.json',dict(weights_and_config_bytes=size,size_pass=size<=500000000,
                                             latency_status='T4_UNVERIFIED'))
        latency(best,ev.dataset,run,cfg)
        write_json(run/'run_manifest.json',dict(task=task,protocol_sha256=protocol_hash,
            fold_sha256=cfg['fold_sha256'],pretrained_sha256=cfg['pretrained_sha256'],
            best_epoch=tracker.best_epoch,stop_epoch=tracker.last_epoch,
            official_val_accessed=False,test_accessed=False,model_family='efficientnet_v2_s'))
        write_json(run/'status.json',dict(state='DONE',task=task,best_epoch=tracker.best_epoch,
                                        fold1_macro_auc=metrics['macro_auc']))
        hashes={str(p.relative_to(run)).replace('\\','/'):digest(p) for p in run.rglob('*')
                if p.is_file() and p.name not in ('DONE.json','latency_last_prediction.csv')}
        write_json(run/'DONE.json',dict(state='DONE',sha256=hashes))
    del model,ema,best,opt,sch,scaler,tr,ev
    torch.cuda.empty_cache()


def analyze(oof_rows):
    if not all((OUT/t/'DONE.json').is_file() for t in TASKS):return
    truth=study_targets_from_rows(oof_rows)
    keys=[(t.subject_id,t.study_id) for t in truth]
    y=np.asarray([t.labels for t in truth],dtype=np.uint8)
    arrays=[]
    for task in TASKS:
        with (OUT/task/'oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        table={(r['Subject_id'],r['Study_id']):[float(r[f'Prob_{i}']) for i in range(1,11)] for r in rows}
        if len(table)!=len(rows) or set(table)!=set(keys):raise ValueError('OOF ID mismatch')
        arrays.append(np.asarray([table[k] for k in keys]))
    auc=np.asarray([[binary_auc(y[:,i],p[:,i]) for i in range(10)] for p in arrays])
    write_csv(OUT/'per_class_auc.csv',[dict(class_id=i+1,positive_studies=int(y[:,i].sum()),
        base_auc=auc[0,i],fusion_auc=auc[1,i],delta=auc[1,i]-auc[0,i]) for i in range(10)])
    patient=np.asarray([k[0] for k in keys]);groups={p:np.where(patient==p)[0] for p in np.unique(patient)}
    names=list(groups);rng=np.random.default_rng(3901);draws=[];invalid=0
    for _ in range(2000):
        sample=rng.choice(names,size=len(names),replace=True)
        ix=np.concatenate([groups[n] for n in sample])
        try:
            diff=np.mean([binary_auc(y[ix,i],arrays[1][ix,i])-binary_auc(y[ix,i],arrays[0][ix,i]) for i in range(10)])
            draws.append(float(diff))
        except ValueError:invalid+=1
    if len(draws)<1900:raise ValueError('Too many invalid bootstrap draws')
    delta=float(np.mean(auc[1]-auc[0]));ci=np.percentile(draws,[2.5,97.5])
    write_json(OUT/'bootstrap_comparison.json',dict(seed=3901,draws=2000,valid=len(draws),invalid=invalid,
        unit='patient',macro_delta=delta,ci95=list(map(float,ci)),
        base_auc=float(np.mean(auc[0])),fusion_auc=float(np.mean(auc[1])),
        prediction_pearson=float(np.corrcoef(arrays[0].ravel(),arrays[1].ravel())[0,1]),
        prediction_mean_abs_diff=float(np.mean(np.abs(arrays[1]-arrays[0])))))
    promising=delta>=.002 and ci[0]>0 and np.min(auc[1]-auc[0])>=-.02
    text=f'# Phase39 Fold1 paired result\n\nBASE {np.mean(auc[0]):.6f}; FUSION {np.mean(auc[1]):.6f}; delta {delta:+.6f}; patient bootstrap CI [{ci[0]:+.6f}, {ci[1]:+.6f}].\n\nState: {"PROMISING_PILOT" if promising else "COMPLETED_NO_GAIN"}. Fold1 historically reused; exploratory only. T4_UNVERIFIED. No Val/Test used for model selection.\n'
    (OUT/'final_decision.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['cpu','audit','preflight','run','analyze','status'],default='run')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='status':
        for t in TASKS:
            p=OUT/t/'status.json';print(t,p.read_text(encoding='utf-8') if p.exists() else 'PENDING')
        return
    if args.action=='cpu':print(cpu_tests());return
    archive,train_rows,oof_rows,_=audit()
    if args.action=='audit':print('PATIENT_ISOLATION_PASS');return
    if args.action=='analyze':analyze(oof_rows);return
    with lock(OUT):
        cpu_tests()
        receipt=OUT/'cuda_smoke_report.json'
        if not receipt.exists() or json.loads(receipt.read_text())['protocol_sha256']!=digest(OUT/'protocol_manifest.json'):
            cuda_smoke(archive,train_rows,oof_rows)
        if args.action=='preflight':print('PREFLIGHT_PASS');return
        for task in TASKS:
            try:run_task(task,archive,train_rows,oof_rows)
            except Exception as exc:
                run=OUT/task;run.mkdir(exist_ok=True)
                write_json(run/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),
                                                 resume_available=(run/'last_training.pt').is_file()))
                raise
        analyze(oof_rows)


if __name__=='__main__':main()
