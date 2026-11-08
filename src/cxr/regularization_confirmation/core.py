"""Fixed BASE/MIXUP comparison with inner selection and untouched outer Fold2.

Only local supervised training. No official Val/Test, teachers or remote writes.
Epoch-boundary restart includes optimizer, EMA, scaler and independent Mixup RNG.
"""
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.data.archive import read_split_rows,image_member,parse_positive_classes
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.engine.metrics import binary_auc
from cxr.models.factory import create_model
from cxr.phase39_support import digest, write_json, write_csv, save_torch, seed_all
from cxr.regularization_confirmation.support import tensor_hash
from cxr.phase39_run import infer, optimizer_scheduler
from cxr.utils.training_resume import capture_rng_state, restore_rng_state

ROOT = Path(__file__).resolve().parents[3]
AUG = dict(enabled=True, name='conservative_cxr_v1', affine_probability=.5,
           rotation_degrees=3., translation_fraction=.02, photometric_probability=.5,
           brightness_contrast_fraction=.05)


def csvrows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def seal(path, record):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Frozen identity changed: ' + str(path))
    else:
        write_json(path, record)


def audit(cfg, config_path, out):
    weight = Path(cfg['pretrained_root']) / 'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth'
    if digest(weight) != cfg['pretrained_sha256']:
        raise ValueError('ImageNet hash mismatch')
    folds = Path(cfg['protocol_root'])
    for name, sha in cfg['fold_sha256'].items():
        if digest(folds/name) != sha: raise ValueError('Fold hash mismatch: '+name)
    hold = Path(cfg['holdout_index'])
    if digest(hold) != cfg['holdout_index_sha256']: raise ValueError('Holdout hash mismatch')
    hp = csvrows(hold)
    splits = sorted({r['split'] for r in hp})
    if splits != ['internal_train', 'internal_val']:
        raise ValueError('Unexpected locked split names: '+str(splits))
    excluded = {r['Subject_id'] for r in hp if r['split']=='internal_val'}
    fm = {r['Subject_id']:int(r['fold']) for r in csvrows(folds/'patient_folds.csv')}
    rows = read_split_rows(Path(cfg['archive']), 'train')
    if {r['Subject_id'] for r in rows} != set(fm): raise ValueError('Patient index mismatch')
    image_index=csvrows(folds/'image_folds.csv')
    expected={(r['Subject_id'],r['Study_id'],r['Dicom_id']):(r['archive_member'],int(r['fold'])) for r in image_index}
    actual={(r['Subject_id'],r['Study_id'],r['Dicom_id']):(image_member(r,'train'),fm[r['Subject_id']]) for r in rows}
    if len(expected)!=len(image_index) or len(actual)!=len(rows) or actual!=expected:raise ValueError('Frozen image/Study mapping mismatch')
    frozen_studies={(r['Subject_id'],r['Study_id']):r['Predict_class'] for r in csvrows(folds/'study_folds.csv')}
    if any((r['Subject_id'],r['Study_id']) not in frozen_studies or
           parse_positive_classes(frozen_studies[(r['Subject_id'],r['Study_id'])])!=parse_positive_classes(r['Predict_class']) for r in rows):
        raise ValueError('Frozen label mapping mismatch')
    usable = set(fm)-excluded
    outer = {p for p in usable if fm[p]==cfg['fold']}
    remaining = sorted(usable-outer, key=lambda p:hashlib.sha256(('MIXUP_CONFIRM_V1|'+p).encode()).hexdigest())
    inner = set(remaining[:math.ceil(.1*len(remaining))]); train=set(remaining)-inner
    if train&inner or train&outer or inner&outer or (train|inner|outer)&excluded:
        raise ValueError('Patient leakage')
    groups = {k:[r for r in rows if r['Subject_id'] in ids] for k,ids in
              [('train',train),('inner',inner),('outer',outer)]}
    counts = {}
    for name, rr in groups.items():
        targets=study_targets_from_rows(rr)
        # Labels are audited for support, never used to choose the split.
        y=np.asarray([t.labels for t in targets],dtype=float)
        if not np.isin(y,[0,1]).all() or (y.sum(0)==0).any() or (y.sum(0)==len(y)).any():
            raise ValueError('Undefined class AUC in fixed split: '+name)
        counts[name]=dict(patients=len({r['Subject_id'] for r in rr}),studies=len(targets),images=len(rr),positive_studies=y.sum(0).astype(int).tolist())
    sources=[Path(__file__),ROOT/'scripts/confirmation_run.py',ROOT/'src/cxr/regularization_confirmation/support.py',ROOT/'src/cxr/phase39_run.py',
             ROOT/'src/cxr/data/dataset.py',ROOT/'src/cxr/models/factory.py',ROOT/'src/cxr/utils/training_resume.py']
    record=dict(state='PASS',config_sha256=digest(config_path),code_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                pretrained_sha256=digest(weight),fold_sha256=cfg['fold_sha256'],holdout_index_sha256=digest(hold),
                excluded_holdout_patients=len(excluded),counts=counts,patient_overlap=0,
                official_val_accessed=False,test_accessed=False,selection=cfg['selection'])
    seal(out/'protocol_manifest.json',record)
    splitrows=[dict(Subject_id=p,fold=fm[p],role=('excluded_holdout' if p in excluded else 'outer' if p in outer else 'inner' if p in inner else 'train')) for p in sorted(fm)]
    dest=out/'patient_roles.csv'
    if dest.exists():
        if csvrows(dest)!=[{k:str(v) for k,v in r.items()} for r in splitrows]: raise ValueError('Split changed')
    else: write_csv(dest,splitrows)
    if shutil.disk_usage(out).free < 8*2**30: raise RuntimeError('Less than 8 GiB free disk')
    return groups


def loader(cfg, rows, training=False, arm="BASE"):
    aug = dict(AUG)
    if arm == "NO_PHOTO":
        aug["brightness_contrast_fraction"] = 0.0  # Preserve random draw count.
    return make_dataloader(Path(cfg['archive']),'train',rows=rows,shuffle=training,seed=42 if training else 43,
        batch_size=cfg['micro_batch'],num_workers=cfg['workers'],persistent_workers=False,
        augmentation=aug if training else None,preprocessing='black_pad_v1',input_size=cfg['input_size'])


class MixRNG:
    def __init__(self):
        self.numpy=np.random.default_rng(424242)
        self.torch=torch.Generator(device='cpu').manual_seed(424242)
    def state(self): return dict(numpy=self.numpy.bit_generator.state,torch=self.torch.get_state())
    def restore(self,s):
        self.numpy.bit_generator.state=s['numpy'];self.torch.set_state(s['torch'])
    def apply(self,x,y):
        lam=float(self.numpy.beta(.2,.2));idx=torch.randperm(len(x),generator=self.torch).to(x.device)
        return lam*x+(1-lam)*x[idx],lam*y+(1-lam)*y[idx]


def build(cfg, steps, pretrained=True, device='cuda'):
    seed_all(cfg['seed']);os.environ['TORCH_HOME']=str(Path(cfg['pretrained_root']))
    model=create_model('efficientnet_v2_s',num_classes=10,pretrained=pretrained).to(device)
    ema=AveragedModel(model,device=device,multi_avg_fn=get_ema_multi_avg_fn(cfg['ema_decay']),use_buffers=True)
    opt,sch=optimizer_scheduler(cfg,model,steps)
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_init_scale'],enabled=device=='cuda')
    return model,ema,opt,sch,scaler


def update(micro, model, ema, opt, sch, scaler, rng, mixup, device='cuda'):
    opt.zero_grad(set_to_none=True)
    denom=sum(b['target'].numel() for b in micro);total=0.
    for b in micro:
        x=b['image'].to(device);y=b['target'].to(device)
        if not bool(b['valid_mask'].all()) or not bool(((y==0)|(y==1)).all()): raise ValueError('Incomplete binary labels')
        if not bool(torch.isfinite(x).all()): raise FloatingPointError('Nonfinite input')
        if mixup:x,y=rng.apply(x,y)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
            logits=model(x)
            loss=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='sum')/denom
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite logits/loss')
        scaler.scale(loss).backward();total+=float(loss.detach())
    scaler.unscale_(opt)
    finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)) if finite else None
    if norm is not None and not math.isfinite(norm):
        raise FloatingPointError('Nonfinite aggregate gradient norm')
    before=scaler.get_scale();scaler.step(opt);scaler.update()
    skipped=scaler.get_scale()<before
    if skipped != (not finite): raise FloatingPointError('Unexpected GradScaler state')
    if not skipped:sch.step();ema.update_parameters(model)
    opt.zero_grad(set_to_none=True)
    return dict(loss=total,skipped=skipped,gradient_norm=norm,scale_before=before,scale_after=scaler.get_scale())


def state(cfgsha, epoch, objects, tracker, history, tr, ev, rng):
    m,e,o,s,g=objects
    return dict(config_sha256=cfgsha,epoch=epoch,model=m.state_dict(),ema=e.state_dict(),optimizer=o.state_dict(),
        scheduler=s.state_dict(),scaler=g.state_dict(),tracker=asdict(tracker),history=history,
        mixup_rng=rng.state(),**capture_rng_state(tr,ev))


def restore(saved, objects, tr, ev, rng):
    def finite(value):
        if isinstance(value,torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite tensor in resume checkpoint')
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
    for key in ('model','ema','optimizer'):finite(saved[key])
    for obj,key in zip(objects,['model','ema','optimizer','scheduler','scaler']):obj.load_state_dict(saved[key])
    restore_rng_state(saved,tr,ev);rng.restore(saved['mixup_rng'])
    return EarlyStopTracker(**saved['tracker'])


def smoke(cfg, groups, out, cfgsha):
    torch.set_num_threads(2)
    # CPU stochastic-stream isolation and mixup-label convexity.
    seed_all(42);before=torch.get_rng_state().clone();nr=np.random.get_state()
    rng=MixRNG();x=torch.arange(24.).reshape(4,6);y=torch.eye(4)
    a,b=rng.apply(x,y)
    assert torch.equal(before,torch.get_rng_state()) and np.array_equal(nr[1],np.random.get_state()[1])
    assert bool(((b>=0)&(b<=1)).all()) and torch.allclose(b.sum(1),torch.ones(4))
    st=rng.state();a1,b1=rng.apply(x,y);rng.restore(st);a2,b2=rng.apply(x,y)
    assert torch.equal(a1,a2) and torch.equal(b1,b2)
    write_json(out/'cpu_smoke_report.json',dict(state='PASS',independent_mixup_rng=True,convex_labels=True,rng_restore=True))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    reports={}
    for arm in cfg['arms']:
        tr=loader(cfg,groups['train'],True,arm);ev=loader(cfg,groups['inner'])
        objects=build(cfg,math.ceil(len(tr)/cfg['accumulation']));m,e,o,s,g=objects;rng=MixRNG()
        initial=tensor_hash(m.state_dict());m.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        it=iter(tr);events=[]
        for _ in range(3):
            events.append(update(list(itertools.islice(it,cfg['accumulation'])),*objects,rng,arm=='MIXUP'))
        if any(v['skipped'] for v in events): raise FloatingPointError('Smoke skipped optimizer update')
        saved=state(cfgsha,0,objects,EarlyStopTracker(cfg['patience']),[],tr,ev,rng)
        path=out/('smoke_'+arm+'.pt');save_torch(path,saved)
        sample=next(iter(ev))['image'].cuda();m.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): ref=m(sample).clone()
        expected_rng=rng.apply(torch.arange(24.).reshape(4,6),torch.eye(4))
        expected_draw=torch.rand(5)
        restored=torch.load(path,map_location='cpu',weights_only=True)
        restore(restored,objects,tr,ev,rng)
        # Reset to captured RNG before the evaluation loader consumption for exact next-draw check.
        rng.restore(saved['mixup_rng'])
        rr=rng.apply(torch.arange(24.).reshape(4,6),torch.eye(4))
        assert torch.equal(rr[0],expected_rng[0])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): error=float((m(sample)-ref).abs().max())
        assert error==0 and int(e.n_averaged)==3 and s.last_epoch==3
        reports[arm]=dict(state='PASS',events=events,initial_state_sha256=initial,restore_max_abs_error=error,
            ema_updates=int(e.n_averaged),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.perf_counter()-start)
        path.unlink();del objects,m,e,o,s,g,saved,restored,tr,ev,it,sample,ref;gc.collect();torch.cuda.empty_cache()
    assert reports['BASE']['initial_state_sha256']==reports[cfg['arms'][1]]['initial_state_sha256']
    write_json(out/'cuda_smoke_report.json',dict(state='PASS',config_sha256=cfgsha,gpu=torch.cuda.get_device_name(),arms=reports))
    print(json.dumps(reports),flush=True)


def train(cfg, groups, out, cfgsha, arm):
    dest=out/arm;dest.mkdir(exist_ok=True)
    if (dest/'TRAIN_DONE.json').exists():
        done=json.loads((dest/'TRAIN_DONE.json').read_text())
        if done['config_sha256']!=cfgsha or done['checkpoint_sha256']!=digest(dest/'ema_best.pth'):raise ValueError('Completed training identity mismatch')
        return
    seal(dest/'resolved_config.json',dict(cfg,arm=arm,mixup_enabled=arm=='MIXUP',augmentation=dict(AUG,brightness_contrast_fraction=0.0 if arm=='NO_PHOTO' else .05)))
    tr=loader(cfg,groups['train'],True,arm);ev=loader(cfg,groups['inner'])
    steps=math.ceil(len(tr)/cfg['accumulation']);objects=build(cfg,steps);m,e,o,s,g=objects;rng=MixRNG()
    tracker=EarlyStopTracker(cfg['patience']);history=[];first=1
    if (dest/'last_training.pt').exists():
        saved=torch.load(dest/'last_training.pt',map_location='cpu',weights_only=True)
        if saved['config_sha256']!=cfgsha:raise ValueError('Resume protocol mismatch')
        best=dest/saved['best_checkpoint_name']
        if best.name!=saved['best_checkpoint_name'] or digest(best)!=saved['best_checkpoint_sha256']:raise ValueError('Resume best checkpoint mismatch')
        shutil.copyfile(best,dest/'ema_best.pth')
        tracker=restore(saved,objects,tr,ev,rng);history=saved['history'];first=saved['epoch']+1;del saved
    targets=study_targets_from_rows(groups['inner']);consecutive=0
    total_skips=sum(r['amp_skips'] for r in history)
    with (dest/'train.log').open('a',encoding='utf-8') as log:
        def emit(msg):print(msg,flush=True);log.write(msg+'\n');log.flush()
        emit(f'START {arm} epoch={first} steps={steps} micro={cfg["micro_batch"]} accumulation={cfg["accumulation"]}')
        for epoch in range(first,cfg['epochs']+1):
            if tracker.stale>=cfg['patience']:break
            m.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();it=iter(tr);loss=0.;skips=0
            for step in range(1,steps+1):
                micro=list(itertools.islice(it,cfg['accumulation']))
                result=update(micro,*objects,rng,arm=='MIXUP');loss+=result['loss'];skips+=int(result['skipped'])
                total_skips+=int(result['skipped']);consecutive=consecutive+1 if result['skipped'] else 0
                if result['skipped']:
                    with (dest/'amp_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,step=step,**result))+'\n')
                if consecutive>=cfg['max_consecutive_skips'] or total_skips>cfg['max_total_skips']:
                    raise FloatingPointError('AMP skip safety limit; retain previous epoch checkpoint')
                if step==1 or step%20==0 or step==steps:
                    status=dict(state='RUNNING',pid=os.getpid(),epoch=epoch,step=step,steps=steps,inner_best_auc=tracker.best_auc,
                        best_epoch=tracker.best_epoch,amp_skips=skips,total_amp_skips=total_skips,ema_updates=int(e.n_averaged),
                        elapsed_epoch_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    write_json(dest/'status.json',status);emit(json.dumps(status))
            auc=infer(e,ev,targets,None,torch.device('cuda'))['macro_auc'];decision=tracker.observe(auc,epoch)
            history.append(dict(epoch=epoch,loss=loss/steps,inner_ema_auc=auc,amp_skips=skips,optimizer_updates=steps-skips,
                ema_updates=int(e.n_averaged),seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            best_name=f'ema_best_epoch_{tracker.best_epoch:02d}.pth'
            if decision.new_best_checkpoint:save_torch(dest/best_name,e.module.state_dict())
            snapshot=state(cfgsha,epoch,objects,tracker,history,tr,ev,rng)
            snapshot.update(best_checkpoint_name=best_name,best_checkpoint_sha256=digest(dest/best_name))
            save_torch(dest/'last_training.pt',snapshot)
            shutil.copyfile(dest/best_name,dest/'ema_best.pth')
            write_csv(dest/'train_history.csv',history);emit('EPOCH_DONE '+json.dumps(history[-1]))
            if decision.should_stop:break
    write_json(dest/'TRAIN_DONE.json',dict(state='TRAIN_DONE',best_epoch=tracker.best_epoch,inner_auc=tracker.best_auc,
        checkpoint_sha256=digest(dest/'ema_best.pth'),config_sha256=cfgsha))
    write_json(dest/'status.json',dict(state='TRAIN_DONE',best_epoch=tracker.best_epoch,inner_auc=tracker.best_auc))
    del objects,m,e,o,s,g,tr,ev;gc.collect();torch.cuda.empty_cache()


def evaluate(cfg,groups,out):
    outer=groups['outer'];ev=loader(cfg,outer);targets=study_targets_from_rows(outer)
    results={}
    for arm in cfg['arms']:
        dest=out/arm;done=json.loads((dest/'TRAIN_DONE.json').read_text())
        if digest(dest/'ema_best.pth')!=done['checkpoint_sha256']:raise ValueError('Best checkpoint changed')
        model=create_model('efficientnet_v2_s',num_classes=10,pretrained=False).cuda()
        model.load_state_dict(torch.load(dest/'ema_best.pth',map_location='cpu',weights_only=True))
        results[arm]=infer(model,ev,targets,dest/'outer_oof_predictions.csv',torch.device('cuda'))
        write_json(dest/'outer_metrics.json',results[arm]);del model;gc.collect();torch.cuda.empty_cache()
    raw=[csvrows(out/a/'outer_oof_predictions.csv') for a in cfg['arms']]
    maps=[{(r['Subject_id'],r['Study_id']):r for r in rows} for rows in raw]
    if any(len(rows)!=len(mapping) for rows,mapping in zip(raw,maps)):raise ValueError('Duplicate OOF Study')
    challenger=cfg['arms'][1]
    keys=sorted(maps[0]);assert set(maps[0])==set(maps[1])
    labels={ (r['Subject_id'],r['Study_id']):[int(str(i) in r['Predict_class'].split(';')) for i in range(1,11)] for r in outer}
    assert set(labels)==set(keys)
    y=np.asarray([labels[k] for k in keys]);ps=[np.asarray([[m[k][f'Prob_{j}'] for j in range(1,11)] for k in keys],dtype=float) for m in maps]
    if any(not np.isfinite(p).all() or ((p<0)|(p>1)).any() for p in ps):raise ValueError('Invalid OOF probabilities')
    patients=sorted({k[0] for k in keys});idx={p:np.asarray([i for i,k in enumerate(keys) if k[0]==p]) for p in patients}
    rng=np.random.default_rng(cfg['bootstrap_seed']);deltas=[];invalid=0
    for _ in range(cfg['bootstrap_replicates']):
        ii=np.concatenate([idx[p] for p in rng.choice(patients,len(patients),replace=True)])
        if (y[ii].sum(0)==0).any() or (y[ii].sum(0)==len(ii)).any():invalid+=1;continue
        deltas.append(float(np.mean([binary_auc(y[ii,j],ps[1][ii,j])-binary_auc(y[ii,j],ps[0][ii,j]) for j in range(10)])))
    per=[dict(class_id=j+1,base_auc=results['BASE']['per_class'][j]['auc'],challenger_auc=results[challenger]['per_class'][j]['auc'],
              delta=results[challenger]['per_class'][j]['auc']-results['BASE']['per_class'][j]['auc'],positive_studies=int(y[:,j].sum())) for j in range(10)]
    delta=results[challenger]['macro_auc']-results['BASE']['macro_auc'];passed=delta>=.001 and min(r['delta'] for r in per)>=-.01
    report=dict(scope=f"OUTER_FOLD{cfg['fold']}_HOLDOUT_EXCLUDED_EXPLORATORY",challenger=challenger,base_auc=results['BASE']['macro_auc'],challenger_auc=results[challenger]['macro_auc'],delta=delta,
        ci95=np.quantile(deltas,[.025,.975]).tolist() if deltas else None,valid_bootstraps=len(deltas),invalid_bootstraps=invalid,
        invalid_policy='discard replicates missing either label for any class; no resampling to reach 2000',
        studies=len(keys),patients=len(patients),decision='PROMISING_PILOT' if passed else 'COMPLETED_NO_GAIN',further_training_started=False)
    write_csv(out/'per_class_comparison.csv',per);write_json(out/'bootstrap_comparison.json',report)
    (out/'final_decision.md').write_text('# Fixed regularization confirmation\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\nNo official Val/Test selection. No automatic confirmation-fold launch. Historical folds are not a blind test.\n',encoding='utf-8')
    write_json(out/'manifest_sha256.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest_sha256.json','queue.lock','queue_stdout.log','queue_stderr.log']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--action',choices=['audit','smoke','queue'],required=True);args=p.parse_args()
    cp=Path(args.config).resolve();cfg=json.loads(cp.read_text(encoding='utf-8'));out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);groups=audit(cfg,cp,out);cfgsha=digest(cp)
    if args.action=='audit':print((out/'protocol_manifest.json').read_text());return
    if args.action=='smoke':smoke(cfg,groups,out,cfgsha);return
    gate=json.loads((out/'cuda_smoke_report.json').read_text())
    if gate['state']!='PASS' or gate['config_sha256']!=cfgsha:raise ValueError('Smoke gate mismatch')
    lock=out/'queue.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        for arm in cfg['arms']:train(cfg,groups,out,cfgsha,arm)
        if not (out/'final_decision.md').exists():evaluate(cfg,groups,out)
        write_json(out/'queue_status.json',dict(state='DONE',pid=os.getpid()))
    except BaseException:
        write_json(out/'queue_status.json',dict(state='FAILED',pid=os.getpid(),traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
