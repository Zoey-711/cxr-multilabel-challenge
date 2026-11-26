"""Independent CPU verification of Phase41R snapshots."""
import csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
ROOT=PROJECT.parent
SRC=ROOT/'cxr-friend-results/snapshots/phase41r-47cd764a/results/phase41r_batch'
prior=ROOT/'cxr-friend-results/analysis/phase43_phase41_final/analyze_phase43.py'
spec=importlib.util.spec_from_file_location('prior41',prior); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
sha=old.sha; pred=old.predictions; metric=old.metric; boot=old.bootstrap
train=pd.read_csv(PROJECT/'batch_assets/train_labels.csv'); val=pd.read_csv(PROJECT/'batch_assets/val_labels.csv')
assert len(train)==18513 and train.Subject_id.nunique()==11055 and len(val)==859
assert train.groupby('Subject_id').fold.nunique().max()==1
release=json.loads((OUT/'release_metadata.json').read_text(encoding='utf-8-sig'))
assets={x['name']:x for x in release['assets']}
assert sha(OUT/'binary_release_manifest.json')==assets['binary_release_manifest.json']['digest'].split(':',1)[1]
manifest=json.loads((SRC/'export_manifest.json').read_text(encoding='utf-8'))
bad=[]
for file,digest in manifest['files'].items():
    p=SRC/file
    if not p.is_file() or sha(p)!=digest:bad.append(file)
status=pd.read_csv(SRC/'reports/experiment_status.csv')
status.to_csv(OUT/'experiment_status.csv',index=False)
allpred={}; allval={}; folds=[]; classes=[]; audit=[]; skips=[]; weights=[]
for variant in ('BASE','FUSION'):
    parts=[]; vparts=[]
    f1=SRC/'reused_fold1'/f'P39_{variant}_F1'
    receipt=json.loads((f1/'REUSED.json').read_text(encoding='utf-8'))
    assert receipt['trained_again'] is False
    for name,digest in receipt['sha256'].items():
        if name.endswith('.pth'):
            a=assets[f'P39_{variant}_F1__ema_best.pth']; assert a['digest']=='sha256:'+digest
            weights.append((variant,1,a['size'],digest))
        else: assert sha(f1/name)==digest,(variant,name)
        local=PROJECT/'outputs/experiments/phase39_multistage_fusion'/f'P39_{variant}_F1'/name
        if name in ('ema_best.pth','oof_predictions.csv','resolved_config.json','run_manifest.json'):
            assert local.is_file() and sha(local)==digest,(variant,name,'local frozen mismatch')
    f1cfg=json.loads((f1/'resolved_config.json').read_text(encoding='utf-8'))
    for fold in range(1,6):
        d=f1 if fold==1 else SRC/'experiments'/f'P41_{variant}_F0{fold}'
        if fold>1:
            done=json.loads((d/'DONE.json').read_text(encoding='utf-8')); assert done['state']=='DONE'
            for name,digest in done['sha256'].items():
                if name.endswith(('.pt','.pth')):
                    if name=='ema_best.pth':
                        a=assets[f'P41_{variant}_F0{fold}__ema_best.pth']; assert a['digest']=='sha256:'+digest
                        weights.append((variant,fold,a['size'],digest))
                else: assert sha(d/name)==digest,(d,name)
            cfg=yaml.safe_load((d/'resolved_config.yaml').read_text(encoding='utf-8'))
            fields=['architecture','input_size','seed','micro_batch','accumulation','epochs','patience','ema_decay','ema_use_buffers','backbone_lr','head_lr','weight_decay','warmup_epochs','augmentation','preprocessing','pretrained_sha256','fold_sha256']
            changes={k:[f1cfg.get(k),cfg.get(k)] for k in fields if f1cfg.get(k)!=cfg.get(k)}
            audit.append(dict(task=d.name,config_differences=changes))
            hist=pd.read_csv(d/'train_history.csv'); ev=d/'amp_events.jsonl'
            events=[json.loads(x) for x in ev.read_text(encoding='utf-8').splitlines() if x.strip()] if ev.exists() else []
            skips.append(dict(task=d.name,epochs=len(hist),optimizer_updates=int(hist.optimizer_updates.sum()),amp_skips=int(hist.amp_skips.sum()),events=len(events),event_skips=sum(bool(e.get('optimizer_skipped')) for e in events)))
        o=pred(d/'oof_predictions.csv',train,fold); v=pred(d/'val_predictions.csv',val)
        assert len(v)==859
        t=pd.read_csv(d/'test_predictions.csv'); assert len(t)==1414 and not t.duplicated(old.KEY).any() and np.isfinite(t[old.PROBS].to_numpy()).all() and t[old.PROBS].apply(lambda c:c.between(0,1).all()).all()
        score,aucs,support,n,patients=metric(train,o)
        folds.append(dict(variant=variant,fold=fold,auc=score,studies=n,patients=patients))
        parts.append(o); vparts.append(v.sort_values(old.KEY).reset_index(drop=True))
    merged=pd.concat(parts,ignore_index=True); assert len(merged)==18513 and merged.Subject_id.nunique()==11055 and not merged.duplicated(old.KEY).any()
    assert merged.merge(train[old.KEY+['fold']],on=old.KEY,validate='one_to_one').groupby('Subject_id').fold.nunique().max()==1
    combined=vparts[0].copy(); combined[old.PROBS]=sum(v[old.PROBS] for v in vparts)/5
    allpred[variant]=merged; allval[variant]=combined
    saved=pd.read_csv(SRC/'validation'/f'{variant}_val_fivefold.csv').sort_values(old.KEY).reset_index(drop=True)
    assert saved[old.KEY].equals(combined[old.KEY])
    audit.append(dict(task=variant,scope='VAL_FIVEFOLD_STORED_VS_RECOMPUTED',max_probability_diff=float(np.max(np.abs(saved[old.PROBS].to_numpy()-combined[old.PROBS].to_numpy()))),stored_auc=metric(val,saved)[0]))
    for scope,truth,p in [('FULL_OOF',train,merged),('VAL_5FOLD',val,combined)]:
        score,aucs,support,n,patients=metric(truth,p)
        for i in range(10): classes.append(dict(scope=scope,variant=variant,class_id=i+1,auc=aucs[i],positive_studies=support[i]))
        audit.append(dict(task=variant,scope=scope,auc=score,studies=n,patients=patients))
partial=train[train.fold!=1]; pbase=allpred['BASE'].merge(partial[old.KEY],on=old.KEY,validate='one_to_one'); pfus=allpred['FUSION'].merge(partial[old.KEY],on=old.KEY,validate='one_to_one')
results={'full':boot(train,allpred['BASE'],allpred['FUSION']),'fold2_5':boot(partial,pbase,pfus),'val':boot(val,allval['BASE'],allval['FUSION'])}
results['fold2_5']['base_auc']=metric(partial,pbase)[0]; results['fold2_5']['fusion_auc']=metric(partial,pfus)[0]
pd.DataFrame(folds).to_csv(OUT/'fold_auc.csv',index=False); pd.DataFrame(classes).to_csv(OUT/'per_class_auc.csv',index=False)
pd.DataFrame(skips).to_csv(OUT/'amp_update_audit.csv',index=False); pd.DataFrame(weights,columns=['variant','fold','bytes','sha256']).to_csv(OUT/'checkpoint_metadata.csv',index=False)
output={'release_tag':release['tag_name'],'published_at':release['published_at'],'target_commitish':release['target_commitish'],'assets_count':len(assets),'assets':[{'name':x['name'],'bytes':x['size'],'digest':x['digest'],'state':x['state']} for x in release['assets']], 'branch':'results/phase41r-recovery','result_commit':'47cd764a4ae4414eedbaaeb08946562cf530550a','code_commit':manifest['code_commit'],'manifest_files':len(manifest['files']),'manifest_mismatches':bad,'model_files':'release metadata only, not locally downloaded','status':status.state.value_counts().to_dict(),'audit':audit,'amp':skips,'bootstrap':results,'weights_bytes':{v:sum(w[2] for w in weights if w[0]==v) for v in ('BASE','FUSION')}}
(OUT/'audit_results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:output[k] for k in ['manifest_files','manifest_mismatches','status','audit','amp','bootstrap','weights_bytes']},ensure_ascii=False))
