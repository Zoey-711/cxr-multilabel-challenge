"""Explicit opt-in, small-result-only push into a separate private result branch."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from cxr.batch.plan import digest,write_json

ALLOWED={'metrics.json','per_class_metrics.csv','val_predictions.csv','oof_predictions.csv',
         'ema_val_predictions.csv','ema_oof_predictions.csv','resolved_config.yaml','model_size.json',
         'run_manifest.json','best_model.sha256','ema_best_model.sha256','train_history.csv'}

def export_small(root,destination):
    root,destination=Path(root),Path(destination);files={}
    for done in (root/'experiments').glob('*/DONE.json'):
        run=done.parent
        for p in run.iterdir():
            if p.name not in ALLOWED or not p.is_file():continue
            if p.stat().st_size>16*2**20:raise ValueError('Unexpected large small-result file')
            text=p.read_text(encoding='utf-8-sig')
            if re.search(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|BEGIN .*PRIVATE KEY|password\s*[:=])',text,re.I):
                raise ValueError('Possible secret in export; blocked')
            rel=p.relative_to(root);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);files[rel.as_posix()]=digest(target)
    for p in (root/'reports').glob('experiment_summary.*'):
        target=destination/'reports'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);files[target.relative_to(destination).as_posix()]=digest(target)
    write_json(destination/'export_manifest.json',{'sha256':files,'weights':'NOT_UPLOADED','raw_images':'EXCLUDED'})
    return files

def upload(root,settings):
    if not settings.get('enabled'):raise ValueError('Explicit result upload authorization required')
    repo=settings['repository'];branch=settings['branch'];checkout=Path(settings['checkout']).resolve()
    if repo!='Zoey-711/cxr-multilabel-challenge' or not re.fullmatch(r'results/e08-tune(?:-[a-zA-Z0-9_-]+)?',branch):
        raise ValueError('Only the separately authorized E08-tune results branch is allowed')
    meta=json.loads(subprocess.check_output(['gh','repo','view',repo,'--json','isPrivate,viewerPermission'],text=True))
    if not meta['isPrivate'] or meta['viewerPermission'] not in ('ADMIN','WRITE','MAINTAIN'):raise ValueError('Private write access not verified')
    remote='https://github.com/'+repo+'.git'
    def git(*args):return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    if not checkout.exists():
        checkout.mkdir(parents=True);git('init');git('remote','add','origin',remote)
        if git('ls-remote','--heads','origin',branch):
            git('fetch','origin',branch);git('checkout','-b',branch,'FETCH_HEAD')
        else:git('checkout','--orphan',branch)
    if git('remote','get-url','origin')!=remote or git('branch','--show-current')!=branch:
        raise ValueError('Upload checkout identity differs')
    root=Path(root)
    if root.name=='e08_tuning':
        destination=checkout/'results/e08_tuning'
        for sub in ('reports','oof','validation','manifests'):
            for p in (root/sub).glob('*'):
                if p.is_file() and p.suffix in ('.json','.csv','.md') and p.stat().st_size<16*2**20:
                    # Only generated aggregate products, never source images/labels/checkpoints.
                    if p.name not in ('five_fold_summary.csv','five_fold_analysis.json','selection.json','extension_status.json') and not (sub in ('oof','validation') and p.name.endswith(('.csv','.provenance.json'))):continue
                    target=destination/sub/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    else:export_small(root,checkout/'results/e08_tuning'/root.name)
    git('add','--','results/e08_tuning')
    if git('diff','--cached','--name-only'):
        git('commit','-m','E08-tune incremental completed-model results')
    git('push','origin',f'HEAD:refs/heads/{branch}') # no force; concurrent conflicts are reported
