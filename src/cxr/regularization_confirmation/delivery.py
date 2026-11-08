"""Opt-in private lightweight delivery; checkpoints/images/labels never pushed."""
import json
import shutil
import subprocess
from pathlib import Path
from cxr.phase39_support import digest,write_json
from cxr.phase42.protocol import verify_receipt
from cxr.regularization_confirmation.support import fingerprint

BRANCH='results/regularization-confirmation-20261108'
PATH_FIELDS={'archive','output','pretrained_root','protocol_root','holdout_index'}


def export_light(root,task_id):
    root=Path(root);src=root/'experiments'/task_id
    receipt=verify_receipt(src)
    target=root/'exports'/task_id;target.mkdir(parents=True,exist_ok=True)
    names=['protocol_manifest.json','bootstrap_comparison.json','per_class_comparison.csv','final_decision.md',
           'cpu_smoke_report.json','cuda_smoke_report.json','runtime_environment.json']
    for arm in ('BASE','MIXUP','NO_PHOTO'):
        if not (src/arm).is_dir():continue
        names.extend(f'{arm}/{name}' for name in ('resolved_config.json','train_history.csv','outer_oof_predictions.csv','outer_metrics.json','TRAIN_DONE.json'))
    hashes={};source_hashes={}
    for rel in names:
        p=src/rel
        if not p.exists():raise FileNotFoundError(p)
        if p.stat().st_size>32*2**20:raise ValueError('Unexpected large lightweight file')
        dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if p.name=='resolved_config.json':
            obj=json.loads(p.read_text(encoding='utf-8'))
            for k in PATH_FIELDS:
                if k in obj:obj[k]='REDACTED_SERVER_PATH'
            payload=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
        else:payload=p.read_bytes()
        if dst.exists() and dst.read_bytes()!=payload:raise ValueError('Completed export changed; preserve historical version')
        if not dst.exists():dst.write_bytes(payload)
        hashes[rel]=digest(dst);source_hashes[rel]=digest(p)
    marker=dict(kind='regularization_confirmation_v1',task_id=task_id,code_commit=receipt['code_commit'],
        identity_sha256=receipt['identity_sha256'],sha256=hashes,source_sha256=source_hashes,
        original_weight_hashes={k:v for k,v in receipt['sha256'].items() if k.endswith('ema_best.pth')},
        weights_status='WEIGHTS_NOT_DOWNLOADED',redacted_fields=sorted(PATH_FIELDS),images_included=False,labels_included=False)
    write_json(target/'export_manifest.json',marker)
    return marker


def publish(root,settings):
    if not settings.get('enabled'):raise ValueError('Friend must explicitly enable private result publishing')
    if settings.get('repository')!='Zoey-711/cxr-multilabel-challenge' or settings.get('branch')!=BRANCH:raise ValueError('Dedicated private results target required')
    def gh(*args):return subprocess.check_output(['gh',*args],text=True).strip()
    meta=json.loads(gh('repo','view',settings['repository'],'--json','isPrivate,viewerPermission'))
    if not meta['isPrivate'] or meta['viewerPermission'] not in ('ADMIN','WRITE','MAINTAIN'):raise ValueError('Private write access required')
    root=Path(root).resolve();checkout=Path(settings['checkout']).resolve()
    from cxr.regularization_confirmation.support import ROOT
    for other in (root,ROOT):
        if checkout==other or checkout in other.parents or other in checkout.parents:raise ValueError('Results checkout overlaps source/output')
    checkout.mkdir(parents=True,exist_ok=True)
    remote='https://github.com/'+settings['repository']+'.git'
    def git(*args):return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    if not (checkout/'.git').exists():
        git('init');git('remote','add','origin',remote)
        if git('ls-remote','--heads','origin',BRANCH):
            git('fetch','origin',BRANCH);git('checkout','-b',BRANCH,'FETCH_HEAD')
            marker=checkout/'results/regularization_confirmation/batch_identity.json'
            if not marker.exists() or json.loads(marker.read_text())!={'kind':'regularization_confirmation_v1'}:raise ValueError('Result branch collision')
        else:git('checkout','--orphan',BRANCH)
    if git('branch','--show-current')!=BRANCH or git('remote','get-url','origin')!=remote:raise ValueError('Wrong upload branch/repository')
    if git('status','--porcelain'):raise ValueError('Result checkout has uncommitted edits; inspect rather than overwrite')
    destination=checkout/'results/regularization_confirmation';destination.mkdir(parents=True,exist_ok=True)
    for marker in (root/'exports').glob('*/export_manifest.json'):
        obj=json.loads(marker.read_text(encoding='utf-8'));source=marker.parent
        for rel,sha in obj['sha256'].items():
            if digest(source/rel)!=sha:raise ValueError('Export hash mismatch')
        # Source identity subdirectory preserves reruns and older server versions.
        dest=destination/obj['identity_sha256']/obj['task_id']
        for p in source.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(source);q=dest/relative;q.parent.mkdir(parents=True,exist_ok=True)
            if q.exists() and digest(q)!=digest(p):raise ValueError('Remote history collision')
            if not q.exists():shutil.copyfile(p,q)
    write_json(destination/'batch_identity.json',dict(kind='regularization_confirmation_v1'))
    git('add','--','results/regularization_confirmation')
    if subprocess.run(['git','-C',str(checkout),'diff','--cached','--quiet']).returncode:
        git('commit','-m','Add verified regularization confirmation results')
    git('push','origin','HEAD:refs/heads/'+BRANCH)
    return dict(branch=BRANCH,commit=git('rev-parse','HEAD'),weights_uploaded=False)
