"""Opt-in private Phase35 branch + uncompressed per-model Release assets."""
import json,re,shutil,subprocess,tempfile
from pathlib import Path
from cxr.batch.plan import digest,write_json
from cxr.batch.artifacts import commit
from cxr.phase35.results import verify_done

def upload(root,settings):
    if not settings.get('enabled'):raise ValueError('Friend must authorize result upload')
    repo=settings['repository'];branch=settings['branch'];tag=settings['release_tag'];checkout=Path(settings['checkout']).resolve();root=Path(root).resolve()
    if repo!='Zoey-711/cxr-multilabel-challenge' or not re.fullmatch(r'results/phase35-fivefold(?:-[A-Za-z0-9_-]+)?',branch):raise ValueError('Dedicated Phase35 branch required')
    if not re.fullmatch(r'phase35-[A-Za-z0-9_.-]+',tag):raise ValueError('Dedicated Phase35 Release tag required')
    if checkout==root or root in checkout.parents or checkout in root.parents:raise ValueError('Upload checkout overlaps results')
    def gh(*a):return subprocess.check_output(['gh',*a],text=True)
    meta=json.loads(gh('repo','view',repo,'--json','isPrivate,viewerPermission'))
    if not meta['isPrivate'] or meta['viewerPermission'] not in ('ADMIN','WRITE','MAINTAIN'):raise ValueError('Private write access required')
    remote='https://github.com/'+repo+'.git'
    def git(*a):return subprocess.check_output(['git','-C',str(checkout),*a],text=True).strip()
    if not checkout.exists():
        checkout.mkdir(parents=True);git('init');git('remote','add','origin',remote)
        if git('ls-remote','--heads','origin',branch):git('fetch','origin',branch);git('checkout','-b',branch,'FETCH_HEAD')
        else:git('checkout','--orphan',branch)
    if git('remote','get-url','origin')!=remote or git('branch','--show-current')!=branch:raise ValueError('Upload checkout identity differs')
    # Refuse unrelated edits; our generated subtree alone may be retried after a failed push.
    dirty=git('status','--porcelain')
    if any(not line[3:].startswith('results/phase35_batch') for line in dirty.splitlines()):raise ValueError('Unrelated upload checkout changes')
    runs=[]
    for done in (root/'experiments').glob('*/DONE.json'):verify_done(done.parent);runs.append(done.parent)
    for marker in (root/'reused_fold1').glob('*/REUSED.json'):
        data=json.loads(marker.read_text())
        if any(digest(marker.parent/r)!=h for r,h in data['sha256'].items()):raise ValueError('Reused model changed')
        runs.append(marker.parent)
    if not runs:return
    release=subprocess.run(['gh','release','view',tag,'--repo',repo],capture_output=True,text=True)
    if release.returncode:
        gh('release','create',tag,'--repo',repo,'--target',commit(),'--title',tag,'--notes','Phase35 inference weights. SHA-256 in results/phase35_batch. No raw images or training labels.')
    assets={a['name']:a for a in json.loads(gh('api',f'repos/{repo}/releases/tags/{tag}'))['assets']}
    binary=[]
    for run in runs:
        for name in ('best_model.pth','ema_best_model.pth'):
            p=run/name;sha=digest(p);asset=f'{run.name}__{name}';label='sha256:'+sha
            if asset in assets:
                if assets[asset]['size']!=p.stat().st_size or assets[asset].get('label')!=label:raise ValueError('Existing Release weight differs; never clobber')
            else:
                # Rename only in a temporary upload folder. No archive/compression, no optimizer states.
                with tempfile.TemporaryDirectory(prefix='phase35_upload_',dir=root) as tmp:
                    q=Path(tmp)/asset;shutil.copyfile(p,q)
                    gh('release','upload',tag,str(q)+'#'+label,'--repo',repo)
            binary.append({'model_id':run.name,'kind':'ema' if name.startswith('ema') else 'raw','asset':asset,'bytes':p.stat().st_size,'sha256':sha,'release_tag':tag})
    dest=checkout/'results/phase35_batch';files={}
    sources=[]
    for run in runs:
        sources.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.yaml','.yml','.sha256','.log','.jsonl','.txt') and not p.name.endswith('.pid.json'))
    for folder in ('reports','analysis','deployment','oof','validation','test_predictions'):
        sources.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md'))
    for p in sources:
        if p.stat().st_size>32*2**20:raise ValueError('Unexpected large text result')
        text=p.read_text(encoding='utf-8-sig')
        if re.search(r'github_pat_|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY|password\s*[:=]',text,re.I):raise ValueError('Possible secret in result')
        rel=p.relative_to(root);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files[rel.as_posix()]=digest(q)
    write_json(dest/'binary_release_manifest.json',{'repository':repo,'release_tag':tag,'assets':binary})
    write_json(dest/'export_manifest.json',{'files':files,'code_commit':commit(),'raw_images':'EXCLUDED','raw_label_files':'EXCLUDED','submission':'NOT_GENERATED'})
    git('add','--','results/phase35_batch')
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified Phase35 incremental results and inference weight links')
    git('push','origin',f'HEAD:refs/heads/{branch}')
