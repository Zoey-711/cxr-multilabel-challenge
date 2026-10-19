"""Versioned, verified portable bundles. No credentials or raw images are packed."""
import json
import tarfile
from datetime import datetime,timezone
from pathlib import Path

from cxr.batch.plan import ROOT,digest,write_json


def archive(path, members):
    with tarfile.open(path,'w:gz',compresslevel=3) as tar:
        for file,relative in sorted(members,key=lambda x:x[1]):
            if file.is_symlink():
                raise ValueError('Symlink not allowed in delivery')
            tar.add(file,arcname=relative,recursive=False)
    # Read back archive entries and verify content hashes, not merely archive checksum.
    import hashlib
    expected={rel:digest(file) for file,rel in members}
    with tarfile.open(path,'r:gz') as tar:
        actual={}
        for item in tar:
            if not item.isfile():raise ValueError('Unexpected archive member')
            actual[item.name]=hashlib.file_digest(tar.extractfile(item),'sha256').hexdigest()
    if actual!=expected:raise ValueError('Archive round-trip integrity failed')
    return expected


def split_archive(path, limit):
    if path.stat().st_size<=limit:return [path]
    parts=[]
    with path.open('rb') as stream:
        i=1
        while True:
            block=stream.read(min(limit,8*2**20))
            if not block:break
            part=path.with_name(path.name+f'.part{i:04d}'); size=0
            with part.open('xb') as out:
                while block:
                    out.write(block);size+=len(block)
                    if size>=limit:break
                    block=stream.read(min(limit-size,8*2**20))
            parts.append(part);i+=1
    path.unlink()  # Only the newly generated temporary combined archive, never source artifacts.
    return parts


def pack(root, *, include_resume=False, part_bytes=1_000_000_000):
    root=Path(root).resolve()
    if (root/'queue.lock').exists():
        raise ValueError('Queue must be drained before final packing')
    if part_bytes<1024:raise ValueError('Volume size too small')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=root/'delivery'/stamp;out.mkdir(parents=True,exist_ok=False)
    results=[];models=[];resumes=[]
    allowed={'.json','.yaml','.yml','.csv','.md','.log','.txt','.sha256','.py','.toml','.sh'}
    for file in root.rglob('*'):
        if not file.is_file() or root/'delivery' in file.parents:continue
        rel='phase17_batch/'+file.relative_to(root).as_posix()
        if file.name in ('best_model.pth','ema_best_model.pth'):
            models.append((file,rel))
        elif file.name=='checkpoint_resume.pth':
            resumes.append((file,rel))
        elif file.suffix in allowed and not any(x in file.name.lower() for x in ('id_rsa','id_ed25519','.env','password','token','secret')):
            results.append((file,rel))
    for run in (root/'experiments').iterdir():
        if not run.is_dir():continue
        for name in ('resolved_config.yaml','run_manifest.json','best_model.sha256','ema_best_model.sha256'):
            p=run/name
            if p.exists():
                models.append((p,'phase17_batch/'+p.relative_to(root).as_posix()))
    for file in (ROOT/'src').rglob('*.py'):
        results.append((file,'code/'+file.relative_to(ROOT).as_posix()))
    for name in ('scripts/batch_delivery.py','scripts/import_friend_results.py','pyproject.toml','README.md','README_results.md'):
        file=ROOT/name
        if file.exists():results.append((file,'code/'+name))
    done=[p.parent.name for p in (root/'experiments').glob('*/DONE.json')]
    plan=__import__('yaml').safe_load((root/'manifests/batch_plan.yaml').read_text(encoding='utf-8'))
    expected=[t['id'] for t in plan['tasks'] if t['enabled']]
    packages={};inventory={}
    for name,members in [('results_bundle.tar.gz',results),('models_bundle.tar.gz',models)]+([('resume_bundle.tar.gz',resumes)] if include_resume else []):
        path=out/name;inventory[name]=archive(path,members)
        for part in split_archive(path,part_bytes):
            packages[part.name]={'bytes':part.stat().st_size,'sha256':digest(part),'archive':name}
    manifest={'state':'verified','generated_utc':datetime.now(timezone.utc).isoformat(),
              'analysis_complete':(root/'analysis/statistics.json').exists() and not (root/'reports/ANALYSIS_FAILED.json').exists(),
              'packages':packages,'archive_members':inventory,'completed_tasks':done,
              'completed_task_count':len(done),'missing_tasks':[t for t in expected if t not in done],
              'model_count':sum(p.name in ('best_model.pth','ema_best_model.pth') for p,_ in models),
              'source':json.loads((root/'manifests/environment.json').read_text(encoding='utf-8')),
              'data_sha256':plan.get('data_sha256'),'fold_hashes':plan.get('fold_hashes'),
              'extraction':'Verify all parts, concatenate numbered parts in order, then safe-extract; never overwrite existing files.'}
    write_json(out/'delivery_manifest.json',manifest)
    if (ROOT/'README_results.md').exists():
        __import__('shutil').copy2(ROOT/'README_results.md',out/'README_results.md')
    (out/'COMPLETE.txt').write_text(f'Packaging verified. Completed: {len(done)}/{len(expected)}.\nMissing: {manifest["missing_tasks"]}\nDownload directory: {out}\nNo remote results were deleted.\n',encoding='utf-8')
    write_json(root/'delivery/latest.json',{'directory':out.name,'manifest_sha256':digest(out/'delivery_manifest.json')})
    return out


def verify_and_extract(bundle,destination):
    bundle,destination=Path(bundle).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    manifest=json.loads((bundle/'delivery_manifest.json').read_text(encoding='utf-8'))
    for name,meta in manifest['packages'].items():
        if Path(name).name!=name:raise ValueError('Unsafe package path')
        p=bundle/name
        if p.stat().st_size!=meta['bytes'] or digest(p)!=meta['sha256']:
            raise ValueError(f'Package checksum mismatch: {name}')
    destination.mkdir(parents=True)
    for name,expected in manifest['archive_members'].items():
        parts=sorted(n for n,m in manifest['packages'].items() if m['archive']==name)
        combined=destination/('_'+Path(name).name)
        with combined.open('xb') as f:
            for part in parts:
                with (bundle/part).open('rb') as source:__import__('shutil').copyfileobj(source,f,8*2**20)
        seen=set()
        with tarfile.open(combined,'r:gz') as tar:
            for member in tar:
                target=(destination/member.name).resolve()
                if not member.isfile() or destination not in target.parents or member.name not in expected or member.name in seen:
                    raise ValueError('Unsafe, duplicate or unexpected archive member')
                seen.add(member.name)
                target.parent.mkdir(parents=True,exist_ok=True)
                # Config/manifest duplicated between model and result packages must be identical.
                if target.exists():
                    if digest(target)!=expected[member.name]:raise FileExistsError(target)
                else:
                    with target.open('xb') as f:__import__('shutil').copyfileobj(tar.extractfile(member),f)
                if digest(target)!=expected[member.name]:raise ValueError('Extracted file checksum mismatch')
        if seen!=set(expected):raise ValueError('Archive missing members')
        combined.unlink()
    write_json(destination/'EXTRACTION_VERIFIED.json',{'state':'verified','delivery_manifest_sha256':digest(bundle/'delivery_manifest.json')})
    return destination
