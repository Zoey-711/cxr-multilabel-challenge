"""Portable inference artifacts and ID-joined labels; no image data is exported."""
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cxr.batch.plan import ROOT, digest, write_json
from cxr.batch.results import read_table, scores, grouped

PROB_COLUMNS = ['Subject_id', 'Study_id'] + [f'Prob_{i}' for i in range(1, 11)]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def commit():
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def write_probabilities(path, table, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, p = table['keys'], np.asarray(table['p'])
    if len(set(keys)) != len(keys) or any(not a or not b for a,b in keys):
        raise ValueError('Duplicate or missing prediction IDs')
    if p.shape != (len(keys),10) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid prediction probabilities')
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(PROB_COLUMNS)
        w.writerows([*key,*vector] for key,vector in zip(keys,p))
    write_json(path.with_suffix('.provenance.json'), {**provenance, 'csv_sha256':digest(path),
               'rows':len(keys),'columns':PROB_COLUMNS,'aggregation':'Study Mean of image sigmoid probabilities'})


def read_probabilities(path, labels_path, expected_keys=None):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=PROB_COLUMNS:
            raise ValueError('Prediction columns/order must exactly match Prob_1..10')
        rows=list(reader)
    with Path(labels_path).open(encoding='utf-8-sig',newline='') as f:
        label_rows=list(csv.DictReader(f))
    labels={(r['Subject_id'],r['Study_id']):r for r in label_rows}
    if len(labels)!=len(label_rows):
        raise ValueError('Duplicate fixed label key')
    keys=[(r['Subject_id'],r['Study_id']) for r in rows]
    if len(keys)!=len(set(keys)) or any(not a or not b for a,b in keys):
        raise ValueError('Duplicate/empty prediction IDs')
    if not set(keys)<=set(labels) or (expected_keys is not None and set(keys)!=set(expected_keys)):
        raise ValueError('Prediction coverage differs from ID-joined evaluation labels')
    p=np.asarray([[float(r[f'Prob_{j}']) for j in range(1,11)] for r in rows])
    y=np.asarray([[int(labels[k][f'Target_{j}']) for j in range(1,11)] for k in keys])
    if p.shape!=(len(keys),10) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid probability/label values')
    return {'keys':keys,'p':p,'y':y,'counts':np.asarray([int(labels[k]['image_count']) for k in keys])}


def label_keys(path, fold=None):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return {(r['Subject_id'],r['Study_id']) for r in rows if fold is None or int(r['fold'])==fold}


def initialize_manifests(root, plan, plan_path):
    root=Path(root)
    for name in ('manifests','experiments','oof','validation','analysis','ensembles','reports','delivery'):
        (root/name).mkdir(parents=True,exist_ok=True)
    manifests=root/'manifests'
    shutil.copy2(plan_path,manifests/'batch_plan.yaml')
    for name,expected in plan['fold_hashes'].items():
        source=ROOT/plan['fold_directory']/name
        if digest(source)!=expected:
            raise ValueError('Frozen fold hash mismatch')
        shutil.copy2(source,manifests/name)
    for name in ('train_labels.csv','val_labels.csv'):
        shutil.copy2(ROOT/'batch_assets'/name,manifests/name)
    reference=read_table(ROOT/plan['e07_reference']['path'])
    write_probabilities(root/'validation/E07_val_probabilities.csv',reference,
                        {'model_id':'E07','frozen_source_sha256':plan['e07_reference']['sha256']})
    write_json(manifests/'environment.json',{'created_utc':utcnow(),'git_commit':commit(),
               'code_sha256':plan['artifacts'],'data_sha256':plan['data_sha256'],'fold_hashes':plan['fold_hashes'],
               'python':__import__('platform').python_version(),'platform':__import__('platform').platform(),
               'note':'GPU/CUDA runtime details are recorded per experiment in environment.json'})
    write_json(manifests/'labels_sha256.json',{name:digest(manifests/name) for name in ('train_labels.csv','val_labels.csv')})


def export_task(run, task, result):
    """Called only after independent best validation + OOF succeeded, before freeze."""
    import torch
    import torchvision
    run=Path(run); batch_root=run.parent.parent
    manifests=batch_root/'manifests'
    shutil.copy2(run/'config.yaml',run/'resolved_config.yaml')
    shutil.copy2(run/'last_training.pt',run/'checkpoint_resume.pth')
    kinds=['raw','ema'] if task['ema'] else ['raw']
    weights={}
    for kind in kinds:
        source=run/('best_model.pt' if kind=='raw' else 'best_ema_model.pt')
        destination=run/('best_model.pth' if kind=='raw' else 'ema_best_model.pth')
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        torch.save(checkpoint['model_state_dict'],destination)
        # Deserialize once and compare every tensor, without constructing a GPU model.
        restored=torch.load(destination,map_location='cpu',weights_only=True)
        if set(restored)!=set(checkpoint['model_state_dict']) or not all(torch.equal(restored[k],v.cpu()) for k,v in checkpoint['model_state_dict'].items()):
            raise ValueError('Exported inference state_dict differs')
        weights[kind]=digest(destination)
        destination.with_suffix('.sha256').write_text(weights[kind]+'  '+destination.name+'\n',encoding='ascii')
        for split,folder,prefix in [('val','validation_analysis',''),('oof','oof_analysis','')]:
            if split=='oof' and task['fold'] is None:
                continue
            table=read_table(run/(folder if kind=='raw' else 'ema_'+folder)/'study_predictions.csv')
            target=run/(('ema_' if kind=='ema' else '')+split+'_predictions.csv')
            provenance={'model_id':task['id'],'kind':kind,'checkpoint_sha256':weights[kind],
                        'fold':task['fold'],'fold_hashes':task.get('fold_hashes',{}),
                        'config_sha256':digest(run/'resolved_config.yaml')}
            write_probabilities(target,table,provenance)
            label_file=manifests/('val_labels.csv' if split=='val' else 'train_labels.csv')
            keys=label_keys(label_file, task['fold'] if split=='oof' else None)
            restored_table=read_probabilities(target,label_file,keys)
            if not np.array_equal(table['y'],restored_table['y']):
                raise ValueError('Frozen evaluation labels do not match independent inference labels')
        metrics={'best_epoch':json.loads((run/('best_metrics.json' if kind=='raw' else 'best_ema_metrics.json')).read_text())['epoch'],
                 'stop_epoch':result['status']['stop_epoch'],'kind':kind,**result['predictions'][kind],
                 'resources':result['resources'],'weights_bytes':destination.stat().st_size,
                 'checkpoint_sha256':weights[kind]}
        write_json(run/('metrics.json' if kind=='raw' else 'ema_metrics.json'),metrics)
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    columns=['epoch','train_loss','val_macro_auc','val_macro_f1','learning_rates','elapsed_seconds','best_epoch','stale','amp_skipped_steps']
    with (run/'train_history.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        for r in history:
            w.writerow({k: (r['val']['macro_auc'] if k=='val_macro_auc' else r['val']['macro_f1'] if k=='val_macro_f1' else
                             json.dumps(r[k]) if k=='learning_rates' else r[k]) for k in columns})
    (run/'train.log').write_text((run/'train_stdout.log').read_text(encoding='utf-8',errors='replace')+
          '\n--- STDERR (preserved separately; merged by stream, not timestamp) ---\n'+
          (run/'train_stderr.log').read_text(encoding='utf-8',errors='replace'),encoding='utf-8')
    rows=[]
    for kind,pred in result['predictions'].items():
        for split,groups in pred.items():
            for group,metrics in groups.items():
                rows.extend({'kind':kind,'split':split,'group':group,**r} for r in metrics['per_class'])
    with (run/'per_class_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    env=json.loads((run/'environment.json').read_text())
    env['torchvision']=torchvision.__version__
    write_json(run/'environment.json',env)
    train_rows=[]
    with (manifests/'image_folds.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames
        train_rows=[r for r in reader if task['fold'] is None or int(r['fold'])!=task['fold']]
    with (run/'training_image_index.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(train_rows)
    files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in run.iterdir() if p.is_file() and p.name not in ('run_manifest.json','DONE.json','batch_frozen_manifest.json')}
    write_json(run/'run_manifest.json',{'model_id':task['id'],'git_commit':commit(),'code_manifest':'../../manifests/environment.json',
        'config_sha256':digest(run/'resolved_config.yaml'),'fold_hashes':task.get('fold_hashes',{}),
        'training_image_index_sha256':digest(run/'training_image_index.csv'),'seed':task['seed'],
        'pretrained':result['pretrained'],'device':env,'end_utc':utcnow(),
        'start_utc':json.loads((run/'batch_timing.json').read_text()).get('start_utc') if (run/'batch_timing.json').exists() else None,
        'state':'complete','files':files,'ema':{'enabled':task['ema'],'decay':.999 if task['ema'] else None,
           'update':'first successful optimizer step copies; then decay each successful optimizer step',
           'bn':'ConvNeXt LayerNorm only; no BatchNorm; no recalibration'}})


def record_failure(root, task, record):
    run=Path(root)/'experiments'/task['id']; run.mkdir(parents=True,exist_ok=True)
    if (run/'DONE.json').exists():
        # A post-freeze backup failure must not mutate a successfully frozen model.
        out=Path(root)/'reports'/(task['id']+'_backup_failure.json'); write_json(out,record); return
    try:
        status=json.loads((run/'status.json').read_text())
    except (OSError,ValueError):
        status={}
    valid=False
    if (run/'last_training.pt').exists():
        try:
            import torch
            state=torch.load(run/'last_training.pt',map_location='cpu',weights_only=True)
            valid=all(k in state for k in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict','epoch','stale','python_random_state'))
        except Exception:
            valid=False
    write_json(run/'FAILED.json',{**record,'model_id':task['id'],'last_successful_epoch':state['epoch'] if valid else status.get('last_epoch'),
               'valid_resume_checkpoint':valid,'error_type':record.get('reason','Unknown').split(':')[0],'time_utc':utcnow()})
    (run/'error.log').write_text(record.get('reason','Unknown failure'),encoding='utf-8')
