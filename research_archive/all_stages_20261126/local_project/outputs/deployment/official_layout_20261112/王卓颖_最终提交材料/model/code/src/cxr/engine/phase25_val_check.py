"""Apply frozen Phase25 pooling rules to existing official-Val image predictions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from cxr.data.archive import archive_path, read_split_rows
from cxr.engine.phase25_audit import aucs, pool, rows, sha, write_csv
from cxr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    out = root / config['output']
    metadata = read_split_rows(archive_path(load_config(root / config['source_config'])), 'val')
    meta = {(r['Subject_id'], r['Study_id'], r['Dicom_id']): r for r in metadata}
    by_study = defaultdict(list)
    for k in meta:
        by_study[k[:2]].append(k)
    keys = sorted(by_study)
    labels = {(r['Subject_id'],r['Study_id']):r for r in rows(root/'batch_assets/val_labels.csv')}
    assert set(keys)==set(labels) and len(keys)==859 and len(meta)==1422
    y = np.array([[int(labels[k][f'Target_{i}']) for i in range(1,11)] for k in keys])
    predictions = {}
    dates = {1:'20261019',2:'20261020',3:'20261021',4:'20261022',5:'20261023'}
    for branch in config['backbone_weights']:
        fold_predictions = []
        for fold in range(1,6):
            run = root/f'outputs/experiments/{branch.lower()}_f{fold}_{dates[fold]}'
            manifest = json.loads((run/'frozen_manifest.json').read_text(encoding='utf-8'))
            image_path = run/'validation_analysis/image_predictions.csv'
            study_path = run/'validation_analysis/study_predictions.csv'
            for path in (image_path, study_path):
                relative=path.relative_to(run).as_posix()
                assert sha(path)==manifest['files'][relative]['sha256']
            images={}
            for r in rows(image_path):
                k=(r['Subject_id'],r['Study_id'],r['Image_id'])
                assert k in meta and k not in images
                images[k]=np.array([float(r[f'probability_{i}']) for i in range(1,11)])
            assert set(images)==set(meta)
            pooled={rule:[] for rule in config['rules']}
            for k in keys:
                p=pool(np.stack([images[x] for x in by_study[k]]),
                       [meta[x]['ViewPosition'] for x in by_study[k]])
                for rule in pooled:
                    pooled[rule].append(p[rule])
            pooled={rule:np.stack(values) for rule,values in pooled.items()}
            reference={(r['Subject_id'],r['Study_id']):np.array([float(r[f'probability_{i}']) for i in range(1,11)])
                       for r in rows(study_path)}
            assert set(reference)==set(keys)
            assert np.max(np.abs(pooled['Mean']-np.stack([reference[k] for k in keys])))<=config['mean_reconstruction_atol']
            fold_predictions.append(pooled)
        predictions[branch]={rule:np.mean([p[rule] for p in fold_predictions],axis=0) for rule in config['rules']}
    predictions['E17']={rule:sum(w*predictions[b][rule] for b,w in config['backbone_weights'].items())
                        for rule in config['rules']}
    records=[]
    for branch,values in predictions.items():
        for rule,p in values.items():
            class_aucs=aucs(y,p)
            records.append({'model':branch,'rule':rule,'studies':len(keys),'macro_auc':float(class_aucs.mean()),
                            **{f'auc_{i+1}':float(v) for i,v in enumerate(class_aucs)}})
    output=out/'aggregation_val_check.csv'
    if output.exists():
        raise FileExistsError(output)
    write_csv(output,records)
    print(json.dumps([r for r in records if r['model']=='E17'],ensure_ascii=False))


if __name__=='__main__':
    main()
