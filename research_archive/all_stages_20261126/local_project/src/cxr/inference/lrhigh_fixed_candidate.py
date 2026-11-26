"""Generate the explicitly requested E03/LR_HIGH EMA/S17-F3 fixed candidate."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.engine.metrics import study_multilabel_metrics
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.inference.versioned_submission import CSV_FIELDS, MATRIX_FIELDS, verify_candidate, write_csv
from cxr.models.factory import create_model


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git_version():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else 'UNAVAILABLE_NOT_A_GIT_CHECKOUT; use source SHA256'


def matrix(path, prefix='probability_'):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {(r['Subject_id'], r['Study_id']): tuple(float(r[f'{prefix}{i}']) for i in range(1, 11)) for r in rows}
    assert len(result) == len(rows), 'Duplicate Study'
    a = np.asarray(list(result.values()))
    assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    return result


def infer(model, archive, split, out):
    loader = make_dataloader(archive, split, batch_size=8, num_workers=0, input_size=384,
                             preprocessing='black_pad_v1')
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.float16):
                p = model(batch['image'].to('cuda')).sigmoid().float().cpu().tolist()
            predictions.extend(ImagePrediction(s, t, i, tuple(v)) for s, t, i, v in
                               zip(batch['subject_id'], batch['study_id'], batch['image_id'], p))
            if len(predictions) % 200 == 0:
                print(f'{split}: {len(predictions)}/{len(loader.dataset)}', flush=True)
    loader.dataset.close()
    torch.cuda.synchronize()
    metadata = read_split_rows(archive, split)
    assert {(p.subject_id, p.study_id, p.image_id) for p in predictions} == {
        (r['Subject_id'], r['Study_id'], r['Dicom_id']) for r in metadata}
    counts = Counter((r['Subject_id'], r['Study_id']) for r in metadata)
    studies = aggregate_mean_probabilities(predictions)
    assert all(s.image_count == counts[(s.subject_id, s.study_id)] for s in studies)
    write_csv(out/f'lr_high_{split}_image_predictions.csv', ('Subject_id','Study_id','Image_id',*MATRIX_FIELDS[2:]),
              [dict(Subject_id=p.subject_id, Study_id=p.study_id, Image_id=p.image_id,
                    **{f:repr(v) for f,v in zip(MATRIX_FIELDS[2:],p.probabilities)}) for p in predictions])
    write_csv(out/f'lr_high_{split}_predictions.csv', MATRIX_FIELDS,
              [dict(Subject_id=s.subject_id, Study_id=s.study_id,
                    **{f:repr(v) for f,v in zip(MATRIX_FIELDS[2:],s.probabilities)}) for s in studies])
    return predictions, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    sources = config['sources']
    for record in sources.values():
        assert sha(record['path']) == record['sha256'], f'Source SHA mismatch: {record["path"]}'
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'resolved_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = create_model('efficientnet_v2_s', pretrained=False)
    model.load_state_dict(torch.load(sources['checkpoint']['path'], map_location='cpu', weights_only=True), strict=True)
    model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    vp, vs = infer(model, config['val_archive'], 'val', out)
    metrics = study_multilabel_metrics(vp, study_targets_from_rows(read_split_rows(config['val_archive'], 'val')))
    current = matrix(out/'lr_high_val_predictions.csv')
    reference = matrix(sources['reference_val']['path'], 'Prob_')
    assert set(current) == set(reference) and len(current) == 859
    delta = np.abs(np.asarray([current[k] for k in sorted(current)])-np.asarray([reference[k] for k in sorted(current)]))
    report = {'macro_auc': metrics['macro_auc'], 'reference_auc': 0.8060120336873349,
              'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()), 'p99_abs':float(np.quantile(delta,.99)),
              'predeclared_identity_gate': 'max_abs<=0.002 and abs(AUC difference)<=0.001; not numerical equivalence',
              'per_class':metrics['per_class']}
    (out/'val_reproduction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    assert delta.max() <= .002 and abs(metrics['macro_auc']-.8060120336873349)<=.001, 'Val reproduction identity gate failed'
    tp, ts = infer(model, config['test_archive'], 'test', out)
    assert len(tp)==2218
    lr = matrix(out/'lr_high_test_predictions.csv')
    e03, s17 = matrix(sources['e03']['path']), matrix(sources['s17']['path'])
    assert set(lr)==set(e03)==set(s17) and len(lr)==1414
    fused = {k:tuple(.375*a+.375*b+.25*c for a,b,c in zip(e03[k],lr[k],s17[k])) for k in lr}
    write_csv(out/'test_predictions.csv', MATRIX_FIELDS,
              [dict(Subject_id=k[0],Study_id=k[1],**{f:repr(v) for f,v in zip(MATRIX_FIELDS[2:],fused[k])}) for k in sorted(fused)])
    with open(sources['template']['path'], encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames)==CSV_FIELDS
        old = list(reader)
    keys = [(r['Subject_id'], r['Study_id'], int(r['Predict_class'])) for r in old]
    assert len(keys)==len(set(keys))==14140 and set(keys)=={(*k,i) for k in fused for i in range(1,11)}
    write_csv(out/'submission.csv', CSV_FIELDS,
              [dict(Subject_id=s,Study_id=t,Predict_class=i,Probability=repr(fused[s,t][i-1])) for s,t,i in keys])
    validation = verify_candidate(out/'test_predictions.csv',out/'submission.csv',set(fused),policy='all_classes')
    # Independently re-read and recompute every CSV value from the three source matrices.
    with (out/'submission.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            k=(row['Subject_id'],row['Study_id']); i=int(row['Predict_class'])-1
            expected=.375*e03[k][i]+.375*lr[k][i]+.25*s17[k][i]
            assert float(row['Probability'])==expected
    validation['source_formula_recheck']='PASS'
    (out/'submission_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    manifest = {'formula':'.375 E03 + .375 ET_LR_HIGH_F01 EMA + .25 local S17-F3',
        'checkpoint_best_epoch':5,'sources':sources,'test_images':2218,'studies':1414,'rows':14140,
        'precision':'LR_HIGH FP16 AMP batch=8, black_pad_v1 384, image sigmoid then Study Mean; frozen E03 and S17 probabilities reused',
        'val_reproduction':report,'runtime':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'cuda':torch.version.cuda,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'val_seconds':vs,'test_seconds':ts},
        'deployment_status':'T4_UNVERIFIED; no claim of historical 78.81 prediction equivalence',
        'git_commit':git_version(),
        'code_sha256':sha(__file__),'upload_status':'NOT_UPLOADED',
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','submission':str(out/'submission.csv'),'sha256':sha(out/'submission.csv')}),flush=True)


if __name__=='__main__':
    main()
