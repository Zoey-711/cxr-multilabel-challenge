"""Generate one fixed E08/H17 candidate from hash-verified Study probabilities."""
import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cxr.data.archive import read_split_rows
from cxr.engine.metrics import binary_auc
from cxr.inference.aggregate import StudyPrediction
from cxr.inference.versioned_submission import (
    CSV_FIELDS, MATRIX_FIELDS, build_candidate_rows, sha256, verify_candidate, write_csv,
)


def matrix(path, keys):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    prefix = 'probability_' if 'probability_1' in fields else 'Prob_'
    lookup = {}
    for row in rows:
        k = row['Subject_id'], row['Study_id']
        v = np.array([float(row[f'{prefix}{i}']) for i in range(1, 11)])
        if k in lookup or not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
            raise ValueError(f'Invalid matrix: {path}')
        lookup[k] = v
    if set(lookup) != set(keys):
        raise ValueError(f'Study coverage mismatch: {path}')
    return np.stack([lookup[k] for k in keys])


def submission(path):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames) != CSV_FIELDS:
            raise ValueError('Invalid source submission header')
        rows = list(reader)
    d = {}
    for row in rows:
        k = row['Subject_id'], row['Study_id'], int(row['Predict_class'])
        p = float(row['Probability'])
        if k in d or not np.isfinite(p) or not 0 <= p <= 1 or not 1 <= k[2] <= 10:
            raise ValueError('Invalid source submission')
        d[k] = p
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['e08_weight'] != 0.75 or cfg['h17_weight'] != 0.25:
        raise ValueError('This candidate is frozen at 0.75 E08 + 0.25 H17')
    out = root / cfg['output_dir']
    if out.exists():
        raise FileExistsError('Candidate directory exists; refusing overwrite')
    e = root / 'outputs/submissions/E08_EMA_20261020'
    h = root / 'outputs/submissions/H17_02_BATCH1_20261024'
    package = root / 'outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate'
    hp = package / 'results'
    review = root / 'outputs/analysis/model_options_review_20261104'
    em = json.loads((e/'manifest.json').read_text(encoding='utf-8'))
    hm = json.loads((h/'manifest.json').read_text(encoding='utf-8'))
    pm = json.loads((hp/'prediction_manifest.json').read_text(encoding='utf-8'))
    sources = {}

    def check(path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'SHA mismatch: {path}')
        sources[str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}

    for path in (e/'manifest.json', h/'manifest.json', hp/'prediction_manifest.json', args.config, Path(__file__)):
        check(path)
    for name in ('test_predictions.csv', 'submission.csv', 'val_predictions.csv'):
        check(e/name, em['files'][name]['sha256'])
    check(h/'submission.csv', hm['submission_sha256'])
    check(hp/'submission.csv', hm['submission_sha256'])
    check(hp/'test_predictions.csv', pm['test_predictions_sha256'])
    if pm['submission_sha256'] != hm['submission_sha256'] or pm['source_checkpoint_sha256'] != hm['checkpoint_sha256']:
        raise ValueError('H17 package/source submission mismatch')
    archive = Path(cfg['archive'])
    rows = read_split_rows(archive, 'test')
    counts = Counter((r['Subject_id'], r['Study_id']) for r in rows)
    keys = sorted(counts)
    if len(rows) != 2218 or len(keys) != 1414 or any('Predict_class' in r for r in rows):
        raise ValueError('Invalid unlabeled official Test metadata')
    if len({(r['Subject_id'], r['Study_id'], r['Dicom_id']) for r in rows}) != 2218:
        raise ValueError('Duplicate Test images')
    ep, hprob = matrix(e/'test_predictions.csv', keys), matrix(hp/'test_predictions.csv', keys)
    es, hs = submission(e/'submission.csv'), submission(h/'submission.csv')
    expected = {(s, st, j) for s, st in keys for j in range(1, 11)}
    if set(es) != expected or set(hs) != expected:
        raise ValueError('Source submission IDs incomplete')
    for i, k in enumerate(keys):
        for j in range(10):
            if es[(*k, j+1)] != ep[i,j] or hs[(*k, j+1)] != hprob[i,j]:
                raise ValueError('Source submission and Study probability mismatch')
    fused = 0.75*ep + 0.25*hprob
    provenance = json.loads((review/'provenance.json').read_text(encoding='utf-8'))
    for rel, digest in provenance['sources'].items():
        check(root/rel, digest)
    label_path = root / 'outputs/experiments/n01_efficientnet_v2_s_384_amp_retry_20261018/ema_validation_analysis/study_predictions.csv'
    with label_path.open(encoding='utf-8-sig', newline='') as f:
        label_rows = list(csv.DictReader(f))
    vk = [(r['Subject_id'], r['Study_id']) for r in label_rows]
    if len(vk)!=859 or len(set(vk))!=859 or len({s for s, _ in vk})!=228:
        raise ValueError('Invalid Val metadata')
    y = np.array([[int(r[f'target_{j}']) for j in range(1,11)] for r in label_rows])
    vp = 0.75*matrix(e/'val_predictions.csv',vk)+0.25*matrix(root/'outputs/analysis/phase49_h17_batch1_acceleration/val_torch_frozen.csv',vk)
    aucs = [binary_auc(y[:,j],vp[:,j]) for j in range(10)]
    auc = float(np.mean(aucs))
    if abs(auc-0.8124268633828053)>1e-12:
        raise ValueError('Frozen candidate Val AUC mismatch')
    models = []
    for name, info in em['models'].items():
        run = root/'outputs/experiments'/info['run_id']
        cp, config = run/info['checkpoint'], run/'config.yaml'
        check(cp, info['checkpoint_sha256']); check(config,info['config_sha256'])
        models.append({'model':name,'weight':0.75*em['fusion_weights'][name],
                       'checkpoint':str(cp),'sha256':sha256(cp),'checkpoint_bytes':cp.stat().st_size,
                       'config':str(config),'config_bytes':config.stat().st_size})
    cp, config = package/'model/H17_02.pth', package/'config/deployment_config.json'
    check(cp,hm['checkpoint_sha256']); check(config,pm['config_sha256'])
    models.append({'model':'H17_BATCH1','weight':0.25,'checkpoint':str(cp),
                   'sha256':sha256(cp),'checkpoint_bytes':cp.stat().st_size,
                   'config':str(config),'config_bytes':config.stat().st_size})
    size = sum(m['checkpoint_bytes']+m['config_bytes'] for m in models)
    out.mkdir(parents=True)
    predictions = [StudyPrediction(k[0],k[1],counts[k],tuple(fused[i])) for i,k in enumerate(keys)]
    mat, candidate, _ = build_candidate_rows(predictions,policy='all_classes')
    write_csv(out/'test_predictions.csv',MATRIX_FIELDS,mat)
    write_csv(out/'submission.csv',CSV_FIELDS,candidate)
    valrows = [{'Subject_id':k[0],'Study_id':k[1],**{f'probability_{j+1}':repr(float(vp[i,j])) for j in range(10)}} for i,k in enumerate(vk)]
    write_csv(out/'val_predictions.csv',MATRIX_FIELDS,valrows)
    report = verify_candidate(out/'test_predictions.csv',out/'submission.csv',set(keys),policy='all_classes')
    actual = submission(out/'submission.csv')
    error = max(abs(actual[k]-(3*es[k]+hs[k])/4) for k in expected)
    if error>1e-12 or list(actual)!=list(es) or len(actual)!=14140:
        raise ValueError('Independent source CSV formula/order verification failed')
    with zipfile.ZipFile(archive) as z:
        meta_sha = hashlib.sha256(z.read('cxr_first/test.csv')).hexdigest()
    report.update({'status':'FORMAT_PASS','independent_formula_max_abs_error':error,
                   'keys_and_order_match_frozen_e08':True,'val_macro_auc':auc,'val_per_class_auc':aucs,
                   'test_auc':None,'model_and_config_bytes':size,'size_status':'PASS' if size<=500000000 else 'FAIL',
                   'latency_status':'SIX_MODEL_T4_UNVERIFIED','deployment_status':'NEW_COMBINED_ENTRY_NOT_VALIDATED',
                   'official_score':None,'upload_status':'NOT_UPLOADED'})
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    manifest = {'candidate_id':out.name,'generated_utc':datetime.now(timezone.utc).isoformat(),
                'formula':'0.75*E08_EMA_20261020 + 0.25*H17_02_BATCH1_20261024',
                'models':models,'sources':sources,'source_git_commit':git.stdout.strip() if git.returncode==0 else None,
                'code_identity':'source SHA is authoritative for uncommitted generator',
                'official_test_metadata_sha256':meta_sha,'test_images':2218,'test_studies':1414,
                'test_labels_used':False,'selection_scope':'EXPLORATORY repeated official Val; no OOF claim',
                'reference_bootstrap':{'seed':42,'replicates':2000,'delta_vs_e08':0.0007582172511235719,
                                       'ci95':[-0.0011224830050283242,0.002585169738885553]},
                'deployment':'User reports prior E08 T4 pass; not evidence for this six-model candidate.'}
    for name, obj in [('submission_validation.json',report),('manifest.json',manifest),('resolved_config.json',cfg)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'generation.log').write_text('CPU only. Source SHA and source CSV/matrix agreement PASS. Official unlabeled Test keys PASS. Val reproduction PASS. Independent CSV formula PASS. No upload.\n',encoding='utf-8')
    hashes = {p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in out.iterdir() if p.is_file()}
    (out/'sha256_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
