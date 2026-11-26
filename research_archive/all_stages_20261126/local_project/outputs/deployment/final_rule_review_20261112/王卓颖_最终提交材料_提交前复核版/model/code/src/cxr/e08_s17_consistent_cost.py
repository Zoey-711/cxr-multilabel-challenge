"""Serial single-image benchmark, sharing predict and Study Mean with predict.py."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from cxr.e08_s17_consistent_entry import checked_config, environment, finalize_manifest, save_json
from cxr.prediction_consistency import (compare_rounds, probabilities, read_index,
                                        read_submission, sha256, validate_index)


def files(folder, count):
    paths = sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(paths) != count or len({p.name for p in paths}) != count:
        raise ValueError(f'Expected {count} images with unique names in {folder}')
    return paths


def run(predict_module):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--pictures-100', required=True, type=Path)
    parser.add_argument('--pictures-5', required=True, type=Path)
    parser.add_argument('--submission', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = checked_config(args.config)
    if sha256(args.submission) != cfg['reference_submission_sha256']:
        raise ValueError('Cannot substitute a newly generated CSV for the frozen 78.81 reference')
    reference = read_submission(args.submission)
    rows = read_index(cfg['full_image_index'])
    image_index, groups = validate_index(rows)
    measured, warm = files(args.pictures_100,100), files(args.pictures_5,5)
    measured_names = [p.name for p in measured]
    if not set(measured_names) <= set(image_index):
        raise ValueError('Timed images absent from locked full Test metadata')
    keys = {(image_index[n]['Subject_id'], image_index[n]['Study_id']) for n in measured_names}
    needed = {n for k in keys for n in groups[k]}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if needed != set(measured_names):
        save_json(out/'report.json', dict(status='SUBMISSION_CONSISTENCY_UNVERIFIED',
                  reason='Timed sample contains partial Studies; image probabilities cannot be compared directly with Study Mean',
                  missing_companion_images=sorted(needed-set(measured_names)), ready_for_official=False))
        finalize_manifest(out)
        return 2
    # Hash all inputs before loading models/timing. This does not decode or predict.
    hashes = {p:sha256(p) for p in measured+warm}
    if {hashes[p] for p in measured} & {hashes[p] for p in warm}:
        raise ValueError('Warmup images must be independent, including copies under different names')
    full_report = Path(cfg['full_test_audit'])
    if not full_report.is_file():
        raise ValueError('Full Test provenance audit required before historical-submission benchmark')
    audit = json.loads(full_report.read_text(encoding='utf-8'))
    audit_manifest = json.loads((full_report.parent/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name in ('audit_report.json','test_image_inventory.json'):
        if sha256(full_report.parent/name) != audit_manifest[name]:
            raise ValueError(f'Full Test audit artifact changed: {name}')
    if audit['config_sha256'] != sha256(args.config) or audit['lock_sha256'] != sha256(cfg['file_lock']):
        raise ValueError('Full Test audit used different code/config/weights')
    inventory = {r['image']:r for r in json.loads((full_report.parent/'test_image_inventory.json').read_text(encoding='utf-8'))}
    for p in measured:
        if hashes[p] != inventory[p.name]['sha256']:
            raise ValueError(f'Timed image bytes differ from original Test: {p.name}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    state = predict_module.load_models(args.config)
    for p in warm:
        predict_module.predict_image(p,state)
        torch.cuda.synchronize()
    rounds, timings = [], {n:[] for n in measured_names}
    # This clock also includes updating Study Mean and writing final Study rows
    # on the last image of each complete Study. No cached outputs are reused.
    with (out/'timings_300.csv').open('x',encoding='utf-8',newline='') as tf, \
         (out/'image_predictions_300.csv').open('x',encoding='utf-8',newline='') as pf, \
         (out/'study_predictions_3_rounds.csv').open('x',encoding='utf-8',newline='') as sf:
        tw,pw,sw=csv.writer(tf),csv.writer(pf),csv.writer(sf)
        tw.writerow(['round','image','elapsed_ms'])
        pw.writerow(['round','image',*[f'Prob_{i}' for i in range(1,11)]])
        sw.writerow(['round','Subject_id','Study_id',*[f'Prob_{i}' for i in range(1,11)]])
        for round_id in range(1,4):
            values, pending = {}, {}
            for path in measured:
                torch.cuda.synchronize()
                start=time.perf_counter()
                value=probabilities(predict_module.predict_image(path,state))
                values[path.name]=value
                pw.writerow([round_id,path.name,*[format(float(x),'.17g') for x in value]])
                pf.flush()
                meta=image_index[path.name];key=(meta['Subject_id'],meta['Study_id'])
                pending.setdefault(key,{})[path.name]=value
                if len(pending[key]) == len(groups[key]):
                    mean=predict_module.aggregate_studies(rows,pending[key],{key})[key]
                    sw.writerow([round_id,*key,*[format(float(x),'.17g') for x in mean]])
                    sf.flush()
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                timings[path.name].append(elapsed)
                tw.writerow([round_id,path.name,format(elapsed,'.9f')]);tf.flush()
            rounds.append(values)
    match=compare_rounds(rows,rounds,reference,measured_names)
    means=[float(np.mean(timings[n])) for n in measured_names]
    report=dict(status=match['status'], comparison=match,
                full_test_audit_status=audit['status'], mean_ms=float(np.mean(means)),
                p50_ms=float(np.percentile(means,50)),p95_ms=float(np.percentile(means,95)),
                max_ms=max(means),records=300,batch=1,warmup=5,rounds=3,
                timer_scope='file read, decode, preprocessing, all six forwards, fusion, image write/flush, Study Mean/final write at Study completion, CUDA sync',
                reference_sha256=sha256(args.submission),environment=environment(),
                t4_status='T4_MEASURED' if 'T4' in torch.cuda.get_device_name() else 'T4_UNVERIFIED',
                ready_for_official=False,official_status='PENDING_ORGANIZER_RETEST')
    if audit['status'] != 'PASS':
        report['status']='BLOCKED_FULL_TEST_INCONSISTENCY'
    save_json(out/'report.json',report)
    finalize_manifest(out)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['status']=='PASS' else 2
