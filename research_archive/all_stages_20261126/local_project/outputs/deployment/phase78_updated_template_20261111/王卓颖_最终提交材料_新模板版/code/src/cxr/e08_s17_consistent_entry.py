"""Locked six-model inference and strict historical-submission audit.

No precision/weight/threshold search. All inference remains batch=1 FP16.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cxr.data.archive import read_split_rows, image_member
from cxr.e08_s17_acceleration import Entry, gray_first, load_models as load_frozen_models
from cxr.prediction_consistency import (
    aggregate_complete, compare_studies, read_index, read_submission,
    probabilities, sha256, validate_index, write_csv,
)

PRECISION = 'PyTorch FP16 autocast; FP32 checkpoints; batch=1; CUDA Graph'


def checked_config(config_path):
    path = Path(config_path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    lock = json.loads(Path(cfg['file_lock']).read_text(encoding='utf-8'))
    locked = set()
    for record in lock['files']:
        source = Path(record['path']).resolve()
        if not source.is_file() or source.stat().st_size != record['bytes'] or sha256(source) != record['sha256']:
            raise ValueError(f'Frozen dependency changed/missing: {source}')
        locked.add(source)
    required = {path, Path(__file__).resolve(), Path(cfg['reference_submission']).resolve(),
                Path(cfg['full_image_index']).resolve(), Path(cfg['s17_checkpoint']).resolve()}
    if not required <= locked:
        raise ValueError('Configuration/source/reference/metadata/checkpoint missing from lock')
    if cfg['reference_submission_sha256'] != sha256(cfg['reference_submission']):
        raise ValueError('Historical 78.81 submission identity mismatch')
    if cfg['formula'] != '0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3':
        raise ValueError('Unexpected fusion formula')
    return cfg


class Predictor:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        models = load_frozen_models(cfg)
        if any(module.training for model in models.values() for module in model.modules()):
            raise ValueError('All model modules must already be eval')
        self.entry = Entry(models, 'cuda_graph')
        self.cfg = cfg


def load_models(config_path):
    return Predictor(checked_config(config_path))


def predict_image(source, state):
    """Same computation for archive audit and timed filesystem input."""
    if isinstance(source, Image.Image):
        return probabilities(state.entry.from_tensor(gray_first(source)))
    return probabilities(state.entry.predict_image(Path(source)))


def aggregate_studies(rows, values, keys=None):
    return aggregate_complete(rows, values, keys)


def environment():
    return dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                precision=PRECISION, seed=42, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def finalize_manifest(output):
    save_json(output/'manifest_sha256.json', {str(p.relative_to(output)): sha256(p)
              for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest_sha256.json'})


def audit_test(config_path, output):
    """Forward-only Test verification against the *existing* submitted CSV.

    No output is a new official submission; failures never alter probabilities.
    """
    cfg = checked_config(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = read_index(cfg['full_image_index'])
    images, groups = validate_index(rows)
    reference = read_submission(cfg['reference_submission'])
    if len(images) != 2218 or len(groups) != 1414 or set(groups) != set(reference):
        raise ValueError('Full Test mapping does not match 2218 images / 1414 reference Studies')
    metadata = read_split_rows(cfg['archive'], 'test')
    mapped = [(r['Dicom_id']+'.jpg', r['Subject_id'], r['Study_id']) for r in metadata]
    expected = [(r['image'], r['Subject_id'], r['Study_id']) for r in rows]
    if mapped != expected:
        raise ValueError('Locked metadata order differs from real archive')
    state = load_models(config_path)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values, inventory = {}, []
    image_rows = []
    repeat_error = 0.
    # Fixed, image-index-based repeat checks, selected before seeing probabilities.
    repeat_positions = {0, 221, 442, 663, 884, 1105, 1326, 1547, 1768, 1989}
    with zipfile.ZipFile(cfg['archive']) as archive:
        for i, row in enumerate(metadata):
            name = row['Dicom_id']+'.jpg'
            raw = archive.read(image_member(row, 'test'))
            with Image.open(io.BytesIO(raw)) as image:
                result = predict_image(image, state)
                if i in repeat_positions:
                    again = predict_image(image, state)
                    repeat_error = max(repeat_error, float(np.max(np.abs(result-again))))
            values[name] = result
            import hashlib
            inventory.append(dict(image=name, Subject_id=row['Subject_id'], Study_id=row['Study_id'],
                                  sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            image_rows.append(dict(image=name, Subject_id=row['Subject_id'], Study_id=row['Study_id'],
                                   **{f'Prob_{j}':format(float(x), '.17g') for j,x in enumerate(result,1)}))
            if (i+1)%200 == 0:
                print(f'TEST_AUDIT {i+1}/2218', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    actual = aggregate_studies(rows, values)
    comparison, details = compare_studies(actual, reference)
    write_csv(out/'test_image_predictions.csv', image_rows)
    write_csv(out/'test_study_predictions.csv', [dict(Subject_id=k[0], Study_id=k[1],
              **{f'Prob_{j}':format(float(x), '.17g') for j,x in enumerate(v,1)}) for k,v in sorted(actual.items())])
    write_csv(out/'historical_comparison.csv', details)
    save_json(out/'test_image_inventory.json', inventory)
    report = dict(status='PASS' if comparison['status']=='PASS' and repeat_error==0 else 'FAIL',
                  historical_submission=comparison, reference_sha256=sha256(cfg['reference_submission']),
                  reference_path=cfg['reference_submission'], config_sha256=sha256(config_path),
                  lock_sha256=sha256(cfg['file_lock']), environment=environment(),
                  internal_repeatability=dict(status='EXACT_MATCH' if repeat_error==0 else 'FAIL',
                                              checked_images=len(repeat_positions), max_abs=repeat_error),
                  images=len(images), studies=len(actual), wall_seconds=elapsed,
                  wall_scope='Archive audit including extra repeats; NOT official latency measurement',
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  t4_status='T4_UNVERIFIED', official_submission_generated=False,
                  conclusion='Historical equivalence remains separate from internal speed optimization equivalence')
    save_json(out/'audit_report.json', report)
    finalize_manifest(out)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['status']=='PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(audit_test(args.config, args.output))


if __name__ == '__main__':
    main()
