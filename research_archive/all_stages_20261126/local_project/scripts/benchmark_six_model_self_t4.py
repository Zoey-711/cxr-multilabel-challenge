"""Private T4 speed reference on 100 frozen Val images; never official compliance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def image_files(directory: Path, expected: int) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(files) != expected or len({p.name for p in files}) != expected:
        raise ValueError(f'Expected exactly {expected} unique image files in {directory}')
    return files


def verify_sample(sample: Path, measured: list[Path], warmup: list[Path]) -> dict:
    data = json.loads((sample / 'manifest_sha256.json').read_text(encoding='utf-8'))
    if data.get('source_split') != 'official Val only' or data.get('timed_images') != 100 or data.get('warmup_images') != 5:
        raise ValueError('Expected the frozen Phase53 private Val timing sample')
    rows = list(csv.DictReader((sample/'timed_image_index.csv').open(encoding='utf-8-sig', newline='')))
    if len(rows) != 100 or {r['image'] for r in rows} != {p.name for p in measured}:
        raise ValueError('Image index differs from timed files')
    if any(int(r['study_image_count']) != 1 for r in rows):
        raise ValueError('This reference script expects complete single-image Studies')
    keys = {(r['Subject_id'], r['Study_id']) for r in rows}
    if len(keys) != 100 or any(not all(k) for k in keys):
        raise ValueError('Duplicate or missing Val Study identifiers')
    actual = {p.parent.name+'/'+p.name: p for p in measured+warmup}
    if set(actual) != set(data['files']) - {'timed_image_index.csv'}:
        raise ValueError('Val sample file inventory differs')
    index_info = data['files']['timed_image_index.csv']
    if (sample/'timed_image_index.csv').stat().st_size != index_info['bytes'] or sha256(sample/'timed_image_index.csv') != index_info['sha256']:
        raise ValueError('Val index hash differs')
    hashes = {}
    for relative, path in actual.items():
        info = data['files'][relative]
        digest = sha256(path)
        if path.stat().st_size != info['bytes'] or digest != info['sha256']:
            raise ValueError('Val image bytes differ: '+relative)
        hashes[relative] = digest
    if len(set(hashes.values())) != 105:
        raise ValueError('Warmup and timed images overlap')
    return {'source_archive_sha256': data['source_archive_sha256'], 'sample_files_sha256': sha256(sample/'manifest_sha256.json'),
            'images_verified': len(hashes), 'studies_verified': len(keys)}


def run(package: Path, sample: Path, output: Path) -> dict:
    import numpy as np
    import torch

    package, sample, output = package.resolve(), sample.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(package) or output.is_relative_to(sample):
        raise ValueError('Choose a new output directory outside model and image folders')
    timed = image_files(sample/'pictures_100', 100)
    warmup = image_files(sample/'pictures_5', 5)
    source = verify_sample(sample, timed, warmup)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Reference measurement requires the rented NVIDIA T4')
    config = json.loads((package/'config.json').read_text(encoding='utf-8'))
    if config.get('formula') != '0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3' or config.get('batch') != 1:
        raise ValueError('Model formula or batch setting differs')
    # The candidate predict module refuses wrong engine/runtime provenance and checks all asset hashes.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package))
    predict = importlib.import_module('predict')
    models = predict.load_models()
    if set(models.get('models', {})) != {'B01', 'B03', 'N01_EMA', 'S17_F3'} or set(models.get('trt_engines', {})) != {'B02','M02'}:
        raise RuntimeError('Six-model runtime did not load the expected four PyTorch and two TensorRT assets')
    for path in warmup:
        predict.predict_image(path, models)
        torch.cuda.synchronize()
    output.mkdir(parents=True)
    records = {p.name: [] for p in timed}
    with (output/'timings_300.csv').open('x', encoding='utf-8', newline='') as times, \
         (output/'image_predictions_300.csv').open('x', encoding='utf-8', newline='') as probs:
        time_writer, prob_writer = csv.writer(times), csv.writer(probs)
        time_writer.writerow(['round', 'image', 'elapsed_ms'])
        prob_writer.writerow(['round', 'image', *[f'Prob_{i}' for i in range(1,11)]])
        for round_no in range(1,4):
            for path in timed:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = np.asarray(predict.predict_image(path, models), dtype=np.float64)
                if result.shape != (10,) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
                    raise RuntimeError('Invalid model output')
                prob_writer.writerow([round_no, path.name, *[format(float(v), '.17g') for v in result]])
                probs.flush()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                time_writer.writerow([round_no, path.name, format(elapsed, '.9f')])
                times.flush()
                records[path.name].append(elapsed)
    per_image = [statistics.mean(records[p.name]) for p in timed]
    report = {
        'status': 'SELF_REFERENCE_ONLY', 'official_compliance': 'UNVERIFIED',
        'reason': 'Private selected Val images; rented environment may not be organizer image; no submitted Test CSV consistency test',
        'gpu': torch.cuda.get_device_name(0), 'batch': 1, 'warmup_images': 5, 'timed_images': 100,
        'rounds': 3, 'timing_records': sum(len(v) for v in records.values()),
        'mean_ms': statistics.mean(per_image), 'p50_ms': float(np.percentile(per_image, 50)),
        'p95_ms': float(np.percentile(per_image, 95)), 'max_ms': max(per_image),
        'timer_scope': 'one file read through six-model prediction, output CSV flush and CUDA synchronization',
        'models': ['B01','B02','B03','M02','N01_EMA','S17_F3'], 'precision': config.get('precision'),
        'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'tensorrt': importlib.import_module('tensorrt').__version__,
        'sample': source, 'source_package_manifest_sha256': sha256(package/'manifest_sha256.json'),
        'prediction_code_sha256': sha256(package/'phase65_source/predict.py'),
        'timings_sha256': sha256(output/'timings_300.csv'),
        'predictions_sha256': sha256(output/'image_predictions_300.csv'),
    }
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Path to compiled candidate/config.json')
    parser.add_argument('--sample', type=Path, required=True, help='Extracted phase53_val_t4_samples directory')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    print(json.dumps(run(args.config.parent, args.sample, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
