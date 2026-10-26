"""Small file and timing helpers used by the isolated Phase39 entry point."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cxr.data.archive import image_member
from cxr.data.dataset import preprocess_image


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Empty CSV')
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_torch(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def latency(model, dataset, run, cfg):
    """Local reference; includes file decode through durable one-row CSV output."""
    folder = run / 'latency_inputs'
    folder.mkdir(exist_ok=True)
    rows = dataset.rows[:25]
    with zipfile.ZipFile(dataset.archive) as archive:
        for row in rows:
            path = folder / (row['Dicom_id'] + '.jpg')
            if not path.exists():
                path.write_bytes(archive.read(image_member(row, 'train')))

    def predict_one(row):
        with Image.open(folder / (row['Dicom_id'] + '.jpg')) as image:
            tensor = preprocess_image(image, 384, preprocessing='black_pad_v1').unsqueeze(0)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            prob = torch.sigmoid(model(tensor.to('cuda'))).float().cpu().tolist()[0]
        with (run / 'latency_last_prediction.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Subject_id', 'Study_id'] + [f'Prob_{i}' for i in range(1, 11)])
            writer.writerow([row['Subject_id'], row['Study_id']] + prob)

    model.eval()
    for row in rows[:5]:
        predict_one(row)
    durations = []
    for _ in range(3):
        for row in rows[5:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict_one(row)
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    write_json(run / 'latency_benchmark.json', dict(
        gpu=torch.cuda.get_device_name(), precision='FP16 autocast', batch_size=1,
        mean_ms=float(np.mean(durations)), p50_ms=float(np.percentile(durations, 50)),
        p95_ms=float(np.percentile(durations, 95)), max_ms=max(durations),
        warmup_images=5, timed_images_per_round=20, rounds=3,
        scope='JPEG open/decode, pad/normalize, transfer, forward, CPU probabilities, CSV write+close',
        status='T4_UNVERIFIED', official_timing_pass=False))


@contextlib.contextmanager
def lock(out):
    path = out / 'pipeline.lock'
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Phase39 pipeline lock exists; inspect its PID before recovery') from exc
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
