"""Fail-closed artifact check for a real G/H training-step preflight.

These files live in a separate *_GH_PREFLIGHT run, never in a formal experiment.
"""
import csv
import json
from pathlib import Path

import torch

from cxr.batch.plan import digest
from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities

PROB_COLUMNS = ['Subject_id', 'Study_id', *(f'Prob_{i}' for i in range(1, 11))]


def _short_inference(model, archive, split, config, run_dir):
    rows = read_split_rows(archive, split)[:2]
    if len(rows) != 2:
        raise ValueError(f'{split} requires two real images for G/H GPU preflight')
    loader = make_dataloader(
        archive, split, rows=rows, batch_size=2, num_workers=0,
        preprocessing=config['data'].get('preprocessing', 'black_pad_v1'),
        input_size=int(config['data']['input_size']), persistent_workers=False)
    try:
        batch = next(iter(loader))
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            values = torch.sigmoid(model(batch['image'].to('cuda'))).float().cpu()
        if values.shape != (2, 10) or not bool(torch.isfinite(values).all()):
            raise ValueError(f'{split} preflight produced invalid ten-class probabilities')
        predictions = [ImagePrediction(batch['subject_id'][i], batch['study_id'][i],
                                      batch['image_id'][i], tuple(values[i].tolist()))
                       for i in range(2)]
        studies = aggregate_mean_probabilities(predictions)
        path = run_dir / f'{split}_probe.csv'
        with path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(PROB_COLUMNS)
            for item in studies:
                writer.writerow([item.subject_id, item.study_id, *[format(p, '.17g') for p in item.probabilities]])
        return {'real_images': 2, 'studies': len(studies), 'sha256': digest(path)}
    finally:
        loader.dataset.close()


def complete_gh_preflight(*, run_dir, config_path, config, model, optimizer, scheduler,
                          scaler, train_loader, val_loader, archive, optimizer_steps,
                          amp_skipped_steps, gradient_norm):
    """Round-trip the real training state, then infer real Val and unlabeled Test."""
    from cxr.utils.training_resume import capture_rng_state, restore_rng_state

    run_dir = Path(run_dir)
    if optimizer_steps != 1 or amp_skipped_steps or not bool(torch.isfinite(gradient_norm)):
        raise ValueError('G/H real training step or AMP gradient preflight failed')
    state = {'epoch': 1, 'best_auc': -1.0, 'stale': 0,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(),
             'scaler_state_dict': scaler.state_dict(),
             **capture_rng_state(train_loader, val_loader)}
    checkpoint = run_dir / 'probe_resume.pth'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(state, checkpoint)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()
              if torch.is_tensor(value)}
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'scaler_state_dict', 'epoch', 'best_auc', 'stale'}
    if not required.issubset(restored):
        raise ValueError('G/H probe resume checkpoint is incomplete')
    # Make restoration observable rather than loading into an unchanged object.
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    optimizer.param_groups[0]['lr'] = 0.0
    scheduler.last_epoch = -999
    torch.rand(1, device='cuda')
    model.load_state_dict(restored['model_state_dict'])
    optimizer.load_state_dict(restored['optimizer_state_dict'])
    scheduler.load_state_dict(restored['scheduler_state_dict'])
    scaler.load_state_dict(restored['scaler_state_dict'])
    restore_rng_state(restored, train_loader, val_loader)
    if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
           if torch.is_tensor(value)):
        raise ValueError('G/H model state changed after checkpoint restoration')
    if len(optimizer.state) == 0 or scheduler.state_dict() != restored['scheduler_state_dict']:
        raise ValueError('G/H optimizer or scheduler did not restore')
    if optimizer.param_groups[0]['lr'] == 0.0:
        raise ValueError('G/H optimizer learning rate did not restore')
    if scaler.state_dict() != restored['scaler_state_dict']:
        raise ValueError('G/H AMP scaler did not restore')
    report = {'status': 'passed', 'config_sha256': digest(config_path),
              'checkpoint_sha256': digest(checkpoint), 'optimizer_steps': optimizer_steps,
              'amp_skipped_steps': amp_skipped_steps, 'checkpoint_restored': True,
              'optimizer_restored': True, 'scheduler_restored': True,
              'scaler_restored': True, 'rng_restored': True,
              'val': _short_inference(model, archive, 'val', config, run_dir),
              'test': _short_inference(model, archive, 'test', config, run_dir)}
    (run_dir / 'probe_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    verify_gh_preflight(run_dir, report['config_sha256'])
    return report


def verify_gh_preflight(run_dir, config_sha256):
    """Do not unlock a formal G/H task on a partial or stale smoke run."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / 'probe_report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'passed' or report.get('config_sha256') != config_sha256 or
            report.get('optimizer_steps') != 1 or report.get('amp_skipped_steps') != 0 or
            any(report.get(key) is not True for key in ('checkpoint_restored',
                'optimizer_restored', 'scheduler_restored', 'scaler_restored', 'rng_restored'))):
        raise ValueError('Incomplete or stale G/H checkpoint preflight')
    if digest(run_dir / 'probe_resume.pth') != report.get('checkpoint_sha256'):
        raise ValueError('G/H resume checkpoint hash changed')
    for split in ('val', 'test'):
        info = report.get(split, {})
        path = run_dir / f'{split}_probe.csv'
        if info.get('real_images') != 2 or digest(path) != info.get('sha256'):
            raise ValueError(f'{split} G/H preflight provenance changed')
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PROB_COLUMNS:
                raise ValueError(f'{split} G/H preflight class order changed')
            rows = list(reader)
        if len(rows) != info.get('studies') or not 1 <= len(rows) <= 2:
            raise ValueError(f'{split} G/H preflight Study count changed')
        keys = [(row['Subject_id'], row['Study_id']) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f'{split} G/H preflight duplicate Study')
        for row in rows:
            for column in PROB_COLUMNS[2:]:
                value = float(row[column])
                if not 0 <= value <= 1:
                    raise ValueError(f'{split} G/H preflight probability invalid')
    return report
