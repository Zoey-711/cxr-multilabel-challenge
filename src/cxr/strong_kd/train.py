"""Independent paired students. Fixed four epochs; Val never selects an epoch."""
import itertools
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from cxr.batch.artifacts import read_probabilities
from cxr.data.archive import read_split_rows
from cxr.data.dataset import make_dataloader, study_targets_from_rows
from cxr.models.factory import create_model
from cxr.phase39_run import optimizer_scheduler, infer
from cxr.phase39_support import seed_all, write_csv, write_json, save_torch
from cxr.utils.training_resume import capture_rng_state, restore_rng_state
from cxr.strong_kd.protocol import training_config, digest, PRETRAINED_SHA
from cxr.strong_kd.loss import step_group


def verify_image_mapping(actual, frozen):
    keys = lambda rr: [(r['Subject_id'], r['Study_id'], r['Dicom_id']) for r in rr]
    found, expected = keys(actual), keys(frozen)
    if len(set(found)) != len(found) or len(set(expected)) != len(expected) or set(found) != set(expected):
        raise ValueError('Training image-to-Study mapping differs from frozen index')


def data(server, task):
    root, source = Path(server['output_root']), Path(server['teacher_root'])
    train = read_split_rows(Path(server['data_root']), 'train')
    val = read_split_rows(Path(server['data_root']), 'val')
    counts = Counter((r['Subject_id'], r['Study_id']) for r in train)
    table = read_probabilities(root/'teachers'/f'{task}.csv', source/'manifests/train_labels.csv', set(counts))
    if len(train) != 30344 or len(counts) != 18513 or len({k[0] for k in counts}) != 11055:
        raise ValueError('Training data coverage differs from frozen source')
    from cxr.strong_kd.teachers import rows
    frozen_images = rows(source/'manifests/image_folds.csv')
    verify_image_mapping(train, frozen_images)
    actual = {(s.subject_id, s.study_id): s.labels for s in study_targets_from_rows(train)}
    if any(not np.array_equal(actual[k], y) or counts[k] != n for k, y, n in zip(table['keys'], table['y'], table['counts'])):
        raise ValueError('Training labels/image counts differ from teacher source')
    vtruth = {(s.subject_id, s.study_id): s.labels for s in study_targets_from_rows(val)}
    fixed_val = {(r['Subject_id'], r['Study_id']): tuple(int(r[f'Target_{j}']) for j in range(1,11))
                 for r in rows(source/'manifests/val_labels.csv')}
    if vtruth != fixed_val or len(val) != 1422 or len(vtruth) != 859:
        raise ValueError('Official Val mapping differs')
    if {k[0] for k in counts} & {k[0] for k in vtruth}:
        raise ValueError('Train/Val patients overlap')
    soft = dict(zip(table['keys'], table['p']))
    weights = {k: 1./n for k, n in counts.items()}
    return train, val, soft, weights


def loaders(server, train, val, soft, weights):
    aug = dict(enabled=True, name='conservative_cxr_v1', affine_probability=.5,
               rotation_degrees=3., translation_fraction=.02, photometric_probability=.5,
               brightness_contrast_fraction=.05)
    tr = make_dataloader(Path(server['data_root']), 'train', rows=train,
        batch_size=8, num_workers=0, shuffle=True, seed=42, input_size=384,
        preprocessing='black_pad_v1', augmentation=aug,
        study_soft_targets=soft, study_soft_weights=weights)
    ev = make_dataloader(Path(server['data_root']), 'val', rows=val,
        batch_size=8, num_workers=0, shuffle=False, seed=43, input_size=384,
        preprocessing='black_pad_v1')
    return tr, ev


def build(cfg, device, steps, pretrained=True):
    seed_all(cfg['seed'])
    model = create_model('efficientnet_v2_s', pretrained=pretrained, num_classes=10).to(device)
    ema = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
    opt, sch = optimizer_scheduler(cfg, model, steps)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type == 'cuda')
    return model, ema, opt, sch, scaler


def finite_state(value):
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite_state(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def checkpoint(path, cfg, identity, epoch, history, parts, tr, ev):
    model, ema, opt, sch, scaler = parts
    state = dict(config=cfg, identity=identity, epoch=epoch, history=history,
        model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
        scheduler=sch.state_dict(), scaler=scaler.state_dict(), **capture_rng_state(tr, ev))
    if not finite_state(state): raise FloatingPointError('Refusing nonfinite recovery checkpoint')
    save_torch(path, state)


def restore(path, cfg, identity, parts, tr, ev):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['config'] != cfg or state['identity'] != identity or not finite_state(state):
        raise ValueError('Recovery identity differs or checkpoint is nonfinite')
    for obj, name in zip(parts, ('model', 'ema', 'optimizer', 'scheduler', 'scaler')):
        obj.load_state_dict(state[name])
    parts[2].zero_grad(set_to_none=True)
    restore_rng_state(state, tr, ev)
    return state['epoch'], state['history']


def smoke(server, task, folder, device):
    cfg = training_config(task)
    train, val, soft, weights = data(server, task)
    # Keep the true teacher map but restrict it to the selected real studies.
    train, val = train[:16], val[:8]
    keys = {(r['Subject_id'], r['Study_id']) for r in train}
    tr, ev = loaders(server, train, val, {k: soft[k] for k in keys}, {k: weights[k] for k in keys})
    os.environ['TORCH_HOME'] = server['pretrained_root']
    if device.type=='cuda' and digest(Path(server['pretrained_root'])/'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth')!=PRETRAINED_SHA:
        raise ValueError('ImageNet initialization hash changed')
    parts = build(cfg, device, 1, pretrained=device.type=='cuda')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    parts[0].train()
    diagnostic = {}
    result = step_group(parts[0], parts[1], list(itertools.islice(iter(tr), 2)), *parts[2:], device, diagnostic)
    if result['skipped'] or int(parts[1].n_averaged) != 1:
        raise ValueError('Smoke must perform a real optimizer/EMA update')
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint(folder/'resume.pt', cfg, 'smoke', 0, [], parts, tr, ev)
    del parts
    parts = build(cfg, device, 1, pretrained=False)
    restore(folder/'resume.pt', cfg, 'smoke', parts, tr, ev)
    resumed = step_group(parts[0], parts[1], list(itertools.islice(iter(tr), 2)), *parts[2:], device, {})
    if resumed['skipped'] or int(parts[1].n_averaged) != 2:
        raise ValueError('Restored optimizer/EMA update failed')
    parts[1].eval()
    with torch.inference_mode():
        logits = parts[1](next(iter(ev))['image'].to(device))
    if not bool(torch.isfinite(logits).all()): raise ValueError('Validation smoke nonfinite')
    return dict(state='PASS', loss=result, restored_loss=resumed,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0.,
        optimizer_updates=2, validation='PASS', device=str(device))


def run_task(server, task, identity, device=None):
    device = torch.device('cuda') if device is None else device
    cfg = training_config(task)
    folder = Path(server['output_root'])/'experiments'/task
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder/'resolved_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Frozen student config changed')
    write_json(config_path, cfg)
    train, val, soft, weights = data(server, task)
    tr, ev = loaders(server, train, val, soft, weights)
    os.environ['TORCH_HOME'] = server['pretrained_root']
    if device.type=='cuda' and digest(Path(server['pretrained_root'])/'hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth')!=PRETRAINED_SHA:
        raise ValueError('ImageNet initialization hash changed')
    steps = math.ceil(len(tr)/2)
    parts = build(cfg, device, steps, pretrained=device.type=='cuda')
    model, ema, opt, sch, scaler = parts
    history, epoch = [], 0
    resume = folder/'last_training.pt'
    if resume.exists(): epoch, history = restore(resume, cfg, identity, parts, tr, ev)
    with (folder/'training.log').open('a', encoding='utf-8') as log:
        for ep in range(epoch+1, cfg['epochs']+1):
            model.train(); start = time.perf_counter()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            sums = dict(hard=0., soft=0., total=0.); skips=consecutive=updates=0
            iterator = iter(tr)
            for step in range(1, steps+1):
                diagnostic = {}
                try:
                    result = step_group(model, ema, list(itertools.islice(iterator, 2)), opt, sch, scaler, device, diagnostic)
                except Exception:
                    write_json(folder/'nonfinite_diagnostic.json', dict(epoch=ep, step=step, **diagnostic))
                    opt.zero_grad(set_to_none=True)
                    raise
                for key in sums: sums[key] += result[key]
                skips += result['skipped']; updates += not result['skipped']
                consecutive = consecutive+1 if result['skipped'] else 0
                if result['skipped']:
                    with (folder/'amp_events.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(epoch=ep, step=step, **diagnostic))+'\n')
                if skips >= cfg['max_epoch_skips'] or consecutive >= cfg['max_consecutive_skips']:
                    raise FloatingPointError('Excessive AMP skips; previous complete epoch retained')
                if device.type=='cuda' and torch.cuda.max_memory_reserved() > server['max_vram_gib']*2**30:
                    raise RuntimeError('VRAM budget exceeded; no batch adjustment')
                if step == 1 or step % 100 == 0 or step == steps:
                    write_json(folder/'status.json', dict(state='RUNNING', epoch=ep, step=step, steps=steps,
                        optimizer_updates=updates, amp_skips=skips, student_oof='N/A'))
            history.append(dict(epoch=ep, **{k+'_loss': v/steps for k,v in sums.items()},
                optimizer_updates=updates, ema_updates=int(ema.n_averaged), amp_skips=skips,
                seconds=time.perf_counter()-start,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0.))
            checkpoint(resume, cfg, identity, ep, history, parts, tr, ev)
            write_csv(folder/'train_history.csv', history)
            log.write(json.dumps(history[-1])+'\n'); log.flush()
    # Only fixed last-epoch artifacts: no official Val checkpoint selection.
    save_torch(folder/'student_raw.pth', model.state_dict())
    save_torch(folder/'student_ema.pth', ema.module.state_dict())
    metrics = {}
    for name, candidate in (('raw', model), ('ema', ema)):
        metrics[name] = infer(candidate, ev, study_targets_from_rows(val), folder/f'{name}_val_predictions.csv', device)
    write_json(folder/'metrics.json', dict(val=metrics, student_oof='N/A', state='EXPLORATORY', epoch=cfg['epochs']))
    write_csv(folder/'per_class_metrics.csv', [dict(version=v, **r) for v,m in metrics.items() for r in m['per_class']])
    write_json(folder/'model_size.json', {v: dict(bytes=(folder/f'student_{v}.pth').stat().st_size+config_path.stat().st_size,
        limit_bytes=500000000, models_loaded=1, latency_status='T4_UNVERIFIED') for v in ('raw','ema')})
    write_json(folder/'run_manifest.json', dict(task=task, identity_sha256=identity,
        pretrained_sha256=PRETRAINED_SHA, pretrained_source='torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        teacher_sha256=digest(Path(server['output_root'])/'teachers'/f'{task}.csv'),
        source_audit_sha256=digest(Path(server['output_root'])/'teacher_audit.json'),
        training_scope='full_train', student_oof='N/A', val_used_for_checkpoint_selection=False,
        test_accessed=False, protocol=cfg, torch=str(torch.__version__), cuda=torch.version.cuda))
    write_json(folder/'status.json', dict(state='DONE', epoch=cfg['epochs'], student_oof='N/A'))
    receipt = dict(identity_sha256=identity, sha256={p.name:digest(p) for p in folder.iterdir()
        if p.is_file() and p.name not in ('DONE.json',) and not p.name.startswith('FAILED')})
    write_json(folder/'DONE.json', receipt)
    return receipt
