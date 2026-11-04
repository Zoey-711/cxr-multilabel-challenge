"""Fixed teacher-only comparison; no search and no outer-fold leakage shortcut."""
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
RECIPES = {
    'SKD_CONTROL': {'D17': .25, 'C17': .25, 'S17': .5},
    'SKD_COMPLEMENT': {'D17': .125, 'C17': .125, 'S17': .25, 'N17_EMA': .5},
}
TRAINING = dict(model='efficientnet_v2_s', input_size=384, seed=42,
                micro_batch=8, accumulation=2, epochs=4, warmup_epochs=2,
                backbone_lr=1e-4, head_lr=3e-4, weight_decay=1e-4,
                ema_decay=.999, hard_fraction=.5, temperature=1.,
                checkpoint_rule='fixed_final_epoch_raw_and_ema',
                soft_weighting='inverse_images_per_study', workers=0,
                amp='fp16', max_epoch_skips=20, max_consecutive_skips=4)
PRETRAINED_SHA = 'dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_lock():
    return json.loads((ROOT/'configs/strong_kd_sources.json').read_text(encoding='utf-8'))


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('mode') != 'full_train_development':
        raise ValueError('BLOCKED_OUTER_TEACHERS: ordinary fivefold OOF is not nested teacher OOF')
    for key in ('teacher_root', 'data_root', 'pretrained_root', 'output_root'):
        if not isinstance(cfg.get(key), str) or not Path(cfg[key]).is_absolute():
            raise ValueError('Absolute path required: '+key)
    output = Path(cfg['output_root']).resolve()
    for path in (ROOT, Path(cfg['teacher_root']), Path(cfg['data_root']), Path(cfg['pretrained_root'])):
        path = path.resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Output must be independent of code, data and frozen teachers')
    return cfg


def task_spec(task):
    seed = 2026 if task.endswith('_SEED2026') else 42
    variant = task.removesuffix('_SEED2026')
    if variant not in ('STUDENT_BASE', *RECIPES):
        raise ValueError('Unknown paired experiment')
    return variant, seed


def training_config(task):
    variant, seed = task_spec(task)
    return dict(TRAINING, experiment_id=task, seed=seed,
                hard_fraction=1. if variant == 'STUDENT_BASE' else .5,
                teacher_formula=RECIPES.get(variant),
                student_oof='N/A', evaluation='official_Val_development_only')


def tasks(cfg=None):
    # Conservative hard reservation, not a measured training-time estimate.
    names = ['STUDENT_BASE', *RECIPES]
    if (cfg or {}).get('replicate_seed_enabled', False):
        names += [n+'_SEED2026' for n in names.copy()]
    return [dict(id=k, variant=k, fold=None, timeout_hours=12, max_retries=1) for k in names]
