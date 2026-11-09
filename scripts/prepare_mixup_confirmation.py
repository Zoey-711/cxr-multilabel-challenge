"""Generate the authorized Fold3-5 configuration from the working pilot config."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import yaml


def derive_config(old, gpu_ids=None, pair_hours=None):
    if old.get('stage') != 'pilot':
        raise ValueError('Expected the original successful pilot server configuration')
    cfg = dict(old)
    prior = PurePosixPath(old['output_root']) if str(old['output_root']).startswith('/') else Path(old['output_root'])
    cfg['pilot_results_root'] = str(prior)
    cfg['output_root'] = str(prior.with_name(prior.name + '_mixup_folds3_5'))
    cfg['stage'] = 'mixup_confirmation'
    cfg.pop('pilot_pairs', None)
    cfg['gpu_ids'] = list(old['gpu_ids'] if gpu_ids is None else gpu_ids)
    if not cfg['gpu_ids'] or len(set(cfg['gpu_ids'])) != len(cfg['gpu_ids']) or any(type(i) is not int or i < 0 for i in cfg['gpu_ids']):
        raise ValueError('GPU IDs must be unique nonnegative integers')
    cfg['max_concurrent'] = min(3, len(cfg['gpu_ids']))
    cfg['gpu_ids'] = cfg['gpu_ids'][:cfg['max_concurrent']]
    budget = old.get('pair_gpu_hours') if pair_hours is None else pair_hours
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('Set an explicit positive pair_gpu_hours budget')
    cfg['pair_gpu_hours'] = budget
    cfg['gpu_hours'] = 3 * budget
    cfg['upload'] = dict(old.get('upload', {}), enabled=False)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path, help='Existing pilot server config')
    p.add_argument('--output-config', type=Path)
    p.add_argument('--gpu-ids', type=int, nargs='+')
    p.add_argument('--pair-gpu-hours', type=float)
    args = p.parse_args()
    cfg = derive_config(yaml.safe_load(args.config.read_text(encoding='utf-8')), args.gpu_ids, args.pair_gpu_hours)
    dest = args.output_config or args.config.with_name('mixup_folds3_5.local.yaml')
    if dest.resolve() == args.config.resolve():
        raise ValueError('Cannot overwrite pilot configuration')
    # Exclusive creation: preserve any previous launch configuration.
    with dest.open('x', encoding='utf-8', newline='\n') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    print(json.dumps(dict(config=str(dest), output=cfg['output_root'],
                         tasks=['MIXUP_F03', 'MIXUP_F04', 'MIXUP_F05'],
                         arms=['BASE', 'MIXUP'], training_runs=6,
                         max_concurrent=cfg['max_concurrent'], gpu_hours_ceiling=cfg['gpu_hours']), indent=2))


if __name__ == '__main__':
    main()
