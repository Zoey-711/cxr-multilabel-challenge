"""Preflight-only G/H code paths on real Train images and synthetic soft targets.

Final G/H configurations and teacher OOF remain separately checked after five folds.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from cxr.batch.plan import ROOT, digest, write_json
from cxr.batch.runner import run_command
from cxr.batch.gh_preflight import verify_gh_preflight


def prepare_probe_configs(plan, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'manifests').mkdir()
    labels = root/'manifests/train_labels.csv'
    shutil.copy2(ROOT/'batch_assets/train_labels.csv', labels)
    with labels.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    keys = sorted({(r['Subject_id'], r['Study_id']) for r in rows})
    if len(keys) != 18513:
        raise ValueError('Preflight teacher requires all 18513 official Train studies')
    teacher = root/'synthetic_teacher_probe.csv'
    with teacher.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Subject_id', 'Study_id', *(f'Prob_{i}' for i in range(1, 11))])
        writer.writerows((*key, *([0.5] * 10)) for key in keys)
    source = {task['id'].split('_')[0]: task for task in plan['tasks'] if task['id'].endswith('_F01')}
    specs = [(f'G17_FULL_{branch}_ROUTE', branch, False) for branch in ('D17', 'C17', 'S17', 'N17')]
    specs += [(f'H17_DISTILL_{branch}_ROUTE', branch, True) for branch in ('D17', 'N17')]
    configs = []
    for name, branch, distilled in specs:
        config = yaml.safe_load((ROOT/source[branch]['config']).read_text(encoding='utf-8'))
        config['experiment']['name'] = name.lower()
        config['data'].pop('train_subset')
        config['data']['full_train_expected_counts'] = {'patients': 11055, 'studies': 18513, 'images': 30344}
        if distilled:
            config['data']['teacher_soft_labels'] = {
                'path': str(teacher.resolve()), 'sha256': digest(teacher),
                'formula': {'D17': 0.25, 'C17': 0.25, 'S17': 0.5},
                'label_sha256': digest(labels)}
            config['training']['distillation'] = {
                'temperature': 1.0, 'hard_fraction': 0.5, 'soft_fraction': 0.5,
                'soft_weighting': 'inverse_images_per_study',
                'normalization': 'separate hard image-label and soft Study-label denominators'}
        path = root/f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        configs.append((name, path))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args()
    from cxr.batch.plan import load_plan
    plan = load_plan(args.config)
    configs = prepare_probe_configs(plan, args.output)
    result = {'state': 'failed', 'teacher_role': 'synthetic preflight only; never a training target', 'runs': []}
    write_json(args.output/'route_report.json', result)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        for name, config in configs:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError('G/H route probe exhausted approved preflight budget')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, CXR_OUTPUT_ROOT=str(args.output.resolve()))
            run_id = name + '_GH_PREFLIGHT'
            rc = run_command(['scripts/batch_train.py', '--config', str(config), '--run-id', run_id,
                              '--preflight-steps', '1'], env,
                             args.output/f'{name}.stdout.log', args.output/f'{name}.stderr.log', remaining)
            if rc:
                raise RuntimeError(f'{name} failed with exit {rc}; see {name}.stderr.log')
            run = args.output/'experiments'/run_id
            status = json.loads((run/'status.json').read_text(encoding='utf-8'))
            if status.get('state') != 'preflight_passed' or status.get('amp_skipped_steps'):
                raise RuntimeError(f'{name} AMP/training status invalid')
            probe = verify_gh_preflight(run, digest(config))
            result['runs'].append({'route': name, 'config_sha256': digest(config),
                                   'checkpoint_sha256': probe['checkpoint_sha256'],
                                   'val_sha256': probe['val']['sha256'],
                                   'test_sha256': probe['test']['sha256']})
            write_json(args.output/'route_report.json', result)
        result['state'] = 'passed'
        write_json(args.output/'route_report.json', result)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        write_json(args.output/'route_report.json', result)
        raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
