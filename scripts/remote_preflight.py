"""Independent 4090 preflight: real-data one-card smoke, skipped CUDA tests, staged allocation."""
import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from cxr.batch.plan import digest, gpu_tokens, write_json


def stage_sizes(max_concurrent):
    if max_concurrent < 1 or max_concurrent > 8:
        raise ValueError('Authorized concurrency must be 1..8')
    return sorted({min(n, max_concurrent) for n in (1, 2, 4, 8)})


def require_three_cuda_tests(report_file):
    root = ET.parse(report_file).getroot()
    cases = root.findall('.//testcase')
    failures = [x for x in cases if x.find('failure') is not None or x.find('error') is not None]
    skipped = [x for x in cases if x.find('skipped') is not None]
    if len(cases) != 3 or failures or skipped:
        raise RuntimeError(f'Expected 3 passed CUDA tests; got {len(cases)} total, '
                           f'{len(failures)} failed, {len(skipped)} skipped')
    return [{'name': x.attrib['name'], 'class': x.attrib.get('classname')} for x in cases]


def run_checked(command, *, env, log, timeout):
    with log.open('w', encoding='utf-8') as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)


def check_gpu_inventory():
    output = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True).stdout
    names = [line.strip() for line in output.splitlines() if line.strip()]
    count = sum('4090' in name for name in names)
    if len(names) != 8 or count != 8:
        raise RuntimeError(f'Expected eight visible RTX 4090 devices; found {names}')
    return {'visible_gpus': names, 'rtx_4090_count': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--max-concurrent', required=True, type=int)
    parser.add_argument('--budget-hours', required=True, type=float,
                        help='Friend-approved preflight GPU-hour cap; separate from training budget')
    args = parser.parse_args()
    devices = gpu_tokens(args.gpus)
    if not 1 <= args.max_concurrent <= len(devices) or args.max_concurrent > 8:
        parser.error('max-concurrent exceeds authorized GPU pool')
    if args.budget_hours <= 0:
        parser.error('Positive friend-approved preflight GPU-hour budget required')
    if args.output.exists():
        raise FileExistsError(f'Preflight output exists; use a new directory: {args.output}')
    args.output.mkdir(parents=True)
    report = {'state': 'failed', 'kind': 'REMOTE_PREFLIGHT', 'host': socket.gethostname(),
              'plan_sha256': digest(args.config), 'gpus': devices,
              'max_concurrent': args.max_concurrent, 'stages': [], 'cuda_tests': [],
              'gpu_hours': 0.0}
    receipt_path = args.output / 'receipt.json'
    write_json(receipt_path, report)
    spent = 0.0
    def remaining(n=1):
        left = args.budget_hours - spent
        if left <= 0:
            raise TimeoutError('Approved preflight GPU-hour budget exhausted')
        return max(1, int(left * 3600 / n))
    try:
        report['gpu_inventory'] = check_gpu_inventory()
        write_json(receipt_path, report)
        # Existing preflight checks ZIP/SHA, all fold splits, resources, 4090 memory,
        # real-data forward/backward, checkpoint resume, and per-backbone benchmarks.
        started = time.monotonic()
        run_checked([sys.executable, 'scripts/batch_preflight.py', '--config', str(args.config),
                     '--output', str(args.output/'one_card_and_devices'), '--gpus', args.gpus,
                     '--max-concurrent', str(args.max_concurrent), '--gpu-smoke', '--benchmark',
                     '--budget-hours', str(args.budget_hours)], env=os.environ.copy(),
                    log=args.output/'base_preflight.log', timeout=remaining())
        spent += (time.monotonic() - started) / 3600  # The base smoke is sequential.
        base = json.loads((args.output/'one_card_and_devices'/'receipt.json').read_text(encoding='utf-8'))
        if base['state'] != 'passed':
            raise RuntimeError('Real-data one-card preflight did not pass')
        report['environment'] = base['environment']
        report['host_resources'] = base['host_resources']
        report['test_data'] = base['test']
        report['real_data_smoke'] = 'passed'
        report['framework'] = base['framework']
        write_json(receipt_path, report)
        infer_env = os.environ.copy()
        infer_env['CUDA_VISIBLE_DEVICES'] = devices[0]
        started = time.monotonic()
        run_checked([sys.executable, 'scripts/remote_real_inference.py', '--config',
                     'configs/batch17/d17_fold1.yaml', '--output',
                     str(args.output/'real_inference.json')], env=infer_env,
                    log=args.output/'real_inference.log', timeout=remaining())
        spent += (time.monotonic()-started)/3600
        report['real_inference'] = json.loads((args.output/'real_inference.json').read_text(encoding='utf-8'))
        if any(report['real_inference'].get(split, {}).get('real_images') != 2
               for split in ('val', 'test')):
            raise RuntimeError('Real Val and unlabeled Test short inference are both mandatory')
        write_json(receipt_path, report)
        # These are precisely the three CUDA tests skipped by the local CPU run.
        test_env = os.environ.copy()
        test_env['CUDA_VISIBLE_DEVICES'] = devices[0]
        test_env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        test_env['PYTHONDONTWRITEBYTECODE'] = '1'
        started = time.monotonic()
        xml = args.output/'cuda_tests.xml'
        run_checked([sys.executable, '-m', 'pytest', '-q', 'tests/test_training_resume.py',
                     'tests/test_archive_dataset.py::test_archive_test_inference_keeps_study_ids_and_has_no_labels',
                     '--junitxml', str(xml), '-p', 'no:cacheprovider'], env=test_env,
                    log=args.output/'cuda_tests.log', timeout=remaining())
        spent += (time.monotonic()-started)/3600
        report['cuda_tests'] = require_three_cuda_tests(xml)
        write_json(receipt_path, report)
        started = time.monotonic()
        route_root = args.output/'gh_route_probe'
        run_checked([sys.executable, 'scripts/remote_gh_route_probe.py', '--config',
                     str(args.config), '--output', str(route_root), '--gpu', devices[0],
                     '--timeout-seconds', str(max(1, remaining()-10))], env=os.environ.copy(),
                    log=args.output/'gh_route_probe.log', timeout=remaining())
        spent += (time.monotonic()-started)/3600
        report['gh_route_probe'] = json.loads((route_root/'route_report.json').read_text(encoding='utf-8'))
        if report['gh_route_probe'].get('state') != 'passed' or len(report['gh_route_probe'].get('runs', [])) != 6:
            raise RuntimeError('G/H route short training, checkpoint, Val/Test probe incomplete')
        write_json(receipt_path, report)
        # Concurrent probes are intentionally short. Never start eight full loaders here.
        for count in stage_sizes(args.max_concurrent):
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                jobs = []
                for index, device in enumerate(devices[:count]):
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = device
                    log = args.output/f'concurrent_{count}_{index}.log'
                    jobs.append(pool.submit(run_checked,
                        [sys.executable, 'scripts/remote_gpu_probe.py', '--expected-token', device],
                        env=env, log=log, timeout=remaining(count)))
                for job in jobs:
                    job.result()
            elapsed = time.monotonic()-started
            spent += count*elapsed/3600
            report['stages'].append({'concurrent_gpus': count, 'devices': devices[:count],
                                     'seconds': round(elapsed, 3), 'state': 'passed'})
            report['gpu_hours'] = spent
            write_json(receipt_path, report)
        report['state'] = 'passed'
        report['gpu_hours'] = spent
        write_json(receipt_path, report)
        print(json.dumps({'state': 'passed', 'receipt': str(receipt_path),
                          'gpu_hours': spent, 'stages': report['stages']}, indent=2))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['gpu_hours'] = spent
        write_json(receipt_path, report)
        raise


if __name__ == '__main__':
    main()
