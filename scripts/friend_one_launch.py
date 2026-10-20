"""Fail-closed one-command REMOTE_PREFLIGHT and resumable 26-task launcher."""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from cxr.batch.artifacts import utcnow
from cxr.batch.plan import digest, gpu_tokens, load_plan, write_json
from scripts.batch_run import validate_remote_receipt


REQUIRED = ('CXR_DATA_ROOT', 'CXR_PROTOCOL_DIR', 'CXR_OUTPUT_ROOT',
            'CXR_PRETRAINED_ROOT', 'CXR_BACKUP_ROOT')


def authorized_settings(env):
    for key in REQUIRED:
        if not env.get(key) or not Path(env[key]).is_absolute():
            raise ValueError(f'{key} must be an explicit absolute path')
    devices = gpu_tokens(env.get('CXR_GPUS', ''))
    if not 1 <= len(devices) <= 8:
        raise ValueError('Authorize 1..8 explicit GPU IDs')
    try:
        concurrent = int(env['CXR_MAX_CONCURRENT'])
        preflight = float(env['CXR_PREFLIGHT_GPU_HOURS'])
        total = float(env['CXR_GPU_HOURS'])
        gh = float(env['CXR_GH_GPU_HOURS'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Explicit concurrency, preflight, total 26-task and G/H GPU-hour budgets required') from exc
    if (not 1 <= concurrent <= len(devices) or
            not all(math.isfinite(x) for x in (preflight, total, gh)) or
            not 0 < preflight < total - gh or not 0 < gh < total):
        raise ValueError('Invalid authorized GPU pool/concurrency or 26-task GPU-hour budgets')
    version = env.get('CXR_PREFLIGHT_VERSION', '')
    if not version or not all(c.isascii() and (c.isalnum() or c in '_-') for c in version):
        raise ValueError('CXR_PREFLIGHT_VERSION must be a safe nonempty identifier')
    return devices, concurrent, preflight, total, gh, version


def _pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            kernel.CloseHandle(handle)
            return True
        error = kernel.GetLastError()
        if error in (5,):  # Access denied means a process exists.
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_child_pidfiles(directory):
    """Never resume while a GPU subprocess from an interrupted launch is live."""
    directory = Path(directory)
    for pidfile in directory.rglob('*.pid.json') if directory.exists() else ():
        try:
            owner = json.loads(pidfile.read_text(encoding='utf-8'))
            pid = int(owner['pid'])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Unverifiable child-process record: {pidfile}') from exc
        if owner.get('host') != socket.gethostname() or pid <= 0 or _pid_alive(pid):
            raise RuntimeError(f'GPU child process may still be active: {pidfile}')
        pidfile.unlink()


def recover_stale_locks(root):
    """Clear only proven dead local owners; never touch a live or foreign queue."""
    root = Path(root)
    check_child_pidfiles(root/'queue_logs')
    for name, state_file in (('queue.lock', 'queue_state.json'),
                             ('extension.lock', 'extension_state.json')):
        lock = root/name
        if not lock.exists():
            continue
        if not (root/state_file).is_file():
            raise RuntimeError(f'{name} exists without a resumable state file')
        try:
            owner = json.loads(lock.read_text(encoding='utf-8'))
            pid = int(owner['pid'])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'{name} has no verifiable owner; manual inspection required') from exc
        if owner.get('host') != socket.gethostname() or pid <= 0 or _pid_alive(pid):
            raise RuntimeError(f'{name} owner is live or on another host; refusing concurrent launch')
        lock.unlink()


def run_logged(command, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(f'\n[{utcnow()}] COMMAND: {command!r}\n')
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        pidfile = Path(str(path) + '.pid.json')
        write_json(pidfile, {'pid': process.pid, 'host': socket.gethostname(), 'command': command})
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGINT)
            process.wait()
            raise
        finally:
            if process.poll() is not None:
                pidfile.unlink(missing_ok=True)
        handle.write(f'[{utcnow()}] EXIT: {exit_code}\n')
    if exit_code:
        raise RuntimeError(f'Command exited {exit_code}; see {path}')


def verify_preflight_artifacts(preflight_dir, receipt):
    """A reused passed receipt must still point to the six verified GPU probes."""
    from cxr.batch.gh_preflight import verify_gh_preflight
    path = Path(preflight_dir)/'gh_route_probe'
    report = json.loads((path/'route_report.json').read_text(encoding='utf-8'))
    if report != receipt.get('gh_route_probe'):
        raise ValueError('G/H route probe report differs from passed receipt')
    for item in report['runs']:
        name = item['route']
        config = path/f'{name}.yaml'
        if digest(config) != item['config_sha256']:
            raise ValueError(f'G/H route configuration changed: {name}')
        run = path/'experiments'/(name + '_GH_PREFLIGHT')
        checked = verify_gh_preflight(run, item['config_sha256'])
        if checked['checkpoint_sha256'] != item['checkpoint_sha256']:
            raise ValueError(f'G/H route checkpoint changed: {name}')


def write_delivery_status(root, plan):
    root = Path(root)
    base_file, extension_file = root/'queue_state.json', root/'extension_state.json'
    base = json.loads(base_file.read_text(encoding='utf-8')) if base_file.is_file() else {}
    extension = json.loads(extension_file.read_text(encoding='utf-8')) if extension_file.is_file() else {}
    states = {name: base.get('tasks', {}).get(name, {}).get('state', 'PENDING')
              for name in (t['id'] for t in plan['tasks'] if t['enabled'])}
    states.update({name: extension.get('tasks', {}).get(name, {}).get('state', 'BLOCKED')
                   for name in (s['id'] for s in plan['conditional_extension']['reserved_slots'])})
    inventory = {}
    for name, state in states.items():
        run = root/'experiments'/name
        marker = run/'DONE.json'
        if state == 'complete' and marker.is_file():
            done = json.loads(marker.read_text(encoding='utf-8'))
            inventory[name] = {'done_sha256': digest(marker),
                               'verified_sha256': done.get('verified_sha256', {})}
    details = {name: (base if name in base.get('tasks', {}) else extension).get('tasks', {}).get(name, {})
               for name in states}
    report = {'generated_utc': utcnow(), 'tasks': states,
              'complete_count': sum(value == 'complete' for value in states.values()),
              'failed_or_blocked': {key: {'state': value, 'reason': details[key].get('reason', '')}
                                    for key, value in states.items()
                                    if value in ('failed', 'BLOCKED', 'test_failed')},
              'model_file_inventory': inventory,
              'delivery_root': str(root), 'archives_created': False}
    write_json(root/'reports/one_launch_status.json', report)
    write_json(root/'manifests/one_launch_delivery_manifest.json', report)
    return report


def launch(plan_path, env, *, python=sys.executable, runner=run_logged):
    devices, concurrent, preflight, total, gh, version = authorized_settings(env)
    root = Path(env['CXR_OUTPUT_ROOT'])/'phase17_batch'
    preflight_dir = Path(env['CXR_OUTPUT_ROOT'])/f'phase17_batch_preflight_{version}'
    receipt_path = preflight_dir/'receipt.json'
    log = Path(env['CXR_OUTPUT_ROOT'])/'phase17_batch_launcher'/f'{version}.log'
    check_child_pidfiles(log.parent)
    for prior in Path(env['CXR_OUTPUT_ROOT']).glob('phase17_batch_preflight_*'):
        check_child_pidfiles(prior)
    runner([python, 'scripts/protocol_assets.py', '--config', str(plan_path),
            '--action', 'import', '--directory', env['CXR_PROTOCOL_DIR']], log)
    plan = load_plan(plan_path)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    else:
        if preflight_dir.exists():
            raise RuntimeError(f'Incomplete preflight directory: {preflight_dir}; use a new version after inspection')
        runner([python, 'scripts/remote_preflight.py', '--config', str(plan_path),
                '--output', str(preflight_dir), '--gpus', ','.join(devices),
                '--max-concurrent', str(concurrent), '--budget-hours', str(preflight)], log)
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    validate_remote_receipt(receipt, digest(plan_path), devices, concurrent, env, socket.gethostname())
    verify_preflight_artifacts(preflight_dir, receipt)
    if receipt.get('gpu_hours', float('inf')) > preflight:
        raise ValueError('Recorded preflight cost exceeds authorized preflight budget')
    if (root/'STOP').exists():
        raise RuntimeError('STOP was explicitly requested; use clear-stop after active jobs exit')
    recover_stale_locks(root)
    try:
        runner([python, 'scripts/batch_run.py', '--config', str(plan_path), '--action', 'run-all',
                '--root', str(root), '--gpus', ','.join(devices), '--max-concurrent', str(concurrent),
                '--gpu-hours', str(total), '--gh-gpu-hours', str(gh), '--receipt', str(receipt_path)], log)
    finally:
        if root.is_dir():
            write_delivery_status(root, plan)
    return write_delivery_status(root, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/batch_plan.yaml'))
    args = parser.parse_args()
    try:
        result = launch(args.config, os.environ)
        print(json.dumps({'state': 'finished', 'complete_count': result['complete_count'],
                          'failed_or_blocked': result['failed_or_blocked']}, ensure_ascii=False))
    except BaseException as exc:
        out = os.environ.get('CXR_OUTPUT_ROOT')
        if out and Path(out).is_absolute():
            write_json(Path(out)/'phase17_batch_launcher/failure.json',
                       {'time_utc': utcnow(), 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
