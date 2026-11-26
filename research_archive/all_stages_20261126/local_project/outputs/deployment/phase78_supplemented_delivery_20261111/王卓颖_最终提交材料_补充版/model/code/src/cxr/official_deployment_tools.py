"""Prepare/verify engines in the organizer's fixed container; never train models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMAGE = 'crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0'
FORMULA = '0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3'
SUBMISSION_SHA = '8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c'
ENGINES = ('B02', 'M02')
CHECKPOINTS = ('B01', 'B03', 'N01_EMA', 'S17_F3')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Unsafe manifest path: {relative}')
    return path


def lock_manifest(root):
    root = Path(root)
    records = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
               if p.is_file() and p.name != 'manifest_sha256.json' and '__pycache__' not in p.parts}
    write_json(root / 'manifest_sha256.json', records)
    return records


def verify_package(root):
    root = Path(root)
    records = json.loads((root / 'manifest_sha256.json').read_text(encoding='utf-8'))
    for relative, expected in records.items():
        p = safe_file(root, relative)
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f'Missing/modified package file: {relative}')
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if cfg['formula'] != FORMULA or cfg['batch'] != 1 or cfg['historical_submission_sha256'] != SUBMISSION_SHA:
        raise ValueError('Frozen formula, batch or submission provenance changed')
    if sha256(root / 'submission.csv') != SUBMISSION_SHA:
        raise ValueError('Historical submission changed')
    return {'status': 'PASS', 'files': len(records), 'submission_sha256': SUBMISSION_SHA}


def environment_errors(info):
    errors = []
    required = {'system': 'Linux', 'python_minor': '3.12', 'torch_base': '2.10.0',
                'cuda': '12.8', 'tensorrt_base': '10.9.0', 'onnxruntime': '1.26.0'}
    for key, expected in required.items():
        if info.get(key) != expected:
            errors.append(f'{key}: expected {expected}, got {info.get(key)}')
    if info.get('tensorrt') not in ('10.9.0', '10.9.0.34'):
        errors.append('TensorRT runtime is not official 10.9.0[.34]')
    if not 90000 <= (info.get('cudnn') or 0) < 100000:
        errors.append('cuDNN major version must be 9')
    if info.get('cuda_devices') != 1 or 'T4' not in (info.get('gpu') or ''):
        errors.append('Expose exactly one NVIDIA T4 to the container')
    if info.get('gpu_memory_bytes', 0) < 14 * 1024**3:
        errors.append('Expected T4 16GB (approximately 15GiB reported)')
    if 'CUDAExecutionProvider' not in info.get('ort_providers', []):
        errors.append('Official ONNX Runtime GPU CUDA provider unavailable')
    versions = re.findall(r'\d+', str(info.get('driver', '')))[:3]
    if len(versions) < 2 or tuple(map(int, versions[:2])) < (570, 117):
        errors.append('Driver must be >=570.117')
    errors.extend(info.get('import_errors', []))
    return errors


def collect_environment():
    info = {'system': platform.system(), 'python': platform.python_version(),
            'python_minor': '.'.join(platform.python_version_tuple()[:2]),
            'cpu_count': os.cpu_count(), 'image_expected': IMAGE, 'import_errors': []}
    for name in ('torch', 'torchvision', 'tensorrt', 'onnxruntime', 'numpy', 'PIL'):
        try:
            module = importlib.import_module(name)
            info[name] = module.__version__
            if name == 'torch':
                info.update(torch_base=module.__version__.split('+')[0], cuda=module.version.cuda,
                            cudnn=module.backends.cudnn.version(), cuda_devices=module.cuda.device_count())
                if module.cuda.is_available():
                    props = module.cuda.get_device_properties(0)
                    info.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                                compute_capability=list(module.cuda.get_device_capability(0)))
            elif name == 'tensorrt':
                info['tensorrt_base'] = '.'.join(module.__version__.split('.')[:3])
            elif name == 'onnxruntime':
                info['ort_providers'] = module.get_available_providers()
        except Exception as exc:
            info['import_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=20)
        info['driver'] = result.stdout.strip().splitlines()[0]
    except Exception as exc:
        info['driver_error'] = str(exc)
    if Path('/proc/meminfo').is_file():
        line = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))
        info['host_ram_bytes'] = int(line.split()[1]) * 1024
    info['errors'] = environment_errors(info)
    info['status'] = 'PASS' if not info['errors'] else 'BLOCKED_ENVIRONMENT'
    info['note'] = 'Version check does not prove image identity, engine compatibility, accuracy or latency.'
    return info


def require_environment():
    info = collect_environment()
    if info['errors']:
        raise RuntimeError(json.dumps(info, ensure_ascii=False))
    return info


def size_report(root):
    root = Path(root)
    names = [f'phase65_source/model/{n}.pt' for n in CHECKPOINTS]
    names += [f'tensorrt_engines/{n}_fp32.engine' for n in ENGINES]
    names += ['config.json', 'phase65_source/frozen_e08_manifest.json', 'engine_build_report.json']
    files = {n: (root/n).stat().st_size for n in names if (root/n).is_file()}
    missing = sorted(set(names) - set(files))
    # Conservative: include runtime source code as necessary supporting files too.
    code = [p for p in (root/'phase65_source').rglob('*.py')]
    code += [p for p in root.glob('*.py')]
    total = sum(files.values()) + sum(p.stat().st_size for p in code)
    package_total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()
                        and 'build_sources' not in p.relative_to(root).parts and '__pycache__' not in p.parts)
    total = max(total, package_total)
    return {'status': 'PENDING_ENGINE_BUILD' if missing else ('PASS' if total <= 500_000_000 else 'FAIL'),
            'files': files, 'missing': missing, 'runtime_code_bytes': sum(p.stat().st_size for p in code),
            'loaded_and_necessary_bytes': total, 'limit_bytes': 500_000_000,
            'conservative_package_bytes_excluding_build_sources': package_total,
            'build_sources_used_at_inference': False,
            'note': 'Conservatively counts all candidate files including audit materials. Build sources never enter compiled candidate.'}


def require_engines(root):
    root = Path(root)
    report_path = root/'engine_build_report.json'
    if not report_path.is_file():
        raise RuntimeError('PENDING_ENGINE_BUILD: run build_engines.py in official T4 container first')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['environment']['tensorrt'] not in ('10.9.0', '10.9.0.34'):
        raise RuntimeError('Engine build TensorRT provenance differs')
    if set(report['engines']) != set(ENGINES):
        raise RuntimeError('Incomplete engine build')
    for name, item in report['engines'].items():
        if sha256(root/f'tensorrt_engines/{name}_fp32.engine') != item['sha256']:
            raise RuntimeError(f'Engine changed after build: {name}')
    sizes = size_report(root)
    if sizes['status'] != 'PASS':
        raise RuntimeError(f'Deployment size check failed: {sizes}')


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Build output must be a new separate directory')
    verify_package(root)
    env = require_environment()
    import tensorrt as trt
    import torch
    import numpy as np
    sources = json.loads((root/'build_sources/provenance.json').read_text(encoding='utf-8'))
    references = json.loads((root/'build_sources/synthetic_reference.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    try:
        for path in root.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(root)
            if relative.parts[0] == 'build_sources' or path.name in ('manifest_sha256.json', 'deployment_size.json'):
                continue
            dest = output/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        engine_dir = output/'tensorrt_engines'
        engine_dir.mkdir(exist_ok=True)
        report = {'status': 'BUILT_PENDING_PREDICTION_VALIDATION', 'environment': env,
                  'engines': {}, 'precision': 'FP32; FP16/INT8/TF32 disabled; no calibration',
                  'image_expected': IMAGE, 'checkpoint_weights_modified': False}
        logger = trt.Logger(trt.Logger.WARNING)
        for name in ENGINES:
            onnx_path = root/f'build_sources/{name}_fp32.onnx'
            if sha256(onnx_path) != sources[name]['onnx_sha256']:
                raise ValueError(f'ONNX hash mismatch: {name}')
            started = time.perf_counter()
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            if network.num_inputs != 1 or network.num_outputs != 1:
                raise ValueError('Expected one input and one output')
            inp, out = network.get_input(0), network.get_output(0)
            if tuple(inp.shape) != (1, 3, 384, 384) or tuple(out.shape) != (1, 10):
                raise ValueError('Unexpected network shapes')
            if inp.dtype != trt.float32 or out.dtype != trt.float32:
                raise ValueError('Expected FP32 I/O')
            inp.name = 'input'  # Name only: preserves the frozen runtime tensor binding API.
            out.name = 'logits'
            config = builder.create_builder_config()
            for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32):
                config.clear_flag(flag)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
            config.max_aux_streams = 0
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError(f'TensorRT build failed: {name}')
            path = engine_dir/f'{name}_fp32.engine'
            path.write_bytes(bytes(serialized))
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None or engine.num_io_tensors != 2:
                raise RuntimeError('Built engine failed deserialization')
            if engine.get_tensor_dtype('input') != trt.float32 or engine.get_tensor_dtype('logits') != trt.float32:
                raise ValueError('Built engine I/O dtype mismatch')
            if tuple(engine.get_tensor_shape('input')) != (1, 3, 384, 384) or tuple(engine.get_tensor_shape('logits')) != (1, 10):
                raise ValueError('Built engine I/O shape mismatch')
            ctx = engine.create_execution_context()
            x = torch.zeros((1, 3, 384, 384), device='cuda', dtype=torch.float32)
            y = torch.empty((1, 10), device='cuda', dtype=torch.float32)
            if ctx is None or not ctx.set_tensor_address('input', x.data_ptr()) or not ctx.set_tensor_address('logits', y.data_ptr()):
                raise RuntimeError('Engine context/binding failed')
            if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('Engine synthetic forward failed')
            torch.cuda.synchronize()
            if not torch.isfinite(y).all().item():
                raise RuntimeError('Engine synthetic logits nonfinite')
            tests = []
            for ref in references[name]:
                values = (np.zeros(ref['shape'], dtype=np.float32) if ref['kind']=='zero' else
                          np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32))
                if hashlib.sha256(values.tobytes()).hexdigest() != ref['input_bytes_sha256']:
                    raise ValueError('Synthetic reference input changed')
                x.copy_(torch.from_numpy(values))
                if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError('Engine reference forward failed')
                torch.cuda.synchronize()
                actual = y.cpu()
                expected = torch.tensor(ref['checkpoint_logits'], dtype=torch.float32)
                logit_error = float((actual-expected).abs().max())
                prob_error = float((actual.sigmoid()-expected.sigmoid()).abs().max())
                if not torch.isfinite(actual).all().item() or logit_error > 1e-3 or prob_error > 1e-4:
                    raise RuntimeError(f'Engine synthetic equivalence failed: {name}, {logit_error}, {prob_error}')
                tests.append({'input':ref['kind'], 'logit_max_abs':logit_error, 'probability_max_abs':prob_error})
            report['engines'][name] = dict(sha256=sha256(path), bytes=path.stat().st_size,
                source=sources[name], build_seconds=time.perf_counter()-started,
                synthetic_finite=True, synthetic_reference_tests=tests, real_prediction_consistency='PENDING')
            del ctx, engine, runtime, x, y, parser, network, builder, serialized
            torch.cuda.empty_cache()
        write_json(output/'engine_build_report.json', report)
        sizes = size_report(output)
        write_json(output/'deployment_size.json', sizes)
        if sizes['status'] != 'PASS':
            raise ValueError('Built candidate exceeds model size limit')
        write_json(output/'package_status.json', {'status': 'BUILT_UNVERIFIED',
                   'remaining': ['full Test historical consistency', 'official 100-image test', 'declaration sign-off']})
        lock_manifest(output)
        # Recount after generated report and hash inventory are present.
        sizes = size_report(output)
        if sizes['status'] != 'PASS':
            raise ValueError('Final candidate package exceeds size limit')
        write_json(output/'deployment_size.json', sizes)
        lock_manifest(output)
        if size_report(output)['status'] != 'PASS':
            raise ValueError('Final manifest/report size exceeds limit')
        verify_package(output)
        return report
    except Exception as exc:
        write_json(output/'BUILD_FAILED.json', {'error': f'{type(exc).__name__}: {exc}',
                   'status': 'NOT_USABLE', 'note': 'Preserved diagnostic output; use a new output directory on retry.'})
        raise


def main(action, root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(root)/'config.json')
    parser.add_argument('--output', type=Path)
    if action == 'check':
        parser.add_argument('--assets-only', action='store_true')
        parser.add_argument('--load-models', action='store_true')
    args = parser.parse_args()
    if args.config.resolve() != (Path(root)/'config.json').resolve():
        raise ValueError('Use this package\'s immutable config.json')
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(Path(root).resolve())):
        raise ValueError('Output must be new and outside immutable package')
    if action == 'build':
        if not args.output:
            parser.error('--output required')
        report = build(root, args.output)
    else:
        report = {'assets': verify_package(root), 'sizes': size_report(root)}
        if not args.assets_only:
            report['environment'] = collect_environment()
            if args.load_models:
                if report['environment']['errors']:
                    raise RuntimeError('Environment failed; no model loading attempted')
                import predict
                state = predict.load_models()
                report['model_load'] = {'status': 'PASS', 'torch_models': sorted(state['models']),
                                        'trt_models': sorted(state['trt_engines'])}
        if args.output:
            write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get('environment', {}).get('errors'):
        raise SystemExit(2)
