import json
from pathlib import Path

import pytest

from cxr.official_deployment_tools import environment_errors, safe_file, size_report, require_engines


def good_env():
    return dict(system='Linux', python_minor='3.12', torch_base='2.10.0', cuda='12.8',
                tensorrt='10.9.0.34', tensorrt_base='10.9.0', onnxruntime='1.26.0',
                cudnn=91002, cuda_devices=1, gpu='Tesla T4', gpu_memory_bytes=15360*1024**2,
                ort_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], driver='570.117.01')


def test_official_environment():
    assert environment_errors(good_env()) == []


@pytest.mark.parametrize('key,value', [('system','Windows'), ('python_minor','3.13'),
    ('torch_base','2.8.0'), ('cuda','12.4'), ('tensorrt','11.4.0.106'),
    ('tensorrt_base','11.4.0'), ('onnxruntime','1.23.2'), ('cudnn',8900),
    ('cuda_devices',2), ('gpu','NVIDIA RTX 4060'), ('driver','570.116'),
    ('ort_providers',['CPUExecutionProvider'])])
def test_wrong_environment_fails(key, value):
    info = good_env()
    info[key] = value
    assert environment_errors(info)


def test_manifest_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_file(tmp_path, '../outside')


def test_missing_engines_not_size_pass(tmp_path):
    assert size_report(tmp_path)['status'] == 'PENDING_ENGINE_BUILD'
    with pytest.raises(RuntimeError, match='PENDING_ENGINE_BUILD'):
        require_engines(tmp_path)


def test_partial_engine_build_rejected(tmp_path):
    (tmp_path/'engine_build_report.json').write_text(json.dumps({
        'environment': {'tensorrt': '10.9.0.34'}, 'engines': {'B02': {}}}))
    with pytest.raises(RuntimeError, match='Incomplete engine'):
        require_engines(tmp_path)


def test_wrong_engine_runtime_rejected(tmp_path):
    (tmp_path/'engine_build_report.json').write_text(json.dumps({
        'environment': {'tensorrt': '11.4.0.106'}, 'engines': {}}))
    with pytest.raises(RuntimeError, match='TensorRT provenance'):
        require_engines(tmp_path)
