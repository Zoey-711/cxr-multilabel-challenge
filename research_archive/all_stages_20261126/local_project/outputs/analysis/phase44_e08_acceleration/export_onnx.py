"""Export frozen E08 submodels without training or changing their checkpoints."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE / 'isolated_deps'))

import torch
import onnx
from onnxconverter_common import float16
from cxr.utils.checkpoint import load_model_state

FROZEN = ROOT / 'outputs/analysis/phase30_deployment/candidate_deployment/frozen_e08_manifest.json'
NAMES = ('B01', 'B02', 'B03', 'M02', 'N01_EMA')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=(*NAMES, 'all'), default='all')
    args = parser.parse_args()
    manifest = json.loads(FROZEN.read_text(encoding='utf-8'))
    out = HERE / 'onnx_models'; out.mkdir(exist_ok=True)
    results = {}
    for name in NAMES if args.model == 'all' else (args.model,):
        source = manifest['models'][name]
        path = ROOT/'outputs/experiments'/source['run_id']/source['checkpoint']
        if path.stat().st_size != source['checkpoint_bytes'] or sha(path) != source['checkpoint_sha256']:
            raise ValueError(f'Frozen source mismatch: {name}')
        fp32 = out/f'{name}_fp32.onnx'
        fp16 = out/f'{name}_fp16.onnx'
        model = load_model_state(path, device='cpu')
        with torch.inference_mode():
            torch.onnx.export(model, torch.zeros(1,3,384,384), str(fp32),
                              input_names=['image'], output_names=['logits'],
                              opset_version=17, dynamo=False, do_constant_folding=True)
        del model; gc.collect()
        graph = onnx.load(str(fp32))
        onnx.checker.check_model(graph)
        graph16 = float16.convert_float_to_float16(graph, keep_io_types=True, disable_shape_infer=True)
        onnx.save(graph16, str(fp16))
        onnx.checker.check_model(str(fp16))
        del graph, graph16; gc.collect()
        results[name] = {'source':str(path),'source_sha256':sha(path),
                         'fp32_bytes':fp32.stat().st_size,'fp32_sha256':sha(fp32),
                         'fp16_bytes':fp16.stat().st_size,'fp16_sha256':sha(fp16),
                         'input_shape':[1,3,384,384],'opset':17,
                         'status':'EXPORTED_CPU_ONLY; GPU accuracy and latency unverified'}
        (HERE/'onnx_export_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(name,results[name]['fp16_bytes'],flush=True)

if __name__=='__main__': main()
