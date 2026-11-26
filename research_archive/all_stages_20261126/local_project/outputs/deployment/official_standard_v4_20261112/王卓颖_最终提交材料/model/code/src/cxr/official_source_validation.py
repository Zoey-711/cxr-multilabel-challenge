"""CPU-only regression of frozen ONNX build sources against their checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from cxr.official_deployment_tools import sha256, write_json


def validate(root):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from cxr.utils.checkpoint import load_model_state

    root = Path(root)
    sources = json.loads((root/'build_sources/provenance.json').read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    results = {}
    references = {}
    for name in ('B02', 'M02'):
        checkpoint = root/f'build_sources/{name}.pt'
        graph_path = root/f'build_sources/{name}_fp32.onnx'
        if sha256(checkpoint) != sources[name]['checkpoint_sha256'] or sha256(graph_path) != sources[name]['onnx_sha256']:
            raise ValueError('Source hash mismatch')
        graph = onnx.load(str(graph_path), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError('Unexpected external ONNX tensor dependency')
        onnx.checker.check_model(graph)
        shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
        if len(graph.graph.input) != 1 or shape(graph.graph.input[0]) != [1,3,384,384]:
            raise ValueError('Unexpected input shape')
        if len(graph.graph.output) != 1 or shape(graph.graph.output[0]) != [1,10]:
            raise ValueError('Unexpected output shape')
        model = load_model_state(checkpoint, device='cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
        rng = np.random.default_rng(42)
        tests = []
        references[name] = []
        for kind in ('zero', 'fixed_random'):
            x = np.zeros((1,3,384,384), dtype=np.float32) if kind == 'zero' else rng.normal(size=(1,3,384,384)).astype(np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(['logits'], {'image': x})[0]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('Nonfinite CPU inference')
            prob = lambda v: torch.sigmoid(torch.from_numpy(v)).numpy()
            logit_diff = float(np.abs(expected-actual).max())
            prob_diff = float(np.abs(prob(expected)-prob(actual)).max())
            tests.append({'input': kind, 'logit_max_abs': logit_diff, 'probability_max_abs': prob_diff,
                          'pass': logit_diff <= 1e-3 and prob_diff <= 1e-4})
            references[name].append({'kind': kind, 'seed': 42,
                'input_bytes_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                'checkpoint_logits': expected.tolist(), 'shape': [1,3,384,384]})
        results[name] = {'tests': tests, 'status': 'PASS' if all(t['pass'] for t in tests) else 'FAIL',
                         'checkpoint_sha256': sha256(checkpoint), 'onnx_sha256': sha256(graph_path),
                         'onnx_checker': 'PASS', 'input_name': graph.graph.input[0].name,
                         'output_name': graph.graph.output[0].name}
        del model, session, graph
        gc.collect()
    reference_path = root/'build_sources/synthetic_reference.json'
    if reference_path.exists():
        raise FileExistsError(reference_path)
    write_json(reference_path, references)
    return {'status': 'PASS' if all(v['status']=='PASS' for v in results.values()) else 'FAIL',
            'scope': 'Synthetic CPU FP32 export regression; not real Val/Test/T4 consistency',
            'thresholds_fixed_before_test': {'logits': 1e-3, 'probabilities': 1e-4},
            'models': results, 'environment': {'torch': torch.__version__, 'ort': ort.__version__, 'onnx': onnx.__version__},
            'gpu_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = validate(args.config.resolve().parent)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 2)


if __name__ == '__main__':
    main()
