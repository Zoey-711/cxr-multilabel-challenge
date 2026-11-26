"""Read-only T4/runtime/asset preflight. No model inference or downloads."""
import json,platform,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'phase65_source/src'))
from cxr.prediction_consistency import sha256


def main():
    import torch,torchvision,tensorrt
    files=json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf-8'))
    for relative,expected in files.items():
        p=ROOT/relative
        if not p.is_file() or sha256(p)!=expected:raise ValueError(f'Asset mismatch: {relative}')
    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None
    report=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
                tensorrt=tensorrt.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                gpu=gpu,verified_files=len(files),asset_sha256='PASS',
                ready_to_attempt_benchmark=bool(gpu and 'T4' in gpu))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['ready_to_attempt_benchmark']:raise SystemExit(2)


if __name__=='__main__':main()
