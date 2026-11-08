"""GPU-isolated pair worker; called by the bounded server queue."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    from cxr.regularization_confirmation import core
    from cxr.regularization_confirmation.support import code_identity
    from cxr.phase39_support import write_json,digest
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));out=Path(cfg['output'])
    if cfg['source_hashes']!=code_identity():raise ValueError('Code changed since preparation')
    import torch,torchvision,platform
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if torch.__version__.split('+')[0]!='2.10.0' or torchvision.__version__.split('+')[0]!='0.25.0' or torch.version.cuda!='12.8':
        raise RuntimeError('Expected pinned torch2.10.0/torchvision0.25.0 CUDA12.8; do not silently change protocol')
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Expected Python3.12')
    write_json(out/'runtime_environment.json',dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
        platform=platform.platform(),precision='FP16 autocast',code_commit=cfg['code_commit']))
    sys.argv=[sys.argv[0],'--config',a.config,'--action','smoke']
    gate=out/'cuda_smoke_report.json'
    if not gate.exists():core.main()
    report=json.loads(gate.read_text(encoding='utf-8'))
    if report['config_sha256']!=digest(a.config) or report['state']!='PASS':raise ValueError('Smoke identity mismatch')
    if max(v['peak_reserved_mib'] for v in report['arms'].values())>cfg['max_vram_gib']*1024:raise RuntimeError('VRAM budget exceeded')
    if a.preflight_only:return
    sys.argv=[sys.argv[0],'--config',a.config,'--action','queue'];core.main()
    hashes={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ('DONE.json','queue.lock') and not p.name.endswith('.tmp')}
    write_json(out/'DONE.json',dict(state='DONE',identity_sha256=cfg['identity'],code_commit=cfg['code_commit'],sha256=hashes))


if __name__=='__main__':main()
