"""Build isolated T4 six-model candidate from verified original Phase65 assets."""
from __future__ import annotations
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
from cxr.prediction_consistency import sha256

PREFIX='PHASE65_FULL_REPRODUCTION_PORTABLE_20261102/'
NAMES={
 'phase65_source/model/B01.pt':'aed81971bc066db8b84486dd27e4b15088af0917cfdd1a9aeed2c0a57ca125e1',
 'phase65_source/model/B03.pt':'22de342d0b8f68016f5d8f69e413d50454a3e4675e1cce07e59fa896af0044fb',
 'phase65_source/model/N01_EMA.pt':'8d8160638c45c1c6afec8d822b38ac1be337335c28b792022c46822e6481142a',
 'tensorrt_engines/B02_fp32.engine':'286cd4527cf4a69bec719b6679d35ed9de1d696ccccb8f62b754cb72df7be70c',
 'tensorrt_engines/M02_fp32.engine':'54cb461deb38d4e744450f5c33fa0e84396b8b36659a3138e5823bd1df84a94e',
}
S17='ce96237beb19f58b1915ac343225cea1916bcadd156c0faac76232905bf44398'
SUBMISSION='8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c'


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Expected one source occurrence: {old[:75]!r}, found {text.count(old)}')
    return text.replace(old,new)


def build(config):
    cfg=json.loads(Path(config).read_text(encoding='utf-8'))
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    source=Path(cfg['phase65_archive']);weight=Path(cfg['s17_checkpoint']);submission=Path(cfg['historical_submission'])
    if sha256(weight)!=S17 or sha256(submission)!=SUBMISSION:raise ValueError('Frozen weight or historical submission differs')
    with zipfile.ZipFile(source) as archive:
        for name,expected in NAMES.items():
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(PREFIX+name) as inp,target.open('xb') as stream:shutil.copyfileobj(inp,stream)
            if sha256(target)!=expected:raise ValueError(f'Phase65 source asset mismatch: {name}')
        extra=['phase65_source/frozen_predict.py','phase65_source/frozen_e08_manifest.json','requirements-lock.txt']
        extra.extend(i.filename[len(PREFIX):] for i in archive.infolist() if i.filename.startswith(PREFIX+'phase65_source/src/') and not i.is_dir())
        for name in extra:
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(PREFIX+name) as inp,target.open('xb') as stream:shutil.copyfileobj(inp,stream)
        original=archive.read(PREFIX+'phase65_source/predict.py').decode('utf-8')
        if hashlib.sha256(original.encode()).hexdigest()!='fcc65dc88ee54b9ce22c166435d71b117c48c1ef68dd57d206ca9fcbefdba681':
            raise ValueError('Wrong Phase65 source code')
        original=replace_once(original,
            'NAMES = ("B01", "B02", "B03", "M02", "N01_EMA")',
            'NAMES = ("B01", "B02", "B03", "M02", "N01_EMA", "S17_F3")')
        original=replace_once(original,
            'GRAPH_NAMES = ("B01", "B03", "N01_EMA")',
            'GRAPH_NAMES = ("B01", "B03", "N01_EMA", "S17_F3")')
        original=replace_once(original,
            'assert set(models) == {"B01", "B03", "N01_EMA"}',
            'assert set(models) == {"B01", "B03", "N01_EMA", "S17_F3"}')
        original=replace_once(original,
            'result = 0.5 * e03 + 0.5 * probs["N01_EMA"]',
            'e08 = 0.5 * e03 + 0.5 * probs["N01_EMA"]\n    result = 0.75 * e08 + 0.25 * probs["S17_F3"]')
        original=replace_once(original,'from pathlib import Path','from pathlib import Path\nimport json')
        original=replace_once(original,'from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD',
                              'from cxr.data.dataset import IMAGENET_MEAN, IMAGENET_STD\nfrom cxr.prediction_consistency import sha256')
        original=replace_once(original,'def load_models():\n',
                              'def load_models():\n    root = Path(__file__).resolve().parent.parent\n'
                              '    manifest = json.loads((root / "manifest_sha256.json").read_text(encoding="utf-8"))\n'
                              '    for relative, expected in manifest.items():\n'
                              '        source = root / relative\n'
                              '        if not source.is_file() or sha256(source) != expected:\n'
                              '            raise ValueError(f"Frozen deployment asset mismatch: {relative}")\n')
        (out/'phase65_source/predict.py').write_text(original,encoding='utf-8')
    target=out/'phase65_source/model/S17_F3.pt';shutil.copy2(weight,target)
    if sha256(target)!=S17:raise ValueError('Copied S17 SHA mismatch')
    shutil.copy2(submission,out/'submission_original_78_81.csv')
    if sha256(out/'submission_original_78_81.csv')!=SUBMISSION:raise ValueError('Submission copy mismatch')
    shutil.copy2(Path(__file__).with_name('e08_s17_t4_cost.py'),out/'phase65_source/src/cxr/e08_s17_t4_cost.py')
    shutil.copy2(Path(__file__).with_name('prediction_consistency.py'),out/'phase65_source/src/cxr/prediction_consistency.py')
    (out/'cost.py').write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source/src"))\nfrom cxr.e08_s17_t4_cost import main\nif __name__ == "__main__": main()\n',encoding='utf-8')
    (out/'predict.py').write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source"))\nsys.path.insert(0,str(Path(__file__).resolve().parent/"phase65_source/src"))\nfrom phase65_source.predict import load_models,predict_image,PRECISION\n',encoding='utf-8')
    config_payload=dict(formula='0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3',
                        historical_submission_sha256=SUBMISSION,
                        precision='B01/B03/N01_EMA/S17_F3 PyTorch FP16 CUDA Graph; B02/M02 TensorRT FP32',
                        status='T4_UNVERIFIED_FOR_SIX_MODELS',batch=1)
    (out/'config.json').write_text(json.dumps(config_payload,indent=2),encoding='utf-8')
    # Hash everything needed at runtime; manifest does not include itself or reports.
    manifest={str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest_sha256.json'}
    (out/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    runtime=['phase65_source/model/B01.pt','phase65_source/model/B03.pt','phase65_source/model/N01_EMA.pt',
             'phase65_source/model/S17_F3.pt','tensorrt_engines/B02_fp32.engine','tensorrt_engines/M02_fp32.engine']
    total=sum((out/n).stat().st_size for n in runtime)
    total+=sum((out/n).stat().st_size for n in ['config.json','phase65_source/frozen_e08_manifest.json'])
    report=dict(loaded_assets=runtime,loaded_and_config_bytes=total,limit_bytes=500000000,
                size_pass=total<=500000000,source_archive_sha256=sha256(source),s17_sha256=S17,
                submission_sha256=SUBMISSION,t4_status='T4_UNVERIFIED',
                numerical_status='PENDING_T4_SIX_MODEL_CHECK',
                note='Original Phase65 5-model T4 benchmark does not validate S17-added 6-model entry')
    (out/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['size_pass']:raise ValueError('Runtime asset size exceeds 500MB')
    print(json.dumps(report))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);args=ap.parse_args()
    build(args.config)


if __name__=='__main__':main()
