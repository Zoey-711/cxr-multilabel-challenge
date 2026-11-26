"""Single preselected Val-passing numerical policy, one historical Test audit."""
import argparse,io,json,time,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cxr.data.archive import read_split_rows,image_member
from cxr.data.dataset import preprocess_image
from cxr.e08_s17_acceleration import NAMES,load_models,fuse
from cxr.historical_val_path_recovery import NumericalGraph
from cxr.prediction_consistency import compare_studies,read_submission,sha256,write_csv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    parent=Path(cfg['output']).parent
    result=json.loads((parent/'run/result.json').read_text(encoding='utf-8'))
    policy=result['selected_policy']
    if policy is None or not any(r['policy']==policy and r['status']=='PASS' and r['batch1_eligible'] for r in result['reports']):
        raise RuntimeError('No Val-passing true batch1 policy: Test audit prohibited')
    for source in cfg['locked_sources']:
        if sha256(source['path'])!=source['sha256']:raise ValueError('Locked source changed')
    # Freeze the sole selected implementation before opening Test metadata/images/reference.
    out=parent/'selected_test_confirmation';out.mkdir(exist_ok=False)
    freeze=dict(policy=policy,val_result_sha256=sha256(parent/'run/result.json'),config_sha256=sha256(args.config),
                val_code_sha256=sha256(Path(__file__).with_name('historical_val_path_recovery.py')),
                test_code_sha256=sha256(Path(__file__)),selection_source='Val only',allows_second_test_candidate=False)
    (out/'selected_policy_lock.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    reference_path=Path(cfg['project'])/'outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv'
    expected_sha='8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c'
    if sha256(reference_path)!=expected_sha:raise ValueError('Wrong historical reference')
    reference=read_submission(reference_path)
    rows=read_split_rows(cfg['archive'],'test');groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['Subject_id'],r['Study_id'])].append(i)
    if len(rows)!=2218 or set(groups)!=set(reference):raise ValueError('Test mapping mismatch')
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    models=load_models(cfg);graph=NumericalGraph(models,policy)
    values=[];start=time.perf_counter()
    with zipfile.ZipFile(cfg['archive']) as z:
        for i,r in enumerate(rows):
            with Image.open(io.BytesIO(z.read(image_member(r,'test')))) as image:
                x=preprocess_image(image,384,preprocessing='black_pad_v1')
            values.append(graph.predict(x))
            if (i+1)%200==0:print(f'LOCKED_TEST {i+1}/2218',flush=True)
    values=np.array(values);actual={k:fuse(values[ii].mean(0)) for k,ii in groups.items()}
    report,detail=compare_studies(actual,reference)
    report.update(policy=policy,images=2218,seconds=time.perf_counter()-start,
                  reference_sha256=expected_sha,t4_status='T4_UNVERIFIED',test_fitting=False,
                  aggregation='Mean image probabilities per model, then exact historical fusion',
                  note='No new official Submission generated; no alternate policy tried after this Test result')
    np.savez_compressed(out/'test_image_components.npz',probabilities=values,image_ids=np.array([r['Dicom_id'] for r in rows]),models=np.array(NAMES))
    write_csv(out/'study_predictions.csv',[dict(Subject_id=k[0],Study_id=k[1],**{f'probability_{j}':float(v) for j,v in enumerate(p,1)}) for k,p in sorted(actual.items())])
    write_csv(out/'comparison.csv',detail)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'manifest_sha256.json').write_text(json.dumps({p.name:sha256(p) for p in out.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
