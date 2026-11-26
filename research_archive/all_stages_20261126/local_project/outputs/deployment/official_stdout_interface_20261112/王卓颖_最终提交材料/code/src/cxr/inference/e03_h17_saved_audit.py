"""CPU-only re-audit of two completed real inference passes; never fabricates predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from cxr.inference.e03_h17_validation import read_probs,compare,roc_auc_score,sha,save,fuse,NAMES
from cxr.data.archive import read_split_rows,parse_positive_classes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'));out=Path(cfg['output'])
    assert not (out/'running.lock').exists()
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['pass_number']==2 and status['images_done']==status['images_total']==1422
    rows=read_split_rows(cfg['archive'],'val');truth={(r['Subject_id'],r['Study_id']):[int(j in parse_positive_classes(r['Predict_class'])) for j in range(1,11)] for r in rows}
    keys=sorted(truth);y=np.array([truth[k] for k in keys]);a={n:read_probs(out/f'pass1_{n}_val.csv',keys) for n in NAMES+['E03_H17']};b={n:read_probs(out/f'pass2_{n}_val.csv',keys) for n in a}
    refs={n:read_probs(cfg['references'][n],keys) for n in NAMES};refs['E03_H17']=fuse(refs)
    comparisons=[compare(a[n],refs[n],y,keys,n) for n in a]
    repeat={n:float(np.abs(a[n]-b[n]).max()) for n in a}
    assert np.max(np.abs(a['E03_H17']-fuse(a)))<1e-14
    e08=read_probs(cfg['e08_reference'],keys);subjects=sorted({k[0] for k in keys});ix={s:i for i,s in enumerate(subjects)};si=np.array([ix[k[0]] for k in keys]);rng=np.random.default_rng(42);diffs=[];invalid=0
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(subjects),len(subjects)),minlength=len(subjects))[si]
        if np.any((y*w[:,None]).sum(0)==0) or np.any(((1-y)*w[:,None]).sum(0)==0):invalid+=1;continue
        diffs.append(float(roc_auc_score(y,a['E03_H17'],sample_weight=w)-roc_auc_score(y,e08,sample_weight=w)))
    source=json.loads(Path(cfg['candidate_manifest']).read_text(encoding='utf-8'))
    assert all(sha(p['path'])==p['sha256'] for p in source['weights']+source['source_configs'])
    report=dict(state='VAL_REPRODUCTION_COMPLETED',images=1422,studies=len(keys),patients=len(subjects),
        input_tensor_max_abs=0.,input_equivalence_evidence='Every image asserted exact in pass1; completed pass2 proves pass1 finished.',
        historical_comparisons=comparisons,self_repeat_max_abs=repeat,self_repeat_pass=all(v<=cfg['self_tolerance'] for v in repeat.values()),
        self_tolerance=cfg['self_tolerance'],historical_tolerance=1e-4,
        bootstrap_vs_e08=dict(draws=2000,valid=len(diffs),invalid=invalid,seed=42,ci95=np.quantile(diffs,[.025,.975]).tolist(),delta=float(roc_auc_score(y,a['E03_H17'])-roc_auc_score(y,e08))),
        elapsed_seconds=status['elapsed_seconds'],timing_scope='Two full image inference passes including preprocessing comparisons, excluding later CPU analysis',
        peak_allocated_bytes=None,peak_memory_status='UNAVAILABLE: original JSON serialization failed after GPU work; no invented peak',
        recovery='Both complete real inference passes preserved. CPU report rebuilt after NumPy integer JSON serialization bug; no GPU rerun or prediction alteration.',
        weights_and_source_configs_bytes=source['checkpoint_and_source_config_bytes'],test_read=False,submission_generated=False,
        latency='NOT_BENCHMARKED; NEW_CANDIDATE_T4_UNVERIFIED')
    save(out/'report.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
