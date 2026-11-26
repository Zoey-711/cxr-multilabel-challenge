import csv, io, json, hashlib, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np

out=Path(__file__).resolve().parent
archive=Path('F:/PHASE65_RANDOM_BENCHMARK_20261102/PHASE65_FULL_REPRODUCTION_PORTABLE_20261102.zip')
def sha(b): return hashlib.sha256(b).hexdigest()
def rows(b): return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
result={}
with zipfile.ZipFile(archive) as z:
    root=z.namelist()[0].split('/')[0]+'/'
    def read(p): return z.read(root+p)
    checks=[]
    for line in read('SHA256SUMS').decode().splitlines():
        if not line.strip(): continue
        expected,p=line.split(None,1); p=p.strip().removeprefix('./')
        try: actual=sha(read(p)); ok=actual==expected
        except KeyError: actual='MISSING'; ok=False
        checks.append({'path':p,'expected':expected,'actual':actual,'pass':ok})
    result['hash_checks']={'count':len(checks),'failed':[x for x in checks if not x['pass']]}
    (out/'file_hash_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    refbytes=read('submission_original_E08_EMA.csv')
    local=out.parents[1]/'submissions/E08_EMA_20261020/submission.csv'
    result['reference_sha256']=sha(refbytes)
    result['matches_local_original']=local.exists() and sha(local.read_bytes())==sha(refbytes)
    ref=defaultdict(dict)
    for r in rows(refbytes): ref[(r['Subject_id'],r['Study_id'])][int(r['Predict_class'])]=float(r['Probability'])
    index=rows(read('benchmark/timed_image_index.csv'))
    groups=defaultdict(list)
    for r in index: groups[(r['Subject_id'],r['Study_id'])].append(r['image'])
    result['sample']={'images':len(index),'studies':len(groups),'complete_per_index':all(len(groups[(r['Subject_id'],r['Study_id'])])==int(r['study_image_count']) for r in index)}
    result['runs']={}
    for run in ('run_01','run_portable_retest'):
        prefix='benchmark/'+run+'/'
        times=rows(read(prefix+'timings_300.csv')); byimage=defaultdict(list)
        for r in times: byimage[r['image']].append(float(r['elapsed_ms']))
        means=np.array([np.mean(v) for v in byimage.values()])
        preds=rows(read(prefix+'predictions_300.csv'))
        pred={(int(r['round']),r['image']):np.array([float(r['Prob_'+str(i)]) for i in range(1,11)]) for r in preds}
        diffs=[]; binary=ranking=0; recompute_error=0.
        stored={(int(r['round']),r['Subject_id'],r['Study_id']):np.array([float(r['Prob_'+str(i)]) for i in range(1,11)]) for r in rows(read(prefix+'study_predictions_300.csv'))}
        for rnd in (1,2,3):
            for key,images in groups.items():
                p=np.mean([pred[rnd,im] for im in images],axis=0)
                q=np.array([ref[key][i] for i in range(1,11)])
                diffs.extend(abs(p-q)); binary+=int(np.any((p>=.5)!=(q>=.5)))
                ranking+=int(not np.array_equal(np.argsort(-p,kind='stable'),np.argsort(-q,kind='stable')))
                recompute_error=max(recompute_error,float(abs(p-stored[(rnd,*key)]).max()))
        result['runs'][run]={'records':len(times),'images':len(byimage),'three_each':all(len(v)==3 for v in byimage.values()),'mean_ms':float(means.mean()),'p50_ms':float(np.percentile(means,50)),'p95_ms':float(np.percentile(means,95)),'max_probability_difference':float(max(diffs)),'mean_probability_difference':float(np.mean(diffs)),'changed_binary_study_rounds':binary,'changed_ranking_study_rounds':ranking,'study_aggregation_recompute_error':recompute_error}
    loaded=['phase65_source/model/'+x+'.pt' for x in ('B01','B03','N01_EMA')]+['tensorrt_engines/'+x+'_fp32.engine' for x in ('B02','M02')]
    result['actually_loaded_weights_bytes']=sum(z.getinfo(root+p).file_size for p in loaded)
    result['loaded_files']={p:z.getinfo(root+p).file_size for p in loaded}
    result['all_checkpoint_and_engine_bytes']=sum(i.file_size for i in z.infolist() if i.filename.endswith(('.pt','.engine')))
result['timing_scope_gap']='Study Mean aggregation and final Study CSV writing are explicitly outside timing (benchmark/cost.py lines 542 onward).'
(out/'audit_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
