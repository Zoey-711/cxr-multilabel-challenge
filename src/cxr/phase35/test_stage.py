"""Resumable label-free Test stage; failure never restarts completed training."""
import json,time
from pathlib import Path
from cxr.batch.plan import digest,write_json
from cxr.batch.runner import run_command
from cxr.batch.test_results import read_test_probabilities

def test_stage(run,config,env,logs,deadline,*,pure=False):
    run,logs=Path(run),Path(logs);logs.mkdir(parents=True,exist_ok=True)
    for kind in ('raw','ema'):
        stem='' if kind=='raw' else 'ema_';p=run/(stem+'test_predictions.csv');meta=p.with_suffix('.provenance.json')
        marker=run/(stem+'test.done.json')
        if marker.exists():
            if any(digest(run/n)!=h for n,h in json.loads(marker.read_text()).items()):raise ValueError('Test artifact changed')
            read_test_probabilities(p);continue
        # Partial output preserved; retry only inference, not training.
        for old in (p,meta):
            if old.exists():old.rename(old.with_name(old.name+f'.partial_{time.time_ns()}'))
        args=['scripts/phase35_test.py','--config',str(config),'--run-dir',str(run),'--checkpoint-kind',kind]
        if pure:args+=['--pure']
        rc=run_command(args,env,logs/(stem+'test.out.log'),logs/(stem+'test.err.log'),deadline-time.monotonic())
        if rc:raise RuntimeError(f'Test inference failed: {run.name}/{kind}; trained model retained')
        read_test_probabilities(p)
        write_json(marker,{x.name:digest(x) for x in (p,meta)})
