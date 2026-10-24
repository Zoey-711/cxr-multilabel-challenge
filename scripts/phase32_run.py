"""Phase32 single entry: Linux preflight then bounded independent GPU queue."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr.phase32.protocol import load_plan,server_config
from cxr.batch.plan import write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--action',choices=['run','validate','preflight','status','stop','resume'],default='run')
    args=parser.parse_args();plan=load_plan();cfg=server_config(args.config);root=Path(cfg['output_root'])
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    if args.action=='validate':print('Eight frozen configs valid; REMOTE_PREFLIGHT still required');return
    if args.action=='status':
        path=root/'queue_state.json';print(path.read_text() if path.exists() else 'NOT_STARTED');return
    if args.action=='stop':
        root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();print('Drain requested: current tasks finish, no new tasks start');return
    from cxr.batch.host_gpu_lock import host_gpu_lock
    from cxr.phase32.queue import lock,run
    # The host lock spans preflight, training, evaluation and result export.
    with host_gpu_lock():
        root.mkdir(parents=True,exist_ok=True);guard=lock(root)
        try:
            if args.action=='resume':(root/'STOP').unlink(missing_ok=True)
            if (root/'STOP').exists():raise RuntimeError('STOP marker set; use --action resume to acknowledge')
            from cxr.phase32.preflight import prepare,preflight
            identity=prepare(cfg,plan)
            hours=preflight(cfg,plan,identity)
            if args.action=='preflight':
                print(f"REMOTE_PREFLIGHT passed; GPU-hours used: {hours:.4f}; receipt: {root/'preflight/REMOTE_PREFLIGHT.json'}")
                return
        except Exception as exc:
            write_json(root/'reports/PREFLIGHT_FAILED.json',{'reason':f'{type(exc).__name__}: {exc}'})
            raise
        finally:guard.unlink(missing_ok=True)
        from cxr.phase32.execution import execute
        state=run(plan,cfg,execute,initial_hours=hours)
        print(json.dumps({k:v['state'] for k,v in state['tasks'].items()}))

if __name__=='__main__':main()
