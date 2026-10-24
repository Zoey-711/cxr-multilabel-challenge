"""One invocation: budgeted E08 component pilots, gated expansion, CPU analysis."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr.e08_tune.protocol import load_plan,server_config,fingerprint
from cxr.e08_tune.queue import lock,run
from cxr.e08_tune.preflight import prepare,preflight
from cxr.e08_tune.execution import execute
from cxr.e08_tune.selection import select_pilots,complete_folds
from cxr.batch.plan import write_json

def stage(plan,tasks,cfg,output,cap):
    plan=copy.deepcopy(plan);plan['tasks']=tasks
    cfg={**cfg,'output_root':str(output),'gpu_hours':cap}
    identity=prepare(cfg,plan)
    spent=preflight(cfg,plan,identity)
    return run(plan,cfg,execute,initial_hours=spent)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['run','validate','status','stop','resume'],default='run');args=p.parse_args()
    plan=load_plan();cfg=server_config(args.config);root=Path(cfg['output_root'])
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    if args.action=='validate':print('Five new pilots; baseline reused; at most eight conditional additional folds.');return
    if args.action=='status':
        for name in ['pilot','expansion']:
            path=root/name/'queue_state.json';print(name, path.read_text() if path.exists() else 'NOT_STARTED')
        return
    if args.action=='stop':
        for path in [root,root/'pilot',root/'expansion']:path.mkdir(parents=True,exist_ok=True);(path/'STOP').touch()
        print('Drain requested; running tasks finish safely');return
    root.mkdir(parents=True,exist_ok=True);guard=lock(root)
    try:
        if args.action=='resume':
            for path in [root,root/'pilot',root/'expansion']:(path/'STOP').unlink(missing_ok=True)
        if (root/'STOP').exists():raise RuntimeError('STOP set; use --action resume')
        pilots=[t for t in plan['tasks'] if t['fold']==1]
        pilot_state=stage(plan,pilots,cfg,root/'pilot',cfg['gpu_hours'])
        if (root/'STOP').exists():return
        selected=select_pilots(root,plan)
        if selected is None:return
        remaining=min(cfg['extension_gpu_hours'],cfg['gpu_hours']-pilot_state['gpu_hours'])
        extension=[t for t in plan['tasks'] if t['variant'] in selected and t['fold']!=1]
        if extension and remaining>0:
            stage(plan,extension,cfg,root/'expansion',remaining)
        elif extension:
            write_json(root/'reports/extension_status.json',{'state':'BLOCKED','reason':'No authorized remaining extension GPU budget'})
        else:write_json(root/'reports/extension_status.json',{'state':'NOT_SELECTED','reason':'No pilot satisfied frozen evidence rules'})
        complete_folds(root,plan,selected,cfg['baseline_results_root'])
        if cfg.get('upload',{}).get('enabled'):
            try:
                from cxr.e08_tune.upload import upload
                upload(root,cfg['upload'])
            except Exception as e:write_json(root/'reports/UPLOAD_FAILED.json',{'reason':str(e)})
    except Exception as e:
        write_json(root/'reports/LAUNCH_FAILED.json',{'error':f'{type(e).__name__}: {e}'})
        raise
    finally:guard.unlink(missing_ok=True)

if __name__=='__main__':main()
