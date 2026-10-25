"""One invocation: verified Fold1 reuse, remote preflight, eight tasks, fivefold analysis."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr.phase35.protocol import load_plan,server_config
from cxr.batch.plan import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['run','validate','preflight','status','stop','resume','analyze','upload'],default='run');a=p.parse_args()
    plan=load_plan();cfg=server_config(a.config);root=Path(cfg['output_root'])
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    if a.action=='validate':print('8 new Fold2-5 tasks; 2 frozen Fold1 reuses; remote preflight required');return
    if a.action=='status':
        path=root/'queue_state.json';print(path.read_text() if path.exists() else 'NOT_STARTED');return
    if a.action=='stop':root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();print('Drain: running tasks finish, no new task launches');return
    if a.action=='upload':
        from cxr.phase35.upload import upload
        from cxr.batch.host_gpu_lock import host_gpu_lock
        with host_gpu_lock():upload(root,cfg['upload'])
        return
    if a.action=='analyze':
        from cxr.phase35.analysis import complete
        from cxr.batch.host_gpu_lock import host_gpu_lock
        with host_gpu_lock():complete(root,plan,cfg['baseline_results_root'])
        return
    from cxr.batch.host_gpu_lock import host_gpu_lock
    from cxr.phase35.queue import lock,run
    from cxr.phase35.preflight import prepare,preflight
    from cxr.phase35.execution import execute
    from cxr.phase35.analysis import complete
    with host_gpu_lock():
        root.mkdir(parents=True,exist_ok=True);guard=lock(root)
        try:
            if a.action=='resume':(root/'STOP').unlink(missing_ok=True)
            if (root/'STOP').exists():raise RuntimeError('STOP present; use --action resume')
            identity=prepare(cfg,plan);hours=preflight(cfg,plan,identity)
        except Exception as e:
            write_json(root/'reports/PREFLIGHT_FAILED.json',{'reason':str(e)});raise
        finally:guard.unlink(missing_ok=True)
        if a.action=='preflight':print('REMOTE_PREFLIGHT passed; no formal training started');return
        state=run(plan,cfg,execute,initial_hours=hours)
        complete(root,plan,cfg['baseline_results_root'])
        if cfg.get('upload',{}).get('enabled'):
            try:
                from cxr.phase35.upload import upload
                upload(root,cfg['upload'])
            except Exception as e:write_json(root/'reports/UPLOAD_FAILED.json',{'reason':str(e)})
        print(json.dumps({k:v['state'] for k,v in state['tasks'].items()}))
if __name__=='__main__':main()
