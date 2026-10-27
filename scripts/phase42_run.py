"""Phase42 single entry: mandatory preflight, pilots, frozen selection, confirmation."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cxr.batch.host_gpu_lock import host_gpu_lock
from cxr.phase39_support import write_json
from cxr.phase42.protocol import load_plan,server_config,fingerprint,tasks,verify_baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--action',choices=('run','preflight','status','stop','resume','analyze','upload'),default='run')
    args=parser.parse_args();plan=load_plan();cfg=server_config(args.config);root=Path(cfg['output_root'])
    if args.action=='status':
        for name in ('pilot_status.csv','confirmation_status.csv','pilot_budget.json','confirmation_budget.json','selection.json','confirmation_analysis.json'):
            path=root/'reports'/name
            print(name, path.read_text(encoding='utf-8') if path.exists() else 'NOT_AVAILABLE')
        return
    if args.action=='stop':
        root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();print('Drain stop: current tasks finish; no new task');return
    with host_gpu_lock():
        root.mkdir(parents=True,exist_ok=True)
        if args.action=='upload':
            from cxr.phase42.upload import upload
            print(upload(root,cfg['upload']));return
        from cxr.phase42.analysis import pilot,confirmation
        if args.action=='analyze':
            selection=pilot(root,plan,cfg,json.loads((root/'queue_pilot.json').read_text()))
            if selection['selected']:confirmation(root,plan,cfg,selection['selected'])
            return
        if args.action=='resume':(root/'STOP').unlink(missing_ok=True)
        elif (root/'STOP').exists():raise RuntimeError('STOP present; use --action resume')
        from cxr.phase42.preflight import prepare,preflight
        try:
            identity=prepare(plan,cfg);receipt=preflight(plan,cfg,args.config.resolve(),identity)
        except Exception as exc:
            write_json(root/'reports/PREFLIGHT_FAILED.json',dict(type=type(exc).__name__,reason=str(exc)));raise
        if args.action=='preflight':print('REMOTE_PREFLIGHT PASS; no formal task started');return
        from cxr.phase42.queue import run
        state=run(tasks(plan),cfg,args.config.resolve(),fingerprint(identity),'pilot')
        selection=pilot(root,plan,cfg,state)
        if selection['state']=='COMPLETE' and selection['selected'] and cfg['confirmation']['enabled'] and not (root/'STOP').exists():
            try:
                from cxr.phase42.preflight import preserve_baseline
                for fold in range(1,6):preserve_baseline(plan,cfg,fold)
            except Exception as exc:
                write_json(root/'reports/CONFIRMATION_BLOCKED.json',dict(reason=str(exc),selected=selection['selected']))
            else:
                run(tasks(plan,selection['selected']),cfg,args.config.resolve(),fingerprint(identity),'confirmation')
                confirmation(root,plan,cfg,selection['selected'])
        if cfg['upload']['enabled']:
            try:
                from cxr.phase42.upload import upload
                upload(root,cfg['upload'])
            except Exception as exc:write_json(root/'reports/UPLOAD_FAILED.json',dict(reason=str(exc)))
        print(json.dumps(dict(pilot_state=selection['state'],selected=selection['selected'],formal_confirmation_authorized=cfg['confirmation']['enabled'])))


if __name__=='__main__':main()
