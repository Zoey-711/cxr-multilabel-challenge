"""Update bounded cost scenarios from target GPU microbenchmarks (not full epochs)."""
import argparse,json
from pathlib import Path
from cxr.batch.plan import load_plan,write_json,ROOT

def estimate(plan,directory):
    reports=[json.loads(p.read_text(encoding='utf-8')) for p in Path(directory).rglob('benchmark_*.json')]
    history=json.loads((ROOT/'docs/batch17_costs.json').read_text(encoding='utf-8'))
    models={'D':'densenet121','C':'convnext_tiny','S':'swin_t','N':'efficientnet_v2_s'}
    bounds={}
    for branch,model in models.items():
        rates=[r['estimated_epoch_minutes'] for r in reports if r.get('model')==model]
        if not rates:continue
        bounds[branch]={'target_microbenchmark_epoch_minutes':[min(rates),max(rates)],
                       'fivefold_historical_stop_gpu_hours':[5*x*history[branch]['epochs']/60 for x in (min(rates),max(rates))],
                       'fivefold_30epoch_gpu_hours':[5*x*30/60 for x in (min(rates),max(rates))]}
    tasks=[]
    for task in plan['tasks']:
        rates=[r['estimated_epoch_minutes'] for r in reports if r.get('model')==task['backbone']]
        tasks.append({'id':task['id'],'model':task['backbone'],'ema':task['ema'],
                      'estimated_30epoch_gpu_hours':[30*x/60 for x in (min(rates),max(rates))] if rates else None,
                      'state':'target_microbenchmark_estimate' if rates else 'UNMEASURED'})
    missing=[t['id'] for t in tasks if t['state']=='UNMEASURED']
    return {'tasks':tasks,'task_count':len(tasks),'missing_target_measurements':missing,
            'total_22_gpu_hours_30epochs':None if missing else [sum(t['estimated_30epoch_gpu_hours'][i] for t in tasks) for i in (0,1)],
            'backbones':bounds,'A_gpu_hours_historical_stop':None if not {'D','C','S'}<=set(bounds) else [sum(bounds[b]['fivefold_historical_stop_gpu_hours'][i] for b in ('D','C','S')) for i in (0,1)],
            'A_gpu_hours_30epochs':None if not {'D','C','S'}<=set(bounds) else [sum(bounds[b]['fivefold_30epoch_gpu_hours'][i] for b in ('D','C','S')) for i in (0,1)],
            'N_gpu_hours_historical_stop':bounds.get('N',{}).get('fivefold_historical_stop_gpu_hours'),
            'N_gpu_hours_30epochs':bounds.get('N',{}).get('fivefold_30epoch_gpu_hours'),
            'caveats':['Microbenchmark only; cached compute + separate ZIP read, no measured full-epoch claim.',
                       'Add independent Val/OOF, checkpoint I/O, EMA Val per epoch, and packing overhead.',
                       'Actual early stopping and shared-storage contention unknown. No automatic budget increase.'],
            'conditional_max_tasks':4,'conditional_enabled':plan['conditional_extension']['enabled'],
            'default_training_tasks':26,'GH_gpu_hours_estimate':None,
            'GH_estimate_reason':'G/H backbone and median epochs freeze only after complete five-fold OOF',
            'EMA_extra_training_tasks':0,
            'wall_clock_hours':None,'reason':'Requires actual friend-authorized GPU model/count and measured rates'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--directory',required=True,type=Path)
    a=p.parse_args();result=estimate(load_plan(a.config),a.directory);write_json(a.directory/'cost_estimate.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
