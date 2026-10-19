"""Update bounded cost scenarios from target GPU microbenchmarks (not full epochs)."""
import argparse,json
from pathlib import Path
from cxr.batch.plan import load_plan,write_json,ROOT

def estimate(plan,directory):
    reports=[json.loads(p.read_text(encoding='utf-8')) for p in Path(directory).glob('benchmark_*.json')]
    history=json.loads((ROOT/'docs/batch17_costs.json').read_text(encoding='utf-8'))
    models={'D':'densenet121','C':'convnext_tiny','S':'swin_t'}
    bounds={}
    for branch,model in models.items():
        rates=[r['estimated_epoch_minutes'] for r in reports if r.get('model')==model]
        if not rates:continue
        bounds[branch]={'target_microbenchmark_epoch_minutes':[min(rates),max(rates)],
                       'fivefold_historical_stop_gpu_hours':[5*x*history[branch]['epochs']/60 for x in (min(rates),max(rates))],
                       'fivefold_30epoch_gpu_hours':[5*x*30/60 for x in (min(rates),max(rates))]}
    return {'backbones':bounds,'A_gpu_hours_historical_stop':[sum(b['fivefold_historical_stop_gpu_hours'][i] for b in bounds.values()) for i in (0,1)],
            'A_gpu_hours_30epochs':[sum(b['fivefold_30epoch_gpu_hours'][i] for b in bounds.values()) for i in (0,1)],
            'caveats':['Microbenchmark only; cached compute + separate ZIP read, no measured full-epoch claim.',
                       'Add independent Val/OOF, checkpoint I/O, EMA Val per epoch, and packing overhead.',
                       'Actual early stopping and shared-storage contention unknown. No automatic budget increase.'],
            'optional_seed_tasks':2,'optional_architecture_tasks':0,'EMA_extra_training_tasks':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--directory',required=True,type=Path)
    a=p.parse_args();result=estimate(load_plan(a.config),a.directory);write_json(a.directory/'cost_estimate.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
