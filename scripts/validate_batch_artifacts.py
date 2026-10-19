"""CPU-only validation of completed models; missing experiments remain explicit."""
import argparse,json
from pathlib import Path
import yaml
from cxr.batch.results import verify_frozen
from cxr.batch.artifacts import read_probabilities,label_keys
from cxr.batch.plan import write_json
from cxr.batch.test_results import read_test_probabilities

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    plan=yaml.safe_load(a.config.read_text(encoding='utf-8'));rows=[]
    required=['best_model.pth','best_model.sha256','resolved_config.yaml','train_history.csv','train.log',
              'metrics.json','per_class_metrics.csv','val_predictions.csv','oof_predictions.csv',
              'test_predictions.csv','model_size.json','inference_benchmark.json','deployment_eligibility.json',
              'run_manifest.json','environment.json','DONE.json']
    for t in plan['tasks']:
        run=a.root/'experiments'/t['id']
        if not (run/'DONE.json').exists():
            rows.append({'id':t['id'],'state':'failed' if (run/'FAILED.json').exists() else
                         'test_pending' if (run/'TEST_PENDING.json').exists() else 'missing'});continue
        verify_frozen(run)
        names=required+(['ema_'+n for n in ('best_model.pth','best_model.sha256','metrics.json','val_predictions.csv','oof_predictions.csv','test_predictions.csv')] if t['ema'] else [])
        for n in names:
            if not (run/n).is_file():raise ValueError(f'Missing required artifact: {t["id"]}/{n}')
        for prefix in ('','ema_') if t['ema'] else ('',):
            read_test_probabilities(run/(prefix+'test_predictions.csv'),expected_studies=plan.get('expected_test_studies',1414))
            for split in ('val','oof'):
                labels=a.root/'manifests'/('val_labels.csv' if split=='val' else 'train_labels.csv')
                keys=label_keys(labels,None if split=='val' else t['fold'])
                read_probabilities(run/(prefix+split+'_predictions.csv'),labels,keys)
        rows.append({'id':t['id'],'state':'verified'})
    write_json(a.root/'reports/artifact_validation.json',rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
