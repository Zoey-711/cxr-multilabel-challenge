"""Export Phase28's JSON calculations as compact CSV and Markdown artifacts."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, fields, records):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    a=ap.parse_args()
    d=json.loads((a.root/'phase28_results.json').read_text(encoding='utf-8'))
    model_rows=[]
    for name,v in d['models'].items():
        model_rows.append({'model':name,'val_macro_auc':v['val_auc'],'oof_macro_auc':v['oof_auc'] if v['oof_auc'] is not None else 'N/A',
                           'val_protocol':'fivefold ensemble' if name in ('FRIEND_E17','LOCAL_E17','N17_EMA') else 'frozen full-model prediction',
                           'oof_protocol':'full fivefold unseen-patient' if v['oof_auc'] is not None else 'N/A'})
    write_csv(a.root/'model_val_oof_comparison.csv',list(model_rows[0]),model_rows)
    per=[]
    for name,v in d['models'].items():
        for i,auc in enumerate(v['val_per_class_auc'],1):
            per.append({'candidate':name,'class_id':i,'val_auc':auc,'delta_vs_e08':auc-d['models']['E08']['val_per_class_auc'][i-1],
                        'oof_auc':v['oof_per_class_auc'][i-1] if v['oof_per_class_auc'] is not None else 'N/A',
                        'oof_correlation_between_sources':'N/A'})
    for name,v in d['candidates'].items():
        for i,auc in enumerate(v['val_per_class_auc'],1):
            per.append({'candidate':name,'class_id':i,'val_auc':auc,'delta_vs_e08':v['val_per_class_delta_vs_e08'][i-1],
                        'oof_auc':v['oof_per_class_auc'][i-1] if v['oof_per_class_auc'] is not None else 'N/A',
                        'oof_correlation_between_sources':v['oof_prediction_correlation_per_class'][i-1] if v['oof_prediction_correlation_per_class'] else 'N/A'})
    write_csv(a.root/'per_class_auc_and_correlation.csv',list(per[0]),per)
    deployment=[]
    for name,v in d['candidates'].items():
        deployment.append({'candidate':name,'reported_weight_bytes':v['reported_inference_weight_bytes'],
                           'weight_count':v['weight_count'],'weight_source':'local frozen sizes + remote run_manifest actual file bytes; remote weights not downloaded',
                           'size_status':v['size_status'],'T4_end_to_end_status':'LATENCY_UNVERIFIED',
                           'test_prediction_status':v['test_status'],'submission_path':v.get('submission_path',''),
                           'final_status':'RESEARCH_ONLY'})
    write_csv(a.root/'deployment_feasibility.csv',list(deployment[0]),deployment)
    boot={name:{k:v[k] for k in ('val_bootstrap_vs_e08','oof_bootstrap_vs_e17','n17_minus_e17_oof_bootstrap') if k in v}
          for name,v in d['candidates'].items()}
    (a.root/'patient_bootstrap.json').write_text(json.dumps(boot,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PHASE28_EXPORT_PASS',len(model_rows),len(per),len(deployment))


if __name__=='__main__':
    main()
