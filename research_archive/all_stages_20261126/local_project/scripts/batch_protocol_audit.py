"""Read-only actual fixed-index audit; never regenerates folds or initializes GPU."""
import argparse,csv,copy,json
from pathlib import Path
import yaml
from cxr.batch.plan import ROOT,load_plan,write_json

def audit(plan):
    directory=ROOT/plan['fold_directory']
    def read(name):
        with (directory/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    patient=read('patient_folds.csv');study=read('study_folds.csv');image=read('image_folds.csv')
    patients={r['Subject_id']:int(r['fold']) for r in patient}
    studies={(r['Subject_id'],r['Study_id']):int(r['fold']) for r in study}
    images={(r['Subject_id'],r['Study_id'],r['Dicom_id']):int(r['fold']) for r in image}
    if (len(patients),len(studies),len(images))!=(11055,18513,30344):raise ValueError('Official index counts changed')
    if len(patient)!=len(patients) or len(study)!=len(studies) or len(image)!=len(images):raise ValueError('Duplicate fixed index')
    if any(patients[s]!=f for (s,st),f in studies.items()) or any(studies[s,st]!=f for (s,st,im),f in images.items()):
        raise ValueError('Patient/Study/image fold leakage')
    result={'counts':{'patients':len(patients),'studies':len(studies),'images':len(images)},'folds':{},'config_comparisons':{}}
    for fold in range(1,6):
        result['folds'][fold]={'patients':sum(f==fold for f in patients.values()),'studies':sum(f==fold for f in studies.values()),'images':sum(f==fold for f in images.values())}
    for task in plan['tasks']:
        cfg=yaml.safe_load((ROOT/task['config']).read_text(encoding='utf-8'))
        if task['group']!='A':continue
        source=ROOT/task['source_config']
        if source.exists():
            original=yaml.safe_load(source.read_text(encoding='utf-8'))
            original['data']['archive_path']=None
            if task['ema']:original['training']['ema']=cfg['training']['ema']
            if original!=cfg:raise ValueError(f'Unapproved algorithm difference: {task["id"]}')
            result['config_comparisons'][task['id']]='identical except env data root and declared EMA observer'
        else:
            result['config_comparisons'][task['id']]='source config not shipped; locked normalized config SHA verified'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=audit(load_plan(a.config));write_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
