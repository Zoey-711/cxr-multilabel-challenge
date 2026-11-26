"""Audited E17 OOF soft targets and Study-balanced distillation loss."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cxr.data.dataset import CXRArchiveDataset


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audit_teacher(root, config, train_rows):
    root = Path(root)
    spec = config['data']['teacher_oof']
    p_path = root/spec['probabilities']
    i_path = root/spec['source_index']
    manifest = json.loads((root/spec['freeze_manifest']).read_text(encoding='utf-8'))
    if manifest['status'] != 'VERIFIED' or manifest['counts']['models'] != 15:
        raise ValueError('Teacher freeze manifest is incomplete')
    if manifest['handoff_sha256'] != {'e17_oof_study_probabilities.csv': spec['probabilities_sha256'],
                                       'e17_oof_source_index.csv': spec['source_index_sha256']}:
        raise ValueError('Teacher source hashes differ from frozen manifest')
    for path, expected in [(p_path,spec['probabilities_sha256']), (i_path,spec['source_index_sha256'])]:
        if digest(path) != expected:
            raise ValueError(f'Teacher file hash changed: {path}')
    expected_folds = manifest['fold_hashes']
    frozen_dir = root/'outputs/analysis/phase17_patient_folds'
    if any(digest(frozen_dir/name) != sha for name,sha in expected_folds.items()):
        raise ValueError('Frozen patient/image Fold index changed')
    patients = {r['Subject_id']:int(r['fold']) for r in read_csv(frozen_dir/'patient_folds.csv')}
    studies = {(r['Subject_id'],r['Study_id']):int(r['fold']) for r in read_csv(frozen_dir/'study_folds.csv')}
    labels = {(r['Subject_id'],r['Study_id']):r for r in read_csv(root/'batch_assets/train_labels.csv')}
    if len(patients)!=11055 or len(studies)!=18513 or len(labels)!=18513:
        raise ValueError('Unexpected official Train/Fold coverage')
    sources = read_csv(i_path)
    probs = read_csv(p_path)
    if list(probs[0]) != ['Subject_id','Study_id']+[f'Prob_{i}' for i in range(1,11)]:
        raise ValueError('Teacher class order differs')
    if list(sources[0]) != ['Subject_id','Study_id','Fold','D17_Model_ID','C17_Model_ID','S17_Model_ID']:
        raise ValueError('Teacher source index schema differs')
    mapping={}
    for source,prob in zip(sources,probs):
        k=source['Subject_id'],source['Study_id']
        if k!=(prob['Subject_id'],prob['Study_id']) or k in mapping or k not in studies:
            raise ValueError(f'Teacher Study ID, ordering or uniqueness differs: {k}')
        fold=int(source['Fold'])
        if studies[k]!=fold or patients[k[0]]!=fold or int(labels[k]['fold'])!=fold:
            raise ValueError(f'Teacher OOF Fold leaks patient: {k}')
        if any(source[f'{branch}_Model_ID']!=f'{branch}_F{fold:02d}' for branch in spec['fusion']):
            raise ValueError(f'Teacher source model ID differs: {k}')
        vector=np.asarray([float(prob[f'Prob_{i}']) for i in range(1,11)],dtype=np.float32)
        if not np.isfinite(vector).all() or ((vector<0)|(vector>1)).any():
            raise ValueError('Teacher soft target is invalid')
        mapping[k]=vector
    if len(probs)!=len(sources) or set(mapping)!=set(studies):
        raise ValueError('Teacher does not cover complete Train Study set exactly once')
    image_counts=Counter((r['Subject_id'],r['Study_id']) for r in train_rows)
    if len(train_rows)!=30344 or set(image_counts)!=set(mapping):
        raise ValueError('Student Train rows do not match teacher Study set')
    if any((r['Subject_id'],r['Study_id']) not in mapping for r in train_rows):
        raise ValueError('Unmatched student image')
    model_records={(r['branch'],int(r['fold'])):r for r in manifest['models']}
    if len(model_records)!=15:
        raise ValueError('Teacher has missing/duplicate backbone Fold model')
    for branch in spec['fusion']:
        for fold in range(1,6):
            record=model_records[branch,fold]
            for artifact,field in [('best_model.pt','checkpoint_sha256'),
                                   ('oof_analysis/study_predictions.csv','oof_prediction_sha256')]:
                path=root/record['artifact_paths'][artifact]
                if digest(path)!=record[field]:
                    raise ValueError(f'Teacher source model changed: {branch} {fold} {artifact}')
            run_manifest=json.loads((root/record['artifact_paths']['frozen_manifest.json']).read_text(encoding='utf-8'))
            if run_manifest['fold_index_sha256']!=expected_folds or run_manifest['fold']!=fold:
                raise ValueError('Teacher checkpoint Fold provenance differs')
    return mapping,image_counts,{'teacher_probability_sha256':digest(p_path),'teacher_index_sha256':digest(i_path),
        'teacher_manifest_sha256':digest(root/spec['freeze_manifest']),'fold_sha256':expected_folds,
        'studies':len(mapping),'patients':len(patients),'images':len(train_rows),
        'model_count':15,'formula':spec['fusion'],'multi_image_studies':sum(n>1 for n in image_counts.values()),
        'soft_weight_sum':sum(n*(1/n) for n in image_counts.values()),
        'patient_leakage':False,'source':'frozen OOF; each patient predicted only by its held-out Fold',
        'student_training_predictions_are_not_oof':True}


class KDDataset(CXRArchiveDataset):
    def __init__(self, *args, soft_targets, image_counts, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_targets=soft_targets
        self.image_counts=image_counts

    def __getitem__(self,index):
        item=super().__getitem__(index)
        key=item['subject_id'],item['study_id']
        item['soft_target']=torch.from_numpy(self.soft_targets[key])
        item['soft_weight']=torch.tensor(1.0/self.image_counts[key],dtype=torch.float32)
        return item


def make_kd_loader(archive, rows, mapping, counts, *, batch_size, num_workers, seed, input_size, preprocessing):
    dataset=KDDataset(archive,'train',rows=rows,unlisted_policy='negative',
                      augmentation=None,preprocessing=preprocessing,input_size=input_size,
                      soft_targets=mapping,image_counts=counts)
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,
                      generator=generator,pin_memory=torch.cuda.is_available(),persistent_workers=False)


def distill_sums(logits, hard, mask, soft, weight, temperature):
    if logits.shape!=hard.shape or hard.shape!=mask.shape or hard.shape!=soft.shape or logits.shape[-1]!=10:
        raise ValueError('Hard/soft class alignment differs')
    if weight.shape!=(logits.shape[0],) or temperature!=1.0:
        raise ValueError('Unapproved KD temperature or Study weight shape')
    if not bool(torch.isfinite(soft).all()) or bool(((soft<0)|(soft>1)).any()):
        raise ValueError('Invalid teacher probability')
    hard_sum=(F.binary_cross_entropy_with_logits(logits.float(),hard.float(),reduction='none')*mask).sum()
    soft_sum=(F.binary_cross_entropy_with_logits(logits.float()/temperature,soft.float(),reduction='none')*
              mask*weight[:,None]).sum()
    return hard_sum,soft_sum
