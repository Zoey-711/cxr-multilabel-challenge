"""CPU-only Study alignment and official CSV rehearsal; no Test labels or GPU."""
import csv
import json

import numpy as np
import pytest

from cxr.batch.test_results import average_test, read_test_probabilities, validate_submission, write_submission, write_test


def table(keys, base):
    return {'keys':keys, 'p':np.tile(np.linspace(base,min(base+.45,1),10),(len(keys),1))}


def test_test_mean_and_full_ten_submission(tmp_path):
    keys=[('subject_00001','study_000001'),('subject_00002','study_000002')]
    first=table(keys,.1)
    second=table(list(reversed(keys)),.2)
    p=tmp_path/'test_predictions.csv'
    write_test(p,first,{'checkpoint_sha256':'a'*64,'no_test_labels':True})
    read_test_probabilities(p,keys,2)
    fused=average_test([first,second],[.25,.75],2)
    assert np.isclose(fused['p'][0,0],.175)
    submission=tmp_path/'E17_submission.csv'
    result=write_submission(submission,fused,{'source_model_ids':['D5','C5','S5']},2)
    assert result['status']=='FORMAT_PASS' and result['rows']==20
    assert validate_submission(submission,keys,2)['study_count']==2
    with submission.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert [r['Predict_class'] for r in rows[:10]]==[str(i) for i in range(1,11)]
    assert rows[0]['Subject_id']=='subject_00001'
    assert json.loads(submission.with_suffix('.manifest.json').read_text())['hidden_test_labels_used'] is False
    with pytest.raises(FileExistsError):
        write_submission(submission,fused,{},2)


def test_submission_rejects_missing_class_duplicate_and_nonfinite(tmp_path):
    keys=[('subject_1','study_1')]
    path=tmp_path/'invalid.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Subject_id','Study_id','Predict_class','Probability'])
        for i in range(1,11):writer.writerow([*keys[0],i,'nan' if i==7 else '.2'])
    with pytest.raises(ValueError,match='Non-finite'):
        validate_submission(path,keys,1)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Subject_id','Study_id','Predict_class','Probability'])
        for i in range(1,11):writer.writerow([*keys[0],1 if i==10 else i,'.2'])
    with pytest.raises(ValueError,match='Duplicate'):
        validate_submission(path,keys,1)


def test_fusion_rejects_study_misalignment():
    first=table([('s1','a'),('s2','b')],.1)
    second=table([('s1','a'),('s2','wrong')],.2)
    with pytest.raises(ValueError,match='differ'):
        average_test([first,second],[.5,.5],2)
