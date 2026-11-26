"""CPU mocks test the real packaged cost loop; not GPU accuracy or timing."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]/'outputs/deployment/王卓颖_官方环境修订构建包_20261110'


@pytest.mark.parametrize('partial', [False, True])
def test_real_cost_loop_and_mapping(monkeypatch, tmp_path, partial):
    counts = {'load': 0, 'predict': 0, 'sync': 0}
    vector = np.arange(1,11)/20
    def load():
        counts['load'] += 1
        return {'fake': True}
    def predict(path, state):
        assert state == {'fake': True}
        counts['predict'] += 1
        return vector.copy()
    def sync():
        counts['sync'] += 1
    fake_predict = SimpleNamespace(load_models=load, predict_image=predict)
    monkeypatch.setitem(sys.modules, 'predict', fake_predict)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True, get_device_name=lambda:'Tesla T4 (CPU MOCK)', synchronize=sync)))
    spec = importlib.util.spec_from_file_location('cost_under_test', PACKAGE/'phase65_source/src/cxr/e08_s17_t4_cost.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.predict is fake_predict
    root = tmp_path/'kit'
    root.mkdir()
    timed, warm = tmp_path/'timed',tmp_path/'warm'
    timed.mkdir(); warm.mkdir()
    rows=[]
    reference={}
    for i in range(100):
        name=f'timed_{i:03}.jpg'
        (timed/name).write_bytes(f'timed-{i}'.encode())
        rows.append(dict(image=name, Subject_id=f'p{i}', Study_id=f's{i}', study_image_count=2 if partial and i==0 else 1))
        reference[(f'p{i}',f's{i}')]=vector
    if partial:
        rows.append(dict(image='missing_study_view.jpg', Subject_id='p0', Study_id='s0', study_image_count=2))
    for i in range(5):
        (warm/f'warm_{i}.jpg').write_bytes(f'warm-{i}'.encode())
    with (root/'test_image_index.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['image','Subject_id','Study_id','study_image_count'])
        writer.writeheader(); writer.writerows(rows)
    submission=root/'submission.csv'
    submission.write_text('CPU MOCK REFERENCE',encoding='utf-8')
    (root/'config.json').write_text(json.dumps({'formula':'0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3',
        'historical_submission_sha256':hashlib.sha256(submission.read_bytes()).hexdigest()}))
    monkeypatch.setattr(module,'ROOT',root)
    monkeypatch.setattr(module,'verify_manifest',lambda root:None)
    monkeypatch.setattr(module,'read_submission',lambda path:reference)
    output=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['cost.py',str(submission),str(timed),str(warm),'--output',str(output)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == (2 if partial else 0)
    report=json.loads((output/'report.json').read_text())
    assert counts['load']==1 and counts['predict']==305 and counts['sync']==605
    assert report['records']==300
    assert report['consistency_status']==('UNVERIFIED_PARTIAL_STUDY' if partial else 'PASS')
    assert len(list(csv.DictReader((output/'timings_300.csv').open())))==300
    assert len(list(csv.DictReader((output/'predictions_300.csv').open())))==300
    assert len(list(csv.DictReader((output/'study_predictions.csv').open())))==(297 if partial else 300)


def test_frozen_prediction_arithmetic_unchanged():
    from cxr.official_package_repair import function_dump
    old=PACKAGE.parent/'王卓颖_最终提交材料_待验收_20261110/phase65_source/predict.py'
    new=PACKAGE/'phase65_source/predict.py'
    for name in ('gray_first_preprocess','predict_image'):
        assert function_dump(old.read_text(encoding='utf-8'),name)==function_dump(new.read_text(encoding='utf-8'),name)
