"""CPU-only synthetic end-to-end cost loop; no measured GPU performance claim."""
import csv
import json
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cxr.e08_s17_consistent_cost import run
from cxr.e08_s17_consistent_entry import aggregate_studies
from cxr.prediction_consistency import sha256


class CostFlowTests(unittest.TestCase):
    def exercise(self, full_audit_status='PASS', corrupt_first_round=False):
        base=Path(__file__).resolve().parents[1]/'outputs/analysis/e08_s17_consistency_standardization_20261110/cpu_cost_harness'/uuid.uuid4().hex
        base.mkdir(parents=True)
        timed=base/'timed';warm=base/'warm';audit=base/'audit'
        for d in (timed,warm,audit):d.mkdir()
        cfgpath=base/'config.json';cfgpath.write_text('{}')
        lock=base/'lock.json';lock.write_text('{}')
        sub=base/'reference.csv';sub.write_text('synthetic fixture')
        rows=[];inventory=[];reference={}
        value=np.linspace(.1,.9,10)
        for i in range(100):
            name=f'image_{i:03d}.jpg';key=('p',f's{i//2:02d}')
            path=timed/name;path.write_bytes(f'CPU_TEST_NOT_AN_IMAGE_{i}'.encode())
            rows.append(dict(image=name,Subject_id=key[0],Study_id=key[1],study_image_count=2))
            inventory.append(dict(image=name,sha256=sha256(path)))
            reference[key]=value
        for i in range(5):(warm/f'warm_{i}.jpg').write_bytes(f'INDEPENDENT_CPU_WARM_{i}'.encode())
        report=dict(status=full_audit_status,config_sha256=sha256(cfgpath),lock_sha256=sha256(lock))
        (audit/'audit_report.json').write_text(json.dumps(report))
        (audit/'test_image_inventory.json').write_text(json.dumps(inventory))
        (audit/'manifest_sha256.json').write_text(json.dumps({n:sha256(audit/n) for n in ['audit_report.json','test_image_inventory.json']}))
        cfg=dict(reference_submission_sha256=sha256(sub),full_image_index='synthetic',full_test_audit=str(audit/'audit_report.json'),file_lock=str(lock))
        calls=dict(load=0,predict=0,aggregate=0)
        def load(_):calls['load']+=1;return object()
        def infer(path,state):
            calls['predict']+=1
            result=value.copy()
            if corrupt_first_round and calls['predict']==6:result[0]=.95
            return result
        def aggregate(rows,values,keys):
            calls['aggregate']+=1
            return aggregate_studies(rows,values,keys)
        entry=SimpleNamespace(load_models=load,predict_image=infer,aggregate_studies=aggregate)
        argv=['cost','--config',str(cfgpath),'--pictures-100',str(timed),'--pictures-5',str(warm),
              '--submission',str(sub),'--output',str(base/'output')]
        with patch.object(sys,'argv',argv), \
             patch('cxr.e08_s17_consistent_cost.checked_config',return_value=cfg), \
             patch('cxr.e08_s17_consistent_cost.read_submission',return_value=reference), \
             patch('cxr.e08_s17_consistent_cost.read_index',return_value=rows), \
             patch('cxr.e08_s17_consistent_cost.environment',return_value={'mode':'SYNTHETIC_CPU_TEST_NOT_A_BENCHMARK'}), \
             patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.synchronize'), \
             patch('torch.cuda.get_device_name',return_value='SYNTHETIC_CPU_TEST'), \
             patch('builtins.print'):
            code=run(entry)
        self.assertEqual(calls,dict(load=1,predict=305,aggregate=150))
        with (base/'output/timings_300.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),300)
        with (base/'output/study_predictions_3_rounds.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),150)
        return code,json.loads((base/'output/report.json').read_text())

    def test_all_rounds_and_shared_functions_execute(self):
        code,r=self.exercise()
        self.assertEqual(code,0);self.assertEqual(r['status'],'PASS')
        self.assertFalse(r['ready_for_official'])

    def test_full_audit_failure_blocks_even_passing_sample(self):
        code,r=self.exercise(full_audit_status='FAIL')
        self.assertEqual(code,2);self.assertEqual(r['status'],'BLOCKED_FULL_TEST_INCONSISTENCY')

    def test_first_round_failure_blocks(self):
        code,r=self.exercise(corrupt_first_round=True)
        self.assertEqual(code,2);self.assertEqual(r['status'],'FAIL')
        self.assertEqual(r['comparison']['rounds'][2]['status'],'PASS')


if __name__=='__main__':unittest.main()
