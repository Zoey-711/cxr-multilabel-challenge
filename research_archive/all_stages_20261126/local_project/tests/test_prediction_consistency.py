"""Regression cases for false-positive official consistency checks (CPU only)."""
import copy
import csv
import io
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from cxr.prediction_consistency import (
    SUBMISSION_COLUMNS, aggregate_complete, compare_rounds, compare_studies,
    read_submission, validate_index,
)


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(image=n,Subject_id='p1',Study_id='s1',study_image_count=2) for n in ['a.jpg','b.jpg']]
        self.p=np.linspace(.05,.95,10)
        self.values={'a.jpg':self.p.copy(),'b.jpg':self.p.copy()}
        self.ref={('p1','s1'):self.p.copy()}

    def test_exact_match(self):
        r,_=compare_studies(self.ref,self.ref)
        self.assertEqual(r['status'],'PASS')

    def test_small_error_with_order_and_gate_preserved(self):
        r,_=compare_studies({('p1','s1'):self.p+1e-5},self.ref)
        self.assertEqual(r['status'],'PASS')

    def test_error_over_tolerance_fails(self):
        r,_=compare_studies({('p1','s1'):self.p+.002},self.ref)
        self.assertEqual(r['status'],'FAIL')

    def test_rank_flip_below_tolerance_fails(self):
        a=self.p.copy();a[:2]=[.1001,.1002]
        b=a.copy();b[:2]=a[:2][::-1]
        r,_=compare_studies({('p1','s1'):a},{('p1','s1'):b})
        self.assertEqual(r['ranking_changed_studies'],1)
        self.assertEqual(r['status'],'FAIL')

    def test_gate_flip_below_tolerance_fails(self):
        a=self.p.copy();a[4]=.49999
        b=a.copy();b[4]=.50001
        r,_=compare_studies({('p1','s1'):a},{('p1','s1'):b})
        self.assertEqual(r['gate_changed_studies'],1)
        self.assertEqual(r['status'],'FAIL')

    def test_missing_study_rejected(self):
        with self.assertRaises(ValueError): compare_studies(self.ref,{**self.ref,('p2','s2'):self.p})

    def test_missing_companion_rejected(self):
        with self.assertRaises(ValueError): aggregate_complete(self.rows,{'a.jpg':self.p})

    def test_false_study_count_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['study_image_count']=1
        with self.assertRaises(ValueError): validate_index(rows)

    def test_duplicate_image_rejected(self):
        with self.assertRaises(ValueError): validate_index(self.rows+[self.rows[0]])

    def test_image_mean_not_single_image(self):
        vals={'a.jpg':self.p-.01,'b.jpg':self.p+.01}
        result=aggregate_complete(self.rows,vals)
        self.assertTrue(np.allclose(result[('p1','s1')],self.p))

    def test_first_round_failure_cannot_hide_behind_last_round(self):
        rounds=[copy.deepcopy(self.values) for _ in range(3)]
        rounds[0]['a.jpg']+=.004
        report=compare_rounds(self.rows,rounds,self.ref,['a.jpg','b.jpg'])
        self.assertEqual(report['status'],'FAIL')
        self.assertEqual(report['rounds'][0]['inconsistent_timed_images'],2)
        self.assertEqual(report['rounds'][2]['status'],'PASS')

    def test_repeatability_separate_from_historical_tolerance(self):
        rounds=[copy.deepcopy(self.values) for _ in range(3)]
        rounds[1]['a.jpg']+=1e-7
        report=compare_rounds(self.rows,rounds,self.ref,['a.jpg','b.jpg'])
        self.assertEqual(report['historical_submission_consistency'],'PASS')
        self.assertEqual(report['internal_repeatability'],'FAIL')

    def test_identical_rounds_pass(self):
        report=compare_rounds(self.rows,[copy.deepcopy(self.values) for _ in range(3)],self.ref,['a.jpg','b.jpg'])
        self.assertEqual(report['status'],'PASS')

    def test_nan_rejected(self):
        p=self.p.copy();p[0]=np.nan
        with self.assertRaises(ValueError): compare_studies({('p1','s1'):p},self.ref)

    def test_missing_class_and_duplicate_submission_rejected(self):
        for labels in [range(1,10),[1,1,*range(3,11)]]:
            stream=io.StringIO()
            writer=csv.writer(stream);writer.writerow(SUBMISSION_COLUMNS)
            for j in labels:writer.writerow(['p1','s1',j,.1])
            stream.seek(0)
            with patch.object(Path,'open',return_value=stream):
                with self.assertRaises(ValueError):read_submission(Path('fixture.csv'),expected_studies=1)


if __name__=='__main__':unittest.main()
