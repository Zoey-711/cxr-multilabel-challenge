"""CPU-only checks for fixed fusion, AUROC ties and patient multiplicities."""
import unittest
import numpy as np
from cxr.inference.e03_h17_validation import fuse, roc_auc_score, NAMES, compare
import json

class ReproductionTests(unittest.TestCase):
    def test_formula(self):
        r=np.random.default_rng(10)
        p={n:r.random((8,10)) for n in NAMES}
        expected=sum(p[n]/12 for n in NAMES[:3])+p['M02']/4+p['H17_BATCH1']/2
        np.testing.assert_allclose(fuse(p),expected,rtol=0,atol=2e-16)

    def test_study_mean_commutes(self):
        r=np.random.default_rng(11); p={n:r.random((7,10)) for n in NAMES}
        np.testing.assert_allclose(fuse(p).mean(0),fuse({n:v.mean(0) for n,v in p.items()}),rtol=0,atol=2e-16)

    def test_weighted_auc_against_pairwise_definition(self):
        r=np.random.default_rng(12)
        for _ in range(100):
            y=np.r_[np.zeros(10),np.ones(10)]; p=r.integers(0,5,20); w=r.integers(1,4,20)
            pos=p[y==1,None]; neg=p[y==0][None,:]; weights=w[y==1,None]*w[y==0][None,:]
            expected=np.sum(weights*((pos>neg)+.5*(pos==neg)))/weights.sum()
            self.assertAlmostEqual(roc_auc_score(y,p,sample_weight=w),expected,places=14)
            self.assertAlmostEqual(roc_auc_score(np.repeat(y,w),np.repeat(p,w)),expected,places=14)

    def test_macro_auc(self):
        y=np.array([[0,1],[1,0],[0,1],[1,0]])
        p=np.array([[.1,.9],[.8,.2],[.3,.7],[.6,.4]])
        self.assertEqual(roc_auc_score(y,p),1.)

    def test_comparison_json(self):
        y=np.tile(np.array([[0],[1]]),(1,10));p=y*.8+.1
        report=compare(p,p,y,[('s1','a'),('s2','b')],'synthetic')
        self.assertEqual(json.loads(json.dumps(report))['max_abs'],0.)

if __name__=='__main__': unittest.main()
