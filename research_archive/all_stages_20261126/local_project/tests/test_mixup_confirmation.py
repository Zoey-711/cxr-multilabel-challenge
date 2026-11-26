"""Regression tests for paired RNG, accumulation and full optimizer recovery."""
import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from cxr.mixup_confirmation import MixRNG, update, state, restore
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.phase34_aux import seed_all
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities


class ControlledTrainingTests(unittest.TestCase):
    def objects(self):
        seed_all(42)
        m=torch.nn.Linear(4,10)
        e=AveragedModel(m,multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
        o=torch.optim.AdamW(m.parameters(),lr=.01)
        s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:1/(step+1))
        g=torch.amp.GradScaler('cuda',enabled=False)
        return m,e,o,s,g
    def batch(self,n):
        return dict(image=torch.arange(n*4,dtype=torch.float32).reshape(n,4)/10,
                    target=torch.zeros(n,10),valid_mask=torch.ones(n,10,dtype=torch.bool))
    def test_mixup_rng_isolation(self):
        seed_all(42);t=torch.get_rng_state().clone();n=np.random.get_state()
        rng=MixRNG();x=torch.randn(4,4);t=torch.get_rng_state().clone()
        rng.apply(x,torch.eye(4))
        self.assertTrue(torch.equal(t,torch.get_rng_state()))
        self.assertTrue(np.array_equal(n[1],np.random.get_state()[1]))
    def test_accumulation_handles_short_final_microbatch(self):
        a=self.objects();b=self.objects();micro=[self.batch(2),self.batch(1)]
        full={k:torch.cat([v[k] for v in micro]) for k in micro[0]}
        update(micro,*a,MixRNG(),False,'cpu');update([full],*b,MixRNG(),False,'cpu')
        for pa,pb in zip(a[0].parameters(),b[0].parameters()):
            torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-7)
    def test_full_optimizer_ema_scheduler_and_mixup_resume(self):
        objects=self.objects();rng=MixRNG();tr=SimpleNamespace(generator=torch.Generator().manual_seed(42));ev=SimpleNamespace(generator=torch.Generator().manual_seed(43))
        micro=[self.batch(2),self.batch(1)]
        update(micro,*objects,rng,True,'cpu')
        tracker=EarlyStopTracker(6);tracker.observe(.7,1)
        with io.BytesIO() as p:
            torch.save(state('test',1,objects,tracker,[],tr,ev,rng),p)
            update(micro,*objects,rng,True,'cpu')
            expected=[copy.deepcopy(o.state_dict()) for o in objects[:2]]
            p.seek(0)
            saved=torch.load(p,map_location='cpu',weights_only=True)
            restored=self.objects();t=restore(saved,restored,tr,ev,rng)
            update(micro,*restored,rng,True,'cpu')
            for obj,ref in zip(restored[:2],expected):
                for k,v in obj.state_dict().items():self.assertTrue(torch.equal(v,ref[k]),k)
            self.assertEqual(restored[3].last_epoch,2)
            self.assertEqual(t.best_epoch,1)
    def test_early_stop_tie_keeps_earlier_checkpoint(self):
        t=EarlyStopTracker(2);self.assertTrue(t.observe(.7,1).new_best_checkpoint)
        self.assertFalse(t.observe(.7,2).new_best_checkpoint)
        self.assertTrue(t.observe(.69,3).should_stop)
        self.assertEqual(t.best_epoch,1)
    def test_study_mean(self):
        rows=[ImagePrediction('p','s','a',tuple([.2]*10)),ImagePrediction('p','s','b',tuple([.8]*10))]
        result=aggregate_mean_probabilities(rows)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].probabilities,tuple([.5]*10))


if __name__=='__main__':unittest.main()
