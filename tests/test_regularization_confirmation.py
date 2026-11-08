"""Regression tests for paired RNG, accumulation and full optimizer recovery."""
import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from cxr.regularization_confirmation.core import MixRNG, update, state, restore
from cxr.engine.early_stopping import EarlyStopTracker
from cxr.phase39_support import seed_all
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
import json
import os
import shutil
import uuid
from unittest.mock import patch
from PIL import Image
from cxr.data.dataset import augment_train_image
from cxr.regularization_confirmation.core import AUG, loader
from cxr.regularization_confirmation.support import ROOT,task_specs,read_config,require_confirmation_gate
from cxr.regularization_confirmation.analysis import compare
from cxr.regularization_confirmation.delivery import export_light
from cxr.phase39_support import digest,write_json


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

    def test_nonfinite_resume_refused(self):
        objects=self.objects();rng=MixRNG()
        tr=SimpleNamespace(generator=torch.Generator());ev=SimpleNamespace(generator=torch.Generator())
        saved=state('test',0,objects,EarlyStopTracker(6),[],tr,ev,rng)
        saved['model']['weight'][0,0]=float('nan')
        with self.assertRaises(FloatingPointError):restore(saved,objects,tr,ev,rng)

    def test_no_photo_preserves_draws_and_identity_pixels(self):
        image=Image.fromarray(np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3))
        settings=dict(AUG,affine_probability=0,photometric_probability=1)
        torch.manual_seed(91);augment_train_image(image,settings);state1=torch.get_rng_state()
        torch.manual_seed(91);result=augment_train_image(image,dict(settings,brightness_contrast_fraction=0));state2=torch.get_rng_state()
        self.assertTrue(torch.equal(state1,state2))
        self.assertTrue(np.array_equal(np.asarray(image),np.asarray(result)))

    def test_no_photo_loader_changes_only_amplitude(self):
        cfg=dict(archive='dummy',micro_batch=8,workers=0,input_size=384)
        with patch('cxr.regularization_confirmation.core.make_dataloader') as make:
            loader(cfg,[],True,'BASE');base=make.call_args.kwargs
            loader(cfg,[],True,'NO_PHOTO');new=make.call_args.kwargs
        expected=copy.deepcopy(base);expected['augmentation']['brightness_contrast_fraction']=0.
        self.assertEqual(expected,new)

    def test_tasks_and_confirmation_folds(self):
        cfg=dict(stage='pilot',pair_gpu_hours=20)
        tasks=task_specs(cfg);self.assertEqual([(t['variant'],t['fold']) for t in tasks],[('MIXUP',2),('NO_PHOTO',3)])
        self.assertEqual([t['fold'] for t in task_specs(dict(cfg,stage='mixup_confirmation'))],[3,4,5])
        self.assertEqual([t['fold'] for t in task_specs(dict(cfg,stage='no_photo_confirmation'))],[4,5])
        with self.assertRaises(ValueError):task_specs(dict(cfg,pilot_pairs=['MIXUP_F02','MIXUP_F02']))

    def test_identical_predictions_bootstrap_zero(self):
        y=np.tile(np.array([[0]*10,[1]*10]),(10,1));p=.1+.8*y
        result=compare(y,p,p,[f'p{i//2}' for i in range(len(y))],draws=50)
        self.assertEqual(result['delta'],0)
        self.assertEqual(result['ci95'],[0,0])
        self.assertEqual(result['valid_replicates'],50)

    def test_actual_effnet_cpu_backward(self):
        from cxr.models.factory import create_model
        torch.set_num_threads(2);model=create_model('efficientnet_v2_s',pretrained=False)
        y=model(torch.rand(2,3,64,64));self.assertEqual(tuple(y.shape),(2,10))
        torch.nn.functional.binary_cross_entropy_with_logits(y,torch.zeros_like(y)).backward()
        self.assertGreater(float(model.features[0][0].weight.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(model.classifier[-1].weight.grad).all()))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'.tmp'/('confirmation_test_'+uuid.uuid4().hex);self.root.mkdir(parents=True)
    def tearDown(self):
        assert (ROOT/'.tmp').resolve() in self.root.resolve().parents
        shutil.rmtree(self.root)
    def test_export_redacts_paths_and_preserves_hashes(self):
        task='MIXUP_F02';src=self.root/'experiments'/task;src.mkdir(parents=True)
        top=['protocol_manifest.json','bootstrap_comparison.json','per_class_comparison.csv','final_decision.md','cpu_smoke_report.json','cuda_smoke_report.json','runtime_environment.json']
        for n in top:(src/n).write_text('{}',encoding='utf-8')
        for arm in ('BASE','MIXUP'):
            folder=src/arm;folder.mkdir()
            for n in ['resolved_config.json','train_history.csv','outer_oof_predictions.csv','outer_metrics.json','TRAIN_DONE.json']:
                (folder/n).write_text(json.dumps({'archive':'/private/data/file.zip'}) if n=='resolved_config.json' else '{}',encoding='utf-8')
            (folder/'ema_best.pth').write_bytes(b'synthetic weight')
        write_json(src/'DONE.json',dict(code_commit='a'*40,identity_sha256='b'*64,sha256={p.relative_to(src).as_posix():digest(p) for p in src.rglob('*') if p.is_file()}))
        report=export_light(self.root,task)
        target=self.root/'exports'/task
        self.assertEqual(json.loads((target/'BASE/resolved_config.json').read_text())['archive'],'REDACTED_SERVER_PATH')
        self.assertFalse(list(target.rglob('*.pth')))
        self.assertEqual(report['weights_status'],'WEIGHTS_NOT_DOWNLOADED')
        for rel,sha in report['sha256'].items():self.assertEqual(digest(target/rel),sha)
        self.assertEqual(export_light(self.root,task),report)
        (src/'BASE/outer_oof_predictions.csv').write_text('tampered')
        with self.assertRaises(ValueError):export_light(self.root,task)
    def test_confirmation_without_pass_is_blocked(self):
        with self.assertRaises(FileNotFoundError):require_confirmation_gate(dict(stage='mixup_confirmation',pilot_results_root=str(self.root)))
    def test_placeholder_paths_refused(self):
        cfg=ROOT/'configs/confirmation_server.example.yaml'
        with self.assertRaises(ValueError):read_config(cfg)
    def test_existing_queue_failure_isolation_and_completed_resume(self):
        from cxr.phase42.queue import run
        tasks=[dict(id=name,variant='MIXUP',fold=2,timeout_hours=1,max_retries=0) for name in ('PAIR_GOOD','PAIR_FAIL')]
        cfg=dict(output_root=str(self.root),pilot_gpu_hours=2,max_concurrent=2,gpu_ids=[0,1],upload={'enabled':False})
        def fake(task,root,config,gpu,identity,attempt):
            if task['id']=='PAIR_FAIL':return dict(state='FAILED',reason='synthetic controlled failure',retryable=False,gpu_hours=.001)
            dest=root/'experiments'/task['id'];dest.mkdir(parents=True)
            (dest/'metric.json').write_text('{}')
            write_json(dest/'DONE.json',dict(identity_sha256=identity,sha256={'metric.json':digest(dest/'metric.json')}))
            return dict(state='DONE',gpu_hours=.001)
        state=run(tasks,cfg,'unused','test_identity','pilot',executor=fake,poll=.001)
        self.assertEqual(state['tasks']['PAIR_GOOD']['state'],'DONE')
        self.assertEqual(state['tasks']['PAIR_FAIL']['state'],'FAILED')
        def forbidden(*args):raise AssertionError('Completed/failed jobs must not automatically rerun')
        resumed=run(tasks,cfg,'unused','test_identity','pilot',executor=forbidden,poll=.001)
        self.assertEqual(resumed['tasks']['PAIR_GOOD']['attempts'],1)
    def test_queue_budget_does_not_launch_unfunded_task(self):
        from cxr.phase42.queue import run
        task=dict(id='PAIR',variant='MIXUP',fold=2,timeout_hours=2,max_retries=0)
        cfg=dict(output_root=str(self.root),pilot_gpu_hours=1,max_concurrent=1,gpu_ids=[0],upload={'enabled':False})
        def forbidden(*args):raise AssertionError('Unfunded GPU job launched')
        state=run([task],cfg,'unused','test_identity','pilot',executor=forbidden,poll=.001)
        self.assertEqual(state['tasks']['PAIR']['state'],'PENDING')
        self.assertEqual(state['tasks']['PAIR']['attempts'],0)


if __name__=='__main__':unittest.main()
