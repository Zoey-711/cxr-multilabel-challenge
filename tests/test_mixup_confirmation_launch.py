import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch', ROOT/'scripts/prepare_mixup_confirmation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from cxr.regularization_confirmation.support import task_specs


class LaunchTests(unittest.TestCase):
    def config(self):
        return dict(stage='pilot', output_root='/data/pilot', gpu_ids=[0,1],
                    pair_gpu_hours=20, pilot_pairs=['MIXUP_F02','NO_PHOTO_F03'],
                    upload=dict(enabled=True))

    def test_exactly_six_runs_and_separate_output(self):
        original=self.config();cfg=module.derive_config(original,[0,1,2])
        self.assertEqual([t['id'] for t in task_specs(cfg)],['MIXUP_F03','MIXUP_F04','MIXUP_F05'])
        self.assertTrue(all(t['variant']=='MIXUP' for t in task_specs(cfg)))
        self.assertEqual(cfg['gpu_hours'],60)
        self.assertEqual(cfg['max_concurrent'],3)
        self.assertNotEqual(cfg['output_root'],original['output_root'])
        self.assertEqual(cfg['pilot_results_root'],original['output_root'])
        self.assertTrue(original['upload']['enabled'])
        self.assertFalse(cfg['upload']['enabled'])

    def test_one_gpu_keeps_all_folds(self):
        cfg=module.derive_config(self.config(),[2])
        self.assertEqual(cfg['max_concurrent'],1)
        self.assertEqual(len(task_specs(cfg)),3)

    def test_reject_invalid_gpu_budget_and_stage(self):
        for ids in ([],[0,0],[-1]):
            with self.assertRaises(ValueError):module.derive_config(self.config(),ids)
        for budget in (0,-1,float('nan')):
            with self.assertRaises(ValueError):module.derive_config(self.config(),pair_hours=budget)
        cfg=self.config();cfg['stage']='mixup_confirmation'
        with self.assertRaises(ValueError):module.derive_config(cfg)


if __name__=='__main__':unittest.main()
