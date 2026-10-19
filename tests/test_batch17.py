import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cxr.batch.plan import ROOT,load_plan,digest,gpu_tokens,retryable,deployment_bytes,write_json
from cxr.batch.results import align,combine,concatenate_oof,bootstrap,scores,grouped
from cxr.batch.backup import backup,restore
from cxr.batch.runner import budget_allows


def table(n=40):
    rng=np.random.default_rng(42)
    y=np.tile(np.arange(n)[:,None]%2,(1,10))
    return {'keys':[(f'{i//2:04d}',f'{i:06d}') for i in range(n)],'y':y,
            'p':rng.random((n,10)), 'counts':np.where(np.arange(n)%3==0,1,2)}


def test_plan_fifteen_tasks_fixed_formulas_and_two_disabled_seeds():
    plan=load_plan(ROOT/'configs/batch_plan.yaml')
    assert len([t for t in plan['tasks'] if t['enabled']])==15
    assert len([t for t in plan['tasks'] if not t['enabled']])==2
    assert plan['fixed_fusions']['E17']=={'D5':.25,'C5':.25,'S5':.5}
    assert plan['optional_architectures']['tasks']==[]


def test_subject_alignment_leading_zeros_and_labels():
    a=table(); b={**a,'keys':a['keys'][::-1],**{k:a[k][::-1] for k in ('p','y','counts')}}
    assert np.array_equal(align(a,b),a['p'])
    assert a['keys'][0]==('0000','000000')
    b=copy.deepcopy(a); b['keys'][0]=('wrong','000000')
    with pytest.raises(ValueError): align(a,b)
    b=copy.deepcopy(a); b['y'][0,0]=1
    with pytest.raises(ValueError): align(a,b)


def test_fixed_fusion_and_oof_no_seen_patient_average():
    a=table(); b={**a,'p':1-a['p']}
    assert np.allclose(combine([a,b],[.5,.5])['p'],.5)
    with pytest.raises(ValueError): combine([a,b],[.5,.6])
    with pytest.raises(ValueError): concatenate_oof([a,b])
    b={**b,'keys':[('new'+s,'new'+st) for s,st in b['keys']]}
    assert len(concatenate_oof([a,b])['keys'])==80


def test_bootstrap_reproducible_paired_and_support():
    a=table(); b={**a,'p':a['p']*.9+.05}
    result=bootstrap({'a':a,'b':b},[('b','a')],draws=15)
    assert result==bootstrap({'a':a,'b':b},[('b','a')],draws=15)
    assert result['differences']['b_minus_a']['ci95']==[0,0]
    assert grouped(a)['single_image']['studies']+grouped(a)['multi_image']['studies']==40
    assert len(scores(a)['per_class'])==10


def test_bootstrap_single_class_draws_counted():
    a=table(6); a['keys']=[(str(i),str(i)) for i in range(6)]
    a['y'][:]=0; a['y'][0,:]=1
    result=bootstrap({'a':a},[],draws=20)
    assert result['invalid_draws']>0
    assert result['valid_draws']==20


def test_budget_allocation_oom_retry_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,4')
    assert gpu_tokens('4')==['4']
    with pytest.raises(ValueError): gpu_tokens('0')
    with pytest.raises(ValueError): gpu_tokens('2,2')
    assert budget_allows(2,3,4,9)
    assert not budget_allows(2,3,4,8.9)
    assert retryable('Stale file handle',1)
    assert not retryable('CUDA out of memory',75)
    assert not retryable('Non-finite gradients',75)
    assert not retryable('ValueError label alignment',1)


def test_backup_restore_hashes_and_no_overwrite(tmp_path):
    source=tmp_path/'source'; source.mkdir(); (source/'weights.pt').write_bytes(b'fake-state-dict')
    target=tmp_path/'backup'; backup(source,target)
    restored=tmp_path/'restored'; restore(target,restored)
    assert digest(source/'weights.pt')==digest(restored/'weights.pt')
    with pytest.raises(FileExistsError): restore(target,restored)
    with pytest.raises(FileExistsError): backup(source,target)
    (target/'weights.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): restore(target,tmp_path/'bad')


def test_model_budget_deduplicates_and_boundary(tmp_path):
    path=tmp_path/'state.pt'; path.write_bytes(b'12345')
    assert deployment_bytes([path,path],5)['fits']
    assert not deployment_bytes([path],4)['fits']


def test_queue_defaults_to_no_training_and_safe_config_paths():
    plan=load_plan(ROOT/'configs/batch_plan.yaml')
    import yaml
    for t in plan['tasks']:
        c=yaml.safe_load((ROOT/t['config']).read_text())
        assert c['data']['archive_path'] is None
        assert c['training']['loss']=='bce_with_logits'
        assert c['validation']['study_fusion']=='mean_probability'
        assert c['data']['input_size']==384
        assert c['training']['early_stopping_patience']==(8 if t['backbone']=='densenet121' else 6)


def test_batch_snapshot_only_declared_training_changes():
    # Source is intentionally not edited while local training uses it.
    text=(ROOT/'scripts/batch_train.py').read_text()
    assert 'best_artifacts' in text and 'restore_rng_state(state, train_loader, val_loader)' in text
    assert 'scaler.load_state_dict' in text and 'scheduler.load_state_dict' in text
    assert 'model_name != "convnext_tiny"' in text
    assert 'ema_model.update_parameters(model)' in text
    assert 'float(ema_settings.get("decay", 0)) != 0.999' in text
