import copy
import pytest
from cxr.batch.deployment import assess, PROTOCOL_FIELDS


def evidence():
    c={'candidate':'test','dependencies':[{'bytes':500_000_000,'sha256':'weights'}],
       'val_auc':.8,'inference_recipe_sha256':'recipe','accuracy_recipe_sha256':'recipe',
       'accuracy_checkpoint_sha256':['weights']}
    p={k:'official-defined' for k in PROTOCOL_FIELDS}
    p.update(confirmed=True,official_source='provided rule',sha256='protocol')
    m={**p,'protocol_sha256':'protocol','checkpoint_sha256':['weights'],
       'inference_recipe_sha256':'recipe','full_pipeline':True,'latency_ms':99.999}
    return c,p,m


def test_strict_latency_and_inclusive_size():
    c,p,m=evidence()
    assert assess(c,p,m)['status']=='PASS'
    for latency in (100.,100.001):
        m['latency_ms']=latency
        assert assess(c,p,m)['status']=='LATENCY_FAIL'
    c['dependencies'][0]['bytes']+=1
    assert assess(c,p,m)['status']=='SIZE_FAIL'


@pytest.mark.parametrize('change', ['hardware','precision','statistic','timing_scope','batch_size','software_environment'])
def test_wrong_environment_is_not_compliant(change):
    c,p,m=evidence();m[change]='A100 or nonofficial protocol'
    assert assess(c,p,m)['status']=='LATENCY_UNVERIFIED'


def test_missing_measurement_full_ensemble_and_accuracy():
    c,p,m=evidence()
    assert assess(c,p)['status']=='LATENCY_UNVERIFIED'
    m['full_pipeline']=False
    assert assess(c,p,m)['status']=='LATENCY_UNVERIFIED'
    m['full_pipeline']=True;m['checkpoint_sha256']=[]
    assert assess(c,p,m)['status']=='LATENCY_UNVERIFIED'
    c['accuracy_recipe_sha256']='old_fp32_not_current_quantized'
    assert assess(c,p,m)['status']=='ACCURACY_PENDING'
    c,p,m=evidence();m['latency_ms']=float('nan')
    assert assess(c,p,m)['status']=='LATENCY_UNVERIFIED'


def test_original_configs_and_conditional_fail_closed():
    from cxr.batch.plan import ROOT,load_plan
    p=load_plan(ROOT/'configs/batch_plan.yaml')
    assert len(p['original_A_config_hashes'])==15
    assert len(p['tasks'])==22
    assert len([t for t in p['tasks'] if t['group']=='N' and t['ema']])==5
    assert len([t for t in p['tasks'] if t['group']=='P' and t['fold']==1])==2
    assert p['conditional_extension']['enabled'] is False
    assert p['conditional_extension']['criteria'] is None
    assert len(p['conditional_extension']['reserved_slots'])==4


def test_n01_ema_buffer_semantics_cpu():
    import torch
    from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
    model=torch.nn.BatchNorm1d(2)
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999),use_buffers=True)
    with torch.no_grad():
        model.running_mean.fill_(1);model.num_batches_tracked.fill_(1)
    ema.update_parameters(model)
    with torch.no_grad():
        model.running_mean.fill_(3);model.num_batches_tracked.fill_(2)
    ema.update_parameters(model)
    assert torch.allclose(ema.module.running_mean,torch.full((2,),1.002))
    # N01 uses torch's arithmetic then casts integer buffers back; no BN recalibration.
    assert ema.module.num_batches_tracked.item()==1


def test_predeclared_full_inference_formula():
    from scripts.batch_ensemble_latency import model_weights
    from cxr.batch.plan import ROOT,load_plan
    plan=load_plan(ROOT/'configs/batch_plan.yaml')
    assert model_weights('E17_F01',plan)=={'D17_F01':.25,'C17_F01':.25,'S17_F01':.5}
    result=model_weights('D5_C1_S1',plan)
    assert all(result[f'D17_F{i:02d}']==.05 for i in range(1,6))
    assert result['C17_F01']==.25 and result['S17_F01']==.5
    assert abs(sum(result.values())-1)<1e-12
