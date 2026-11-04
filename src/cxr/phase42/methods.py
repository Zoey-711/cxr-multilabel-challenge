"""Optimizer grouping and the sole declared loss change (E03)."""
import math
import torch
from cxr.phase39_run import step_group as original_step
from cxr.phase42.models import parameter_groups


def optimizer_scheduler(cfg, model, steps):
    back,head=parameter_groups(model)
    optimizer=torch.optim.AdamW([dict(params=back,lr=cfg['backbone_lr']),dict(params=head,lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    warm,total=cfg['warmup_epochs']*steps,cfg['epochs']*steps
    def factor(step):
        if step<warm:return (step+1)/max(1,warm)
        return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def step_group(cfg,model,ema,micro,opt,sch,scaler,device,diagnostics=None):
    eps=cfg.get('label_smoothing',0.)
    if eps:
        if eps!=.05: raise ValueError('Frozen smoothing is .05')
        micro=[dict(batch,target=(1-eps)*batch['target']+eps*.5) for batch in micro]
    if diagnostics is None:
        return original_step(cfg,model,ema,micro,opt,sch,scaler,device)
    return original_step(cfg,model,ema,micro,opt,sch,scaler,device,diagnostics)
