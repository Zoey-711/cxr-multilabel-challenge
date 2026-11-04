"""H17's fixed hard/soft BCE, with explicit gradient-accumulation denominators."""
import torch
from torch.nn import functional as F


def step_group(model, ema, batches, optimizer, scheduler, scaler, device, diagnostic, hard_fraction=.5):
    if hard_fraction not in (.5, 1.): raise ValueError('Only frozen BCE/KD losses supported')
    optimizer.zero_grad(set_to_none=True)
    hard_den = sum(float(b['valid_mask'].sum()) for b in batches)
    soft_den = sum(float((b['valid_mask'] * b['soft_weight'][:, None]).sum()) for b in batches)
    if hard_den <= 0 or soft_den <= 0:
        raise ValueError('No valid hard/soft labels in accumulated group')
    hard_value = soft_value = 0.
    for batch in batches:
        x, y = (batch[k].to(device) for k in ('image', 'target'))
        q = batch['soft_target'].to(device) if hard_fraction < 1 else y
        mask = batch['valid_mask'].to(device).float()
        w = batch['soft_weight'].to(device).float()
        if y.shape != q.shape or y.ndim != 2 or y.shape[1] != 10 or w.shape != (len(y),):
            raise ValueError('Teacher targets must be explicitly joined by Study; ten classes required')
        if any(not bool(torch.isfinite(t).all()) for t in (x, y, q, w)) or bool(((q<0)|(q>1)).any()) or bool((w<=0).any()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite/invalid inputs, targets or Study weights')
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x)
            hard = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction='none') * mask).sum()/hard_den
            soft = (F.binary_cross_entropy_with_logits(logits.float(), q, reduction='none') * mask * w[:, None]).sum()/soft_den if hard_fraction < 1 else logits.new_zeros(())
            loss = hard_fraction*hard + (1-hard_fraction)*soft
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            diagnostic.update(stage='forward', logits_finite=bool(torch.isfinite(logits).all()), loss_finite=bool(torch.isfinite(loss)))
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite student forward/loss')
        scaler.scale(loss).backward()
        hard_value += float(hard.detach()); soft_value += float(soft.detach())
    scaler.unscale_(optimizer)
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    before = float(scaler.get_scale())
    if bad:
        diagnostic.update(stage='gradients', nonfinite_parameters=bad[:16], scaler_before=before)
        if not scaler.is_enabled():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite gradients without enabled GradScaler')
        scaler.step(optimizer); scaler.update()
        after = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        diagnostic.update(optimizer_skipped=True, scaler_after=after)
        if after >= before:
            raise FloatingPointError('Nonfinite gradients without scaler backoff')
        return dict(hard=hard_value, soft=soft_value, total=hard_fraction*hard_value+(1-hard_fraction)*soft_value, skipped=True)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    scaler.step(optimizer); scaler.update()
    skipped = float(scaler.get_scale()) < before
    if not skipped:
        scheduler.step(); ema.update_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    diagnostic.update(optimizer_skipped=skipped, scaler_before=before,
                      scaler_after=float(scaler.get_scale()), gradient_norm=float(norm))
    return dict(hard=hard_value, soft=soft_value, total=hard_fraction*hard_value+(1-hard_fraction)*soft_value, skipped=skipped)
