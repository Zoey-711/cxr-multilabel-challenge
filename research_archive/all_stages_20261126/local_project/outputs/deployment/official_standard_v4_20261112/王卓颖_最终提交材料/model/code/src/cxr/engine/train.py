"""Minimal training-step interface; dataset integration awaits metadata audit."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingBatch:
    """Images, ten targets, and an explicit validity mask for each label."""

    images: Any
    targets: Any
    label_mask: Any


def train_step(model: Any, batch: TrainingBatch, optimizer: Any, *, device: Any) -> float:
    """One masked BCE step. The caller decides label encoding after audit."""
    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training") from exc

    model.train()
    images = batch.images.to(device)
    targets = batch.targets.to(device)
    mask = batch.label_mask.to(device)
    if targets.ndim != 2 or targets.shape[1] != 10 or mask.shape != targets.shape:
        raise ValueError("Targets and label_mask must both have shape [batch, 10]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("label_mask must contain only 0 or 1")
    if not torch.all(torch.isfinite(targets[mask == 1])):
        raise ValueError("Valid targets must be finite")
    if not torch.all((targets[mask == 1] == 0) | (targets[mask == 1] == 1)):
        raise ValueError("Valid targets must be binary")
    valid_count = mask.sum()
    if valid_count.item() == 0:
        raise ValueError("Training batch contains no valid labels")
    logits = model(images)
    if logits.shape != targets.shape:
        raise ValueError("Model logits must have shape [batch, 10]")
    optimizer.zero_grad(set_to_none=True)
    safe_targets = torch.where(mask == 1, targets, torch.zeros_like(targets))
    per_label = functional.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    loss = (per_label * mask).sum() / valid_count
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())
