"""Elementwise multi-label losses. Apply the validity mask in the training loop."""

import torch
from torch.nn import functional as F


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    detach_focal_weight: bool = True,
) -> torch.Tensor:
    """ASL in float32 for stable FP16 AMP; return unreduced [batch, class] loss."""
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have the same [batch, class] shape")
    if gamma_neg < 0 or gamma_pos < 0 or not 0 <= clip < 1 or eps <= 0:
        raise ValueError("Invalid asymmetric loss parameters")
    x = logits.float()
    y = targets.float()
    if torch.any((y < 0) | (y > 1)):
        raise ValueError("ASL targets must be in [0, 1]")
    p_pos = torch.sigmoid(x)
    p_neg = (1.0 - p_pos + clip).clamp(max=1.0)
    log_likelihood = y * torch.log(p_pos.clamp(min=eps)) + (1.0 - y) * torch.log(p_neg.clamp(min=eps))
    focal_base = 1.0 - p_pos * y - p_neg * (1.0 - y)
    gamma = gamma_pos * y + gamma_neg * (1.0 - y)
    if detach_focal_weight:
        focal_base = focal_base.detach()
    return -log_likelihood * focal_base.clamp(min=0).pow(gamma)


def per_label_loss(logits: torch.Tensor, targets: torch.Tensor, training: dict) -> torch.Tensor:
    name = training["loss"]
    if name == "bce_with_logits":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if name == "asymmetric_loss":
        return asymmetric_loss_with_logits(logits, targets, **training["asl"])
    raise ValueError(f"Unsupported loss: {name}")
