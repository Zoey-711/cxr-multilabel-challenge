"""Low-capacity correction of frozen Study mean-probability logits."""

import torch
from torch import nn


class MultiViewResidual(nn.Module):
    """Use cross-view feature variance; a single real view yields exactly zero."""

    def __init__(self, feature_dim: int = 768, dropout: float = 0.1, max_logit_delta: float = 0.25):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(feature_dim, 10, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.max_logit_delta = max_logit_delta

    def forward(self, features: torch.Tensor, valid: torch.Tensor,
                baseline_logits: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or valid.shape != features.shape[:2] or baseline_logits.shape != (len(features), 10):
            raise ValueError("Invalid feature, mask or baseline shape")
        if not valid.any(dim=1).all():
            raise ValueError("Every Study needs at least one real image")
        mask = valid.unsqueeze(-1)
        count = mask.sum(dim=1)
        mean = (features * mask).sum(dim=1) / count
        variance = (((features - mean.unsqueeze(1)) * mask).square()).sum(dim=1) / count
        raw_delta = self.projection(self.dropout(variance))
        delta = self.max_logit_delta * torch.tanh(raw_delta)
        return baseline_logits + delta
