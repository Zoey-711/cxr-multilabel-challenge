"""Permutation invariant, padding-aware study classifiers for frozen image features."""

import torch
from torch import nn


class StudyPool(nn.Module):
    def __init__(self, kind: str, feature_dim: int = 768):
        super().__init__()
        if kind not in {"v00", "v01", "v02"}:
            raise ValueError(kind)
        self.kind = kind
        self.classifier = nn.Linear(feature_dim, 10)
        if kind == "v01":
            self.gate_tanh = nn.Linear(feature_dim, 128)
            self.gate_sigmoid = nn.Linear(feature_dim, 128)
            self.attention = nn.Linear(128, 1)
        elif kind == "v02":
            self.project_in = nn.Linear(feature_dim, 128)
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1,
                                               activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.project_out = nn.Linear(128, feature_dim)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or valid.shape != features.shape[:2]:
            raise ValueError("Expected [batch, images, features] and [batch, images] mask")
        if not valid.any(dim=1).all():
            raise ValueError("Every study needs at least one real image")
        mask = valid.unsqueeze(-1)
        if self.kind == "v01":
            scores = self.attention(torch.tanh(self.gate_tanh(features)) *
                                    torch.sigmoid(self.gate_sigmoid(features))).squeeze(-1)
            weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(dim=1)
            pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        elif self.kind == "v02":
            encoded = self.encoder(self.project_in(features), src_key_padding_mask=~valid)
            pooled = self.project_out((encoded * mask).sum(dim=1) / mask.sum(dim=1))
        else:
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1)
        return self.classifier(pooled)
