"""Residual pooling preserves one-view predictions and ignores padding."""

import pytest
import torch

from cxr.models.study_residual import MultiViewResidual


def test_single_view_exact_baseline_and_padding():
    model = MultiViewResidual().eval()
    with torch.no_grad():
        model.projection.weight.fill_(0.2)
    features = torch.randn(2, 3, 768)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    baseline = torch.randn(2, 10)
    with torch.no_grad():
        result = model(features, mask, baseline)
        assert torch.equal(result[0], baseline[0])
        modified = features.clone()
        modified[~mask] = 1e6
        assert torch.allclose(result, model(modified, mask, baseline), atol=1e-6)
        assert torch.allclose(result, model(features[:, [1, 0, 2]], mask[:, [1, 0, 2]], baseline), atol=1e-6)
        assert torch.max(torch.abs(result - baseline)) <= 0.25 + 1e-6


def test_zero_initialization_and_multiview_gradient():
    model = MultiViewResidual()
    x = torch.randn(2, 2, 768)
    baseline = torch.zeros(2, 10)
    mask = torch.ones(2, 2, dtype=torch.bool)
    result = model(x, mask, baseline)
    assert torch.equal(result, baseline)
    result.sum().backward()
    assert torch.isfinite(model.projection.weight.grad).all()
    assert model.projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model(x, torch.zeros_like(mask), baseline)
