"""Study pooling must ignore padding and accept one or many images."""

import pytest
import torch

from cxr.models.study_pool import StudyPool


@pytest.mark.parametrize("kind", ["v00", "v01", "v02"])
def test_padding_and_permutation(kind):
    torch.manual_seed(42)
    model = StudyPool(kind).eval()
    images = torch.randn(2, 3, 768)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    with torch.no_grad():
        original = model(images, mask)
        changed = images.clone()
        changed[~mask] = 1e5
        assert torch.allclose(original, model(changed, mask), atol=1e-5)
        reordered = images[:, [1, 0, 2]]
        reordered_mask = mask[:, [1, 0, 2]]
        assert torch.allclose(original, model(reordered, reordered_mask), atol=1e-5)
    assert original.shape == (2, 10)
    assert torch.isfinite(original).all()


@pytest.mark.parametrize("kind", ["v00", "v01", "v02"])
def test_gradient_and_empty_rejection(kind):
    model = StudyPool(kind)
    x = torch.randn(2, 2, 768)
    mask = torch.tensor([[True, False], [True, True]])
    loss = model(x, mask).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="at least one"):
        model(x, torch.zeros_like(mask))
