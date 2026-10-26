import pytest
import torch
from torch.nn import functional as F

from cxr.engine.losses import asymmetric_loss_with_logits, per_label_loss


def test_zero_gamma_zero_clip_matches_bce() -> None:
    logits = torch.tensor([[0.0, 1.5, -2.0], [2.0, -1.0, 0.4]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    actual = asymmetric_loss_with_logits(logits, targets, gamma_neg=0, gamma_pos=0, clip=0)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(actual, expected, atol=1e-6)


def test_asl_downweights_easy_negative_and_mask_excludes_unknown() -> None:
    logits = torch.tensor([[-5.0, 0.7]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0]])
    asl = asymmetric_loss_with_logits(logits, targets)
    bce = per_label_loss(logits, targets, {"loss": "bce_with_logits"})
    assert asl.shape == logits.shape
    assert asl[0, 0] < bce[0, 0]
    (asl * torch.tensor([[0.0, 1.0]])).sum().backward()
    assert logits.grad[0, 0] == 0
    assert torch.isfinite(logits.grad).all()


def test_asl_extreme_logits_have_finite_gradient() -> None:
    logits = torch.tensor([[-20.0, 20.0], [20.0, -20.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = asymmetric_loss_with_logits(logits, targets).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_asl_accumulation_uses_actual_valid_label_count() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.6, -0.8], [-0.2, 0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    params = {"loss": "asymmetric_loss", "asl": {"gamma_neg": 4.0, "gamma_pos": 1.0,
                                                   "clip": 0.05, "eps": 1e-8,
                                                   "detach_focal_weight": True}}
    complete = (per_label_loss(logits, targets, params) * mask).sum() / mask.sum()
    # The final accumulation group can contain fewer microbatches than planned.
    parts = [(per_label_loss(logits[:2], targets[:2], params) * mask[:2]).sum(),
             (per_label_loss(logits[2:], targets[2:], params) * mask[2:]).sum()]
    grouped = sum(parts) / mask.sum()
    assert torch.allclose(grouped, complete)
    grouped.backward()
    assert logits.grad[1, 1] == 0
    assert torch.isfinite(logits.grad).all()


def test_asl_matches_author_formula_and_detached_weight_gradient() -> None:
    x = torch.tensor([[-4.0, -1.0, 0.0, 2.0], [3.0, -2.0, 1.0, 0.2]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    pos = x.sigmoid()
    neg = (1 - pos + 0.05).clamp(max=1)
    log_terms = y * torch.log(pos.clamp(min=1e-8)) + (1-y) * torch.log(neg.clamp(min=1e-8))
    with torch.no_grad():
        focal = (1 - pos*y - neg*(1-y)).pow(y + 4*(1-y))
    reference = -log_terms * focal
    actual = asymmetric_loss_with_logits(x, y)
    assert torch.allclose(actual, reference, atol=1e-7)
    a = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(reference.sum(), x)[0]
    assert torch.allclose(a, b, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_asl_fp16_autocast_is_finite_on_cuda() -> None:
    logits = torch.tensor([[-20.0, 20.0]], device="cuda", requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = asymmetric_loss_with_logits(logits.half(), targets).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
