"""N01 classifier shape and fixed EMA update behavior."""

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.models.factory import create_model, imagenet_weight_info


def test_efficientnet_v2_s_has_ten_logit_head_and_official_weight_name():
    model = create_model("efficientnet_v2_s", pretrained=False)
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 10
    assert imagenet_weight_info("efficientnet_v2_s")[2] == "efficientnet_v2_s-dd5fe13b.pth"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="N01 training and EMA require CUDA")
def test_ema_averages_parameters_and_batchnorm_buffers_on_cuda():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1)).cuda()
    averaged = AveragedModel(model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(0.5),
                            use_buffers=True)
    with torch.no_grad():
        model[0].running_mean.fill_(2.0)
        model[1].weight.fill_(2.0)
        averaged.update_parameters(model)
        model[0].running_mean.fill_(4.0)
        model[1].weight.fill_(4.0)
        averaged.update_parameters(model)
    assert averaged.n_averaged.item() == 2
    assert torch.allclose(averaged.module[0].running_mean, torch.full((2,), 3.0, device="cuda"))
    assert torch.allclose(averaged.module[1].weight, torch.full((1, 2), 3.0, device="cuda"))
