"""Phase39 single-image EfficientNetV2-S models; fusion remains at inference."""
import torch
from torch import nn
from torch.nn import functional as F

from cxr.models.factory import create_model


class MultiStageEfficientNet(nn.Module):
    def __init__(self, backbone: nn.Module, mid_stage: int = 5, projection_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.mid_stage = mid_stage
        # Infer channels from the actual installed torchvision graph, in eval mode.
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            mid = None
            for i, stage in enumerate(backbone.features):
                x = stage(x)
                if i == mid_stage:
                    mid = x.shape[1]
            end = x.shape[1]
        backbone.train(was_training)
        if mid is None or end != backbone.classifier[-1].in_features:
            raise ValueError('Unexpected EfficientNetV2-S feature stages')
        old = backbone.classifier[-1]
        self.mid_channels, self.final_channels = mid, end
        self.projection = nn.Linear(mid, projection_dim)
        self.classifier = nn.Linear(end + projection_dim, 10)
        with torch.no_grad():
            self.classifier.weight[:, :end].copy_(old.weight)
            self.classifier.bias.copy_(old.bias)
            nn.init.normal_(self.classifier.weight[:, end:], std=0.001)
        backbone.classifier = nn.Identity()

    def forward(self, image):
        x = image
        mid = None
        for i, stage in enumerate(self.backbone.features):
            x = stage(x)
            if i == self.mid_stage:
                mid = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if mid is None:
            raise ValueError('Intermediate stage missing')
        final = self.backbone.avgpool(x).flatten(1)
        return self.classifier(torch.cat((final, self.projection(mid)), dim=1))


def build_model(task: str, *, pretrained: bool):
    if task not in ('P39_BASE_F1', 'P39_FUSION_F1'):
        raise ValueError(task)
    backbone = create_model('efficientnet_v2_s', pretrained=pretrained)
    if task == 'P39_BASE_F1':
        return backbone
    # Building the new head must not alter the later DataLoader RNG sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3901)
        return MultiStageEfficientNet(backbone)
