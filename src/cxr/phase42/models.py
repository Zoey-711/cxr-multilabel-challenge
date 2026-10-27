"""Frozen lightweight heads. Every added inference branch remains in the export."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth
from cxr.models.factory import create_model

IDS = ('A01', 'A02', 'A04', 'B01', 'B02', 'B03', 'B04', 'C01', 'C02',
       'C03', 'C04', 'D01', 'D02', 'D03', 'D04', 'E03', 'E04', 'F01', 'F04')


class GeM(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.p = nn.Parameter(torch.full((channels,), 3.))

    def forward(self, x):
        # Power reductions are FP32 even under AMP. Negative SiLU responses are
        # clamped, as declared in the frozen GeM specification.
        p = self.p.float().clamp(1., 6.).reshape(1, -1, 1, 1)
        return x.float().clamp_min(1e-6).pow(p).mean((-2, -1)).pow(1 / p.flatten())


class Candidate(nn.Module):
    def __init__(self, backbone, variant):
        super().__init__()
        self.backbone, self.variant = backbone, variant
        previous = backbone.training
        backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            channels = {}
            for i, stage in enumerate(backbone.features):
                x = stage(x)
                channels[i] = x.shape[1]
        backbone.train(previous)
        self.mid_stage = 5 if variant == 'A01' else 4
        self.mid_channels, self.final_channels = channels[self.mid_stage], channels[len(channels)-1]
        c, m = self.final_channels, self.mid_channels
        old = backbone.classifier[-1]
        self.dropout = backbone.classifier[0]
        backbone.classifier = nn.Identity()
        self.classifier = old
        if variant == 'A01':
            self.projection = nn.Linear(m, 64)
            self.gate = nn.Linear(c + 64, 64)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
            self.classifier = nn.Linear(c + 64, 10)
            with torch.no_grad():
                self.classifier.weight[:, :c].copy_(old.weight)
                self.classifier.bias.copy_(old.bias)
        elif variant == 'A02':
            self.mid_projection = nn.Conv2d(m, 64, 1)
            self.end_projection = nn.Conv2d(c, 64, 1)
            self.spatial = nn.Conv2d(64, 64, 3, padding=1, groups=64)
            self.classifier = nn.Linear(64, 10)
        elif variant == 'A04':
            self.mid_projection = nn.Conv2d(m, 64, 1)
            self.end_projection = nn.Conv2d(c, 64, 1)
            self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
            self.classifier = nn.Linear(c + 64, 10)
        elif variant == 'B01':
            self.pool = GeM()
        elif variant == 'B02':
            self.classifier = nn.Linear(c * 2, 10)
        elif variant == 'B03':
            self.classifier = nn.Linear(c * 5, 10)
        elif variant == 'B04':
            self.scores = nn.Conv2d(c, 1, 1)
        elif variant == 'C01':
            self.scores = nn.Conv2d(c, 10, 1)
        elif variant == 'C02':
            self.projection = nn.Conv2d(c, 64, 1)
            self.queries = nn.Parameter(torch.randn(10, 64) * .02)
            self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
            self.classifier = nn.Linear(64, 1)
            self.bias = nn.Parameter(torch.zeros(10))
        elif variant == 'C03':
            self.evidence = nn.Conv2d(c, 10, 1)
            self.pool = GeM(10)
            self.scale = nn.Parameter(torch.ones(10))
            self.bias = nn.Parameter(torch.zeros(10))
            self.classifier = nn.Identity()
        elif variant == 'C04':
            self.interaction = nn.Sequential(nn.Linear(10, 4), nn.Tanh(), nn.Linear(4, 10))
        elif variant == 'D01':
            self.spatial = nn.Conv2d(c, c, 7, padding=3, groups=c)
        elif variant == 'D02':
            self.channel_gate = nn.Sequential(nn.Linear(c, 80), nn.SiLU(), nn.Linear(80, c), nn.Sigmoid())
        elif variant == 'D03':
            self.channel_conv = nn.Conv1d(1, 1, 5, padding=2, bias=False)
        elif variant == 'D04':
            self.spatial = nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c)
        elif variant == 'F01':
            self.coord_reduce = nn.Conv2d(c, 40, 1)
            self.coord_h = nn.Conv2d(40, c, 1)
            self.coord_w = nn.Conv2d(40, c, 1)
        elif variant == 'F04':
            self.token_down = nn.Linear(144, 8, bias=False)
            self.token_up = nn.Linear(8, 144, bias=False)
        else:
            raise ValueError(variant)

    def forward(self, image):
        x, mid = image, None
        for i, stage in enumerate(self.backbone.features):
            x = stage(x)
            if i == self.mid_stage:
                mid = x
        v, gap = self.variant, x.mean((-2, -1))
        if v == 'A01':
            m = self.projection(mid.mean((-2, -1)))
            gate = torch.sigmoid(self.gate(torch.cat((gap, m), 1)))
            z = torch.cat((gap, m * gate), 1)
        elif v == 'A02':
            z = self.mid_projection(mid) + F.interpolate(self.end_projection(x), size=mid.shape[-2:], mode='bilinear', align_corners=False)
            z = (z + self.spatial(z)).mean((-2, -1))
        elif v == 'A04':
            # Fixed 4x4 grids bound the attention matrix independently of image size.
            q = F.adaptive_avg_pool2d(self.end_projection(x), 4).flatten(2).transpose(1, 2)
            kv = F.adaptive_avg_pool2d(self.mid_projection(mid), 4).flatten(2).transpose(1, 2)
            z = torch.cat((gap, self.attention(q, kv, kv, need_weights=False)[0].mean(1)), 1)
        elif v == 'B01': z = self.pool(x)
        elif v == 'B02': z = torch.cat((gap, x.amax((-2, -1))), 1)
        elif v == 'B03': z = torch.cat((gap, F.adaptive_avg_pool2d(x, 2).flatten(1)), 1)
        elif v == 'B04': z = (x.flatten(2) * self.scores(x).flatten(2).float().softmax(-1).to(x.dtype)).sum(-1)
        elif v == 'C01':
            a = self.scores(x).flatten(2).float().softmax(-1).to(x.dtype)
            z = torch.bmm(a, x.flatten(2).transpose(1, 2))
            return (self.dropout(z) * self.classifier.weight.unsqueeze(0)).sum(-1) + self.classifier.bias
        elif v == 'C02':
            tokens = self.projection(x).flatten(2).transpose(1, 2)
            q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            z = self.attention(q, tokens, tokens, need_weights=False)[0]
            return self.classifier(self.dropout(z)).squeeze(-1) + self.bias
        elif v == 'C03': return self.dropout(self.pool(F.softplus(self.evidence(x).float()))) * self.scale + self.bias
        elif v == 'C04':
            logits = self.classifier(self.dropout(gap))
            return logits + self.interaction(logits)
        elif v in ('D01', 'D04'): z = (x + F.silu(self.spatial(x))).mean((-2, -1))
        elif v == 'D02': z = gap * self.channel_gate(gap)
        elif v == 'D03': z = gap * torch.sigmoid(self.channel_conv(gap.unsqueeze(1))).squeeze(1)
        elif v == 'F01':
            h, w = x.shape[-2:]
            hpool, wpool = x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(2, 3)
            compressed = F.silu(self.coord_reduce(torch.cat((hpool, wpool), 2)))
            ah, aw = compressed.split((h, w), 2)
            z = (x * self.coord_h(ah).sigmoid() * self.coord_w(aw).sigmoid().transpose(2, 3)).mean((-2, -1))
        elif v == 'F04':
            if x.shape[-2:] != (12, 12):
                raise ValueError('F04 requires frozen 384 input / 12x12 final feature map')
            tokens = x.flatten(2)
            z = (tokens + self.token_up(F.gelu(self.token_down(tokens)))).mean(-1)
        else: raise ValueError(v)
        return self.classifier(self.dropout(z))


def build_model(variant, *, pretrained=False):
    if variant not in ('BASE', *IDS): raise ValueError(variant)
    backbone = create_model('efficientnet_v2_s', pretrained=pretrained)
    if variant in ('BASE', 'E03', 'E04'):
        if variant == 'E04':
            # Scale the existing linearly increasing torchvision schedule .2 -> .3.
            for module in backbone.modules():
                if isinstance(module, StochasticDepth): module.p *= 1.5
        return backbone
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4201)
        return Candidate(backbone, variant)


def parameter_groups(model):
    head = list(model.classifier.parameters()) if not isinstance(model, Candidate) else [
        p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    ids = {id(p) for p in head}
    return [p for p in model.parameters() if id(p) not in ids], head
