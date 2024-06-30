import torch
from torch import nn
from model.network.backbone.transformer import Transformer


class GatedFusion(nn.Module):
    def __init__(self, mode: str, in_channel: int, out_channel: int):
        super().__init__()
        self.mode = mode
        self.map = nn.Sequential(nn.Linear(in_channel, in_channel),
                                 nn.Sigmoid())
        self.bottleneck = nn.Sequential(nn.Linear(in_channel, out_channel),
                                        nn.LeakyReLU(inplace=True))

    def forward(self, x: dict):
        feature = torch.cat([x[m] for m in self.mode], dim=1)
        weights = self.map(feature)
        out = self.bottleneck(weights * feature)
        return out


class TransformFusion(nn.Module):
    def __init__(self, mode: str, num_heads: int, in_channel: int):
        super().__init__()
        self.mode = mode
        self.attention = Transformer(in_channel, num_heads=num_heads)
        self.token = nn.Parameter(torch.zeros(1, 1, in_channel), requires_grad=True)

    def forward(self, x: dict):
        x = torch.cat([x[m].unsqueeze(1) for m in self.mode] + [self.token], dim=1)
        output = self.attention(x)[:, -1]
        return output


class Concat(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x: dict):
        feature = torch.cat([x[m] for m in self.mode], dim=1)
        return feature


class WeightedConcat(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        num_ele = len(mode)
        self.weight = nn.Parameter(torch.zeros(num_ele, dtype=torch.float), requires_grad=True)

    def forward(self, x: dict):
        weights = nn.functional.softmax(self.weight, dim=0)
        feature = torch.cat([x[m] * weights[i] for i, m in enumerate(self.mode)], dim=1)

        return feature


class FusionModule(nn.Module):
    def __init__(self, mode: str, in_channel: int, num_class: int, fusion_way='gate_fusion', clf_way='mlp',
                 num_heads: int = 4):
        super().__init__()
        self.mode = mode
        self.fusion_way = fusion_way
        self.clf_way = clf_way
        self.out_channel = 0
        print(fusion_way)
        if fusion_way == 'concat':
            self.feature_fusion = Concat(mode)
            self.out_channel = in_channel

        elif fusion_way == 'transform':
            self.feature_fusion = TransformFusion(mode, num_heads, in_channel)
            self.out_channel = in_channel

        elif fusion_way == 'gate_fusion':
            self.feature_fusion = GatedFusion(mode, in_channel, in_channel // 2)
            self.out_channel = in_channel // 2

        elif fusion_way == 'weighted_concat':
            self.feature_fusion = WeightedConcat(mode)
            self.out_channel = in_channel

        else:
            raise ValueError('Unknown fusion way %s!' % self.fusion_way)

        if clf_way == 'mlp':
            self.clf = nn.Sequential(nn.Linear(self.out_channel, 32),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(32, num_class))

        elif clf_way == 'linear':
            self.clf = nn.Sequential(nn.Linear(self.out_channel, num_class))
        else:
            raise ValueError('Unknown clf way %s!' % clf_way)

    def parse_feature(self, x: dict):
        return self.feature_fusion(x)

    def forward(self, x: dict):
        feature = self.parse_feature(x)
        out = self.clf(feature)
        return out
