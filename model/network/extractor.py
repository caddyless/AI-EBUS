import torch
from torch import nn

from model.network.backbone.resnet import ResExtractor
from model.network.backbone.seincept import SEExtractor
from model.network.backbone import MLP


class NoneModule(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.out_channel = in_channel

    def forward(self, x):
        return x


class Extractor(nn.Module):
    def __init__(self, method: str = 'seincept', params: dict = None):
        super().__init__()

        if method == 'seincept':
            self.extractor = SEExtractor(**params)
        elif method == 'resnet':
            self.extractor = ResExtractor(**params)
        elif method == 'linear':
            self.extractor = nn.Linear(**params)
        elif method == 'mlp':
            self.extractor = MLP(**params)
        elif method == 'none':
            self.extractor = NoneModule(**params)
        else:
            raise NotImplemented('The method %s has not been implemented yet!' % method)

        if method == 'linear':
            self.out_channel = params['out_features']
        else:
            self.out_channel = self.extractor.out_channel

    def forward(self, x: torch.Tensor):
        return self.extractor(x)
