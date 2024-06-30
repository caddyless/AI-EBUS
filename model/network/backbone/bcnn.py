"""Fine-tune all layers for bilinear CNN.
Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_all.py --base_lr 0.05 \
        --batch_size 64 --epochs 100 --weight_decay 5e-4
"""

import torch
import torch.nn as nn
from model.network.backbone.seincept import SEExtractor


class BCNN(nn.Module):

    def __init__(self, in_channel, num_class):
        super().__init__()
        self.in_channel = in_channel
        self.num_class = num_class
        self.features = SEExtractor(in_channel)
        self.out_channel = self.features.out_channel

        # Linear classifier.
        self.fc = nn.Linear(self.out_channel**2, num_class)
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        in_size = 224

        N = X.size()[0]
        assert X.size() == (N, self.in_channel, in_size, in_size), 'X size is %s' % str(X.size())
        X = self.features(X)
        out_size = X.size(2)
        # print(X.size())
        assert X.size() == (N, self.out_channel, out_size, out_size)
        X = X.view(N, self.out_channel, -1)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (self.out_channel**2)  # Bilinear
        # assert X.size() == (N, self.out_channel, self.out_channel)
        X = X.view(N, -1)
        X = torch.sqrt(torch.abs(X) + 1e-5)
         # check_nan(X)
        out = nn.functional.normalize(X, eps=1e-3)
        out = self.fc(out)
        # assert out.size() == (N, self.num_class)
        return out


def bcnnet(in_channel, num_class=2):
    return BCNN(in_channel, num_class)


def check_nan(x: torch.Tensor):
    print(torch.isnan(x).sum())
