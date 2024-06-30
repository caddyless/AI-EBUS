import torch.nn as nn
import torch

from model.network.backbone.seincept import FireBlock


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, in_channel, num_class):
        super().__init__()
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_channel = 7
        self.down = nn.Sequential(conv_block(ch_in=in_channel, ch_out=64),
                                  pool,
                                  conv_block(ch_in=64, ch_out=128),
                                  pool,
                                  conv_block(ch_in=128, ch_out=256),
                                  pool,
                                  conv_block(ch_in=256, ch_out=512))

        self.up = nn.Sequential(up_conv(ch_in=512, ch_out=256),
                                up_conv(ch_in=256, ch_out=128),
                                up_conv(ch_in=128, ch_out=64),
                                nn.Conv2d(64, in_channel, kernel_size=1, stride=1, padding=0))
        self.fire = FireBlock(512, 16)
        self.adp = nn.AdaptiveMaxPool2d(1)
        self.cls = nn.Linear(1024, num_class)

    def forward(self, x):
        out = torch.cat(x, 1)
        batch_size = out.size(0)
        out = self.down(out)
        rebuild = self.up(out)
        cls_feature = self.fire(out)
        score = self.cls(self.adp(cls_feature).view(batch_size, -1))
        return rebuild, score

    def feature(self, x):
        in_tensor = torch.cat(x, 1)
        bottleneck = self.down(in_tensor)
        cls_feature = self.fire(bottleneck)
        return cls_feature

    def save_features(self, x):
        in_tensor = torch.cat(x, 1)
        bottleneck = self.down(in_tensor)
        cls_feature = self.fire(bottleneck)
        return cls_feature
