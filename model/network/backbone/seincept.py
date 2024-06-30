import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.backbone.fcalayer import FcaLayer


class BasicSeperableConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel,
                                   **kwargs)
        # self.frn1 = FRN(in_channel, learnable_eps=True)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1)
        # self.frn2 = FRN(out_channel, learnable_eps=True)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.depthwise(x)
        # x = self.frn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        # x = self.frn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)
        # self.frn = FRN(out_channel, learnable_eps=True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.frn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackConv(nn.Module):

    def __init__(self, in_channel, out_channel, stacks=1, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        convs = [nn.Conv2d(in_channel, out_channel, 3, padding=1, **kwargs)]
        for i in range(stacks - 1):
            convs.append(nn.Conv2d(out_channel, out_channel, 3, groups=out_channel, padding=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = self.convs(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Mix1a(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.branch0 = StackConv(in_channel, out_channel, 3, stride=2)
        self.branch1 = StackConv(in_channel, out_channel, 2, stride=2)
        self.branch2 = StackConv(in_channel, out_channel, 1, stride=2)
        self.branch3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     nn.Conv2d(in_channel, out_channel, kernel_size=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class Mix2a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv(96, 64, kernel_size=1, stride=1, padding=0),
            BasicConv(64, 128, kernel_size=3, stride=3, padding=0)
        )

        self.branch1 = nn.Sequential(
            BasicConv(96, 64, kernel_size=1, stride=1),
            BasicConv(64, 128, kernel_size=5, stride=3, padding=1)
        )

        self.branch2 = nn.Sequential(
            # BasicSeperableConv(99, 64, kernel_size=1, stride=1),
            BasicConv(96, 128, kernel_size=7, stride=3, padding=2),
            # BasicConv(64, 128, kernel_size=(7, 1), stride=(1, 3), padding=(2, 0)),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mix3a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv(384, 256, kernel_size=1, stride=1),
            BasicConv(256, 256, kernel_size=4, stride=4)
        )

        self.branch1 = nn.Sequential(
            BasicConv(384, 256, kernel_size=1, stride=1),
            BasicConv(256, 256, kernel_size=6, stride=4, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicSeperableConv(384, 256, kernel_size=1, stride=1),
            BasicSeperableConv(256, 256, kernel_size=(1, 8), stride=(4, 1), padding=(0, 2)),
            BasicSeperableConv(256, 256, kernel_size=(8, 1), stride=(1, 4), padding=(3, 0)),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class AdaptivePool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        kernel_size = x.size(2)
        out = F.avg_pool2d(x, kernel_size)
        return out.view(out.size(0), -1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(9, 9)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Fire(nn.Module):

    def __init__(self, in_channel, reduction):
        super().__init__()
        squzee_channel = in_channel // reduction
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.squeeze(x)
        x2 = torch.cat([
            self.expand_1x1(x1) + x,
            self.expand_3x3(x1) + x
        ], 1)

        return x2


class FireBlock(nn.Module):
    def __init__(self, in_channel, reduction, attention: str = 'se', **kwargs):
        super().__init__()
        self.fire = Fire(in_channel, reduction)
        if attention == 'se':
            self.se = SEModule(2 * in_channel, reduction)
        elif attention == 'fca':
            self.se = FcaLayer(2 * in_channel, reduction, **kwargs)

    def forward(self, x):
        x1 = self.fire(x)
        x2 = self.se(x1)
        return x2


class SEExtractor(nn.Module):
    def __init__(self, in_channel, level=4, attention: str = 'normal', channel_attention: str = 'se'):
        super().__init__()
        init_channel = 2 ** level

        self.adaptive = nn.AdaptiveAvgPool2d(1)
        if channel_attention == 'se':
            kwargs = [{} for i in range(4)]
        elif channel_attention == 'fca':
            kwargs = [{'width': 112 // (2 ** i), 'height': 112 // (2 ** i)} for i in range(4)]
        else:
            raise ValueError('Unknown channel attention: %s' % channel_attention)

        if attention == 'normal':
            self.out_channel = init_channel * 64
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.feature = nn.Sequential(
                Mix1a(in_channel, init_channel),
                FireBlock(init_channel * 4, 8, channel_attention, **kwargs[0]),
                pool,
                FireBlock(init_channel * 8, 8, channel_attention, **kwargs[1]),
                pool,
                FireBlock(init_channel * 16, 8, channel_attention, **kwargs[2]),
                pool,
                FireBlock(init_channel * 32, 8, channel_attention, **kwargs[3])
            )

        elif attention == 'GateAttention':
            self.out_channel = 112 * init_channel
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.out_channel = init_channel * 112
            self.warm = nn.Sequential(
                Mix1a(in_channel, init_channel),
                FireBlock(init_channel * 4, 8, channel_attention, **kwargs[0]),
                self.pool)
            self.init_layer = FireBlock(init_channel * 8, 8, channel_attention, **kwargs[1])
            self.intermediate = FireBlock(init_channel * 16, 8, channel_attention, **kwargs[2])
            self.last_layer = FireBlock(init_channel * 32, 8, channel_attention, **kwargs[3])
            self.attn1 = GridAttentionBlock(init_channel * 16, init_channel * 64, init_channel * 32, up_factor=4,
                                            normalize_attn=True)
            self.attn2 = GridAttentionBlock(init_channel * 32, init_channel * 64, init_channel * 32, up_factor=2,
                                            normalize_attn=True)

        elif attention == 'LinearAttention':

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.out_channel = init_channel * 64 * 3
            self.warm = nn.Sequential(
                Mix1a(in_channel, init_channel),
                FireBlock(init_channel * 4, 8, channel_attention, **kwargs[0]),
                self.pool)
            self.init_layer = FireBlock(init_channel * 8, 8, channel_attention, **kwargs[1])
            self.intermediate = FireBlock(init_channel * 16, 8, channel_attention, **kwargs[2])
            self.last_layer = FireBlock(init_channel * 32, 8, channel_attention, **kwargs[3])
            # self.adaptive = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(init_channel * 64, init_channel * 64, kernel_size=14, bias=True)
            self.projector1 = ProjectorBlock(init_channel * 16, init_channel * 64)
            self.projector2 = ProjectorBlock(init_channel * 32, init_channel * 64)
            self.attn1 = LinearAttentionBlock(init_channel * 64)
            self.attn2 = LinearAttentionBlock(init_channel * 64)
            self.attn3 = LinearAttentionBlock(init_channel * 64)

        else:
            raise ValueError('Unknown mode %s!' % attention)

        self.attention = attention

    def forward(self, x):
        if self.attention == 'normal':
            return self.adaptive(self.feature(x))

        elif self.attention == 'GateAttention':
            x = self.warm(x)
            l1 = self.init_layer(x)
            x = self.pool(l1)
            l2 = self.intermediate(x)
            x = self.pool(l2)
            l3 = self.last_layer(x)
            g = self.adaptive(l3).sum((2, 3))
            c1, g1 = self.attn1(l1, l3)
            c2, g2 = self.attn2(l2, l3)
            g = torch.cat([g1, g2, g], 1)
            return g

        elif self.attention == 'LinearAttention':
            x = self.warm(x)
            l1 = self.init_layer(x)
            x = self.pool(l1)
            l2 = self.intermediate(x)
            x = self.pool(l2)
            l3 = self.last_layer(x)
            # g = self.adaptive(l3)
            g = self.fc1(l3)
            c1, g1 = self.attn1(self.projector1(l1), g)
            c2, g2 = self.attn2(self.projector2(l2), g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat([g1, g2, g3], 1)
            return g

        else:
            raise ValueError('Unknown mode %s' % self.mode)


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        return self.op(inputs)


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l + g)  # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, W, H), g


class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_))  # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)  # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)  # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)
        return c.view(N, 1, W, H), output


class SEInception(nn.Module):

    def __init__(self, in_channel, num_class=2):
        super().__init__()

        self.feature = SEExtractor(in_channel)
        out_channel = self.feature.out_channel
        self.avg = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(out_channel, num_class)

    def forward(self, x):
        out = self.feature(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def parse_feature(self, x):
        return self.feature(x)


def seincept(in_channel, num_class=2, datatype='e'):
    model = SEInception(in_channel, num_class)
    # init_dict = model.state_dict()
    # state = torch.load('../models/pretrain/512-channel-pretrain.pth')
    # pretrained_dict = {k: v for k, v in state['param'].items() if 'classifier' not in k}
    # init_dict.update(pretrained_dict)
    # model.state_dict(init_dict)
    # for name, m in model.named_children():
    #     print(name)
    # init_weights(model, init_type='xavier')
    return model
