import torch.nn as nn
import torch
import torch.nn.functional as F


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


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g


class FireBlock(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.fire = Fire(in_channel, reduction)
        self.se = SEModule(2 * in_channel, reduction)
        self.out_channel = 2 * in_channel

    def forward(self, x):
        x1 = self.fire(x)
        x2 = self.se(x1)
        return x2


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 5, 2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


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


class SharedConvolution(nn.Module):

    def __init__(self, out_channel):
        super().__init__()
        self.branch0 = nn.Conv2d(3, out_channel, kernel_size=7, stride=2, padding=2)
        self.branch1 = nn.Conv2d(3, out_channel, kernel_size=5, stride=2, padding=1)
        self.branch2 = nn.Conv2d(3, out_channel, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     nn.Conv2d(3, out_channel, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class MultiAttention(nn.Module):

    def __init__(self, in_channel, num_class):
        super().__init__()
        s_pool = nn.MaxPool2d(2)
        b_pool = nn.MaxPool2d(4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        mix_channel = 8
        self.c = mix_channel
        self.out_channel = mix_channel * 64
        self.shared = SharedConvolution(mix_channel)
        self.e = nn.Sequential(FireBlock(mix_channel * 4, 8),
                               s_pool,
                               FireBlock(mix_channel * 8, 8),
                               s_pool,
                               FireBlock(mix_channel * 16, 8))

        self.b = nn.Sequential(FireBlock(mix_channel * 4, 16),
                               b_pool,
                               FireBlock(mix_channel * 8, 16))

        self.f = nn.Sequential(FireBlock(mix_channel * 4, 16),
                               b_pool,
                               FireBlock(mix_channel * 8, 16))

        # self.map_be = nn.Linear(mix_channel * 16, mix_channel * 32, bias=False)
        # self.map_fe = nn.Linear(mix_channel * 16, mix_channel * 32, bias=False)

        self.classifier = nn.Conv2d(self.out_channel, num_class, 1)

    def feature(self, x):
        e, b, f = x
        out_e = self.shared(e)
        out_b = self.shared(b)
        out_f = self.shared(f)

        out_e = self.e(out_e)
        out_b = self.b(out_b)
        out_f = self.f(out_f)

        feature = torch.cat([out_e, out_b, out_f], 1)
        out = self.adaptive_pool(feature)
        out = out.view(out.size(0), -1)
        return out

    def save_features(self, x):
        features = self.feature(x)

        e_features = features[:, : self.c * 32]
        b_features = features[:, self.c * 32: self.c * 48]
        f_features = features[:, self.c * 48:]

        return e_features, b_features, f_features

    def forward(self, x, flag=False):
        e, b, f = x
        out_e = self.shared(e)
        out_b = self.shared(b)
        out_f = self.shared(f)

        out_e = self.e(out_e)
        out_b = self.b(out_b)
        out_f = self.f(out_f)

        feature = torch.cat([out_e, out_b, out_f], 1)
        out = self.classifier(feature)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        # if flag:
        #     c_loss = contrastive_loss(out_e.mean((2, 3)), self.map_be(out_b.mean((2, 3))), self.map_fe(out_f.mean((2, 3))))
        #     return out, c_loss
        # else:
        return out


class CentralBlock(nn.Module):

    def __init__(self, in_channel, datatype='ebf', init_layer=False):
        super().__init__()
        self.init_layer = init_layer
        modal_map = {'e': in_channel, 'b': in_channel//2, 'f': in_channel//2}
        self.fires = []
        for d in datatype:
            self.fires.append(FireBlock(modal_map[d], 8))
        self.fires = nn.ModuleList(self.fires)
        self.weights = nn.Parameter(torch.tensor([1/len(datatype) for i in range(len(datatype))], dtype=torch.float),
                                    requires_grad=True).cuda()
        self.pool = nn.MaxPool2d(2)
        if not init_layer:
            self.c = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float), requires_grad=True)
            self.fire_all = FireBlock(sum([modal_map[d] for d in datatype]), 8)
            self.out_channel = self.fire_all.out_channel + sum([f.out_channel for f in self.fires])
        else:
            self.out_channel = sum([f.out_channel for f in self.fires])

    def forward(self, x):
        hc = x[0]
        data = x[1]
        out = []
        for d, m in zip(data, self.fires):
            out.append(self.pool(m(d)))
        w = torch.softmax(self.weights, 0)
        modalities = torch.cat([w[i] * out[i] for i in range(w.size(0))], 1)
        if self.init_layer:
            hc1 = modalities
        else:
            out_hc = self.fire_all(hc)
            out_hc = self.pool(out_hc)
            wc = torch.softmax(self.c, 0)
            hc1 = out_hc * wc[0] + modalities * wc[1]
        return hc1, out


class InitLayer(nn.Module):

    def __init__(self, channel=16, datatype='ebf'):
        super().__init__()
        modal_map = {'e': [3, 2*channel], 'b': [1, channel], 'f': [3, channel]}
        self.convs = []
        for d in datatype:
            self.convs.append(nn.Conv2d(modal_map[d][0], modal_map[d][1], 3, 2, 1))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        out = []
        for d, m in zip(x, self.convs):
            out.append(m(d))
        return out


class Classifier(nn.Module):

    def __init__(self, feature_num, num_class=2, datatype='ebf'):
        super().__init__()
        modal_map = {'e': 2*feature_num, 'b': feature_num, 'f': feature_num}
        self.cls = []
        for d in datatype:
            self.cls.append(nn.Conv2d(modal_map[d], num_class, 1, 1))
        self.cls = nn.ModuleList(self.cls)
        self.cls_all = nn.Conv2d(sum([modal_map[d] for d in datatype]), num_class, 1, 1)

    def forward(self, x):
        a, data = x
        out = []
        for d, m in zip(data, self.cls):
            f = m(d)
            f = f.view(f.size(0), f.size(1), -1).mean(2)
            out.append(f)
        hc = self.cls_all(a)
        hc = hc.view(hc.size(0), hc.size(1), -1).mean(2)
        result = [hc] + out
        return result


class CentralNet(nn.Module):

    def __init__(self, in_channel, num_class, datatype='ebf'):
        super().__init__()
        mix_channel = 32
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.init = InitLayer(mix_channel // 2, datatype)
        self.block1 = CentralBlock(mix_channel, datatype, True)
        self.block2 = CentralBlock(mix_channel * 2, datatype)
        self.block3 = CentralBlock(mix_channel * 4, datatype)
        self.block4 = CentralBlock(mix_channel * 8, datatype)
        self.cls = Classifier(mix_channel * 8, num_class, datatype)

    def forward(self, x, vis=False):
        out = self.init(x)
        out = self.block1([0, out])
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        if vis:
            return out
        else:
            return self.cls(out)

    def feature(self, x):
        out = self.forward(x, vis=True)
        return out[0]

    def save_features(self, x):
        out = self.forward(x, vis=True)
        for i, item in enumerate(out):
            item = self.adp(item)
            item = item.view(item.size(0), -1)
            out[i] = item
        return out


def contrastive_loss(e, b, f):
    distance = torch.pow(b - e, 2).sum() + torch.pow(f - e, 2).sum()
    c_loss = max(torch.tensor(0, dtype=torch.float, device='cuda'), 2 * torch.log(distance + 1) - 0.1)
    return c_loss
