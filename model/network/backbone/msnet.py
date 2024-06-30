import torch
import torch.nn as nn


# define wrapped nn.module for adapt to dict input
####################################################################

def iter_input(func):
    def wrapper(module: nn.Module, x: dict):
        out = {k: func(module, v) for k, v in x.items()}
        return out

    return wrapper


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)

    @iter_input
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out


class ReLU(nn.ReLU):
    def __init__(self, inplace=True):
        super().__init__(inplace=inplace)

    @iter_input
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out


class UpSample(nn.Upsample):
    def __init__(self, scale_factor=2):
        super().__init__(scale_factor=scale_factor)

    @iter_input
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out


class Sigmoid(nn.Sigmoid):
    def __init__(self):
        super().__init__()

    @iter_input
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride):
        super().__init__(kernel_size=kernel_size, stride=stride)

    @iter_input
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out


######################################################################


class ModeSpeBatchNorm(nn.Module):
    def __init__(self, ch_out, mode: set):
        super(ModeSpeBatchNorm, self).__init__()
        self.bn = nn.ModuleDict()
        self.mode = mode
        for m in mode:
            self.bn[m] = nn.BatchNorm2d(ch_out)

    def forward(self, x: dict):
        return {k: self.bn[k](v) for k, v in x.items()}

    def scalars(self):
        data = {'mean': {}, 'variance': {}}

        for m in self.mode:
            data['mean'][m] = self.bn[m].weight.data.mean().item()
            data['variance'][m] = self.bn[m].bias.data.mean().item()

        return data


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, mode: set):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                  ModeSpeBatchNorm(ch_out, mode),
                                  ReLU(inplace=True),
                                  Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                  ModeSpeBatchNorm(ch_out, mode),
                                  ReLU(inplace=True))

    def forward(self, x: dict):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, mode: set):
        super(AttentionBlock, self).__init__()
        self.mode = mode
        self.W_g = nn.Sequential(
            Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ModeSpeBatchNorm(F_int, mode)
        )

        self.W_x = nn.Sequential(
            Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ModeSpeBatchNorm(F_int, mode)
        )

        self.psi = nn.Sequential(
            Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            ModeSpeBatchNorm(1, mode),
            Sigmoid()
        )

        self.relu = ReLU(inplace=True)

    def forward(self, g: dict, x: dict):
        mode = x.keys()
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu({k: (g1[k] + x1[k]) for k in mode})
        psi = self.psi(psi)

        return {k: (x[k] * psi[k]) for k in mode}


class UpAttConv(nn.Module):
    def __init__(self, ch_in, ch_out, mode: set):
        super().__init__()
        self.mode = mode
        self.up = nn.Sequential(UpSample(scale_factor=2),
                                Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                ModeSpeBatchNorm(ch_out, mode=mode),
                                ReLU(inplace=True)
                                )
        self.att = AttentionBlock(ch_out, ch_out, ch_out // 2, mode)
        self.conv = ConvBlock(ch_in, ch_out, mode)

    def forward(self, x: dict, down_x: dict):
        mode = x.keys()
        x = self.up(x)
        y = self.att(x, down_x)
        out = {k: torch.cat([y[k], x[k]], dim=1) for k in mode}
        out = self.conv(out)
        return out


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, mode: set):
        super().__init__()
        self.mode = mode
        self.up = nn.Sequential(UpSample(scale_factor=2),
                                Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                ModeSpeBatchNorm(ch_out, mode=mode),
                                ReLU(inplace=True)
                                )
        self.conv = ConvBlock(ch_in, ch_out, mode)

    def forward(self, x: dict, down_x: dict):
        mode = x.keys()
        x = self.up(x)
        out = {k: torch.cat([down_x[k], x[k]], dim=1) for k in mode}
        out = self.conv(out)
        return out


class Encoder(nn.Module):
    def __init__(self, img_ch=1, init_ch=32, stage=5, mode: set = 'BEF'):
        super().__init__()
        self.mode = mode
        self.img_ch = img_ch
        self.out_ch = int(init_ch * 2 ** (stage - 1))
        self.stage = stage
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)

        self.encode = self.make_layer(in_ch=init_ch)

    def make_layer(self, in_ch=32):
        channels = [self.img_ch] + [int(in_ch * 2 ** i) for i in range(self.stage)]

        layer = nn.ModuleList()
        for i in range(self.stage):
            module = ConvBlock(ch_in=channels[i], ch_out=channels[i + 1], mode=self.mode)
            layer.append(module)

        return layer

    def forward(self, x: dict) -> dict:
        """

        :rtype: list, out is a list of mediate feature from different stage, and each ele of
                out is a list contain feature from different sites
        """
        mode = x.keys()
        # encoding path
        out = {k: [] for k in mode}

        for i in range(self.stage - 1):
            x = self.encode[i](x)
            for k in mode:
                out[k].append(x[k])
            x = self.max_pool(x)
        x = self.encode[-1](x)

        for k in mode:
            out[k].append(x[k])

        return out


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, up_block=UpAttConv, stage=5, mode: set = 'BEF'):
        super().__init__()
        self.mode = mode
        self.stage = stage
        self.layer = self.make_layer(up_block, in_ch)
        self.conv_seg = Conv2d(in_channels=int(in_ch * 2 ** -(stage - 1)), out_channels=out_ch, kernel_size=1,
                               stride=1, padding=0)
        self.conv_contour = Conv2d(in_channels=int(in_ch * 2 ** -(stage - 1)), out_channels=out_ch, kernel_size=1,
                                   stride=1, padding=0)
        self.sigmoid = Sigmoid()

    def make_layer(self, up_block, in_ch=512):
        stage = self.stage
        channels = [int(in_ch * 2 ** (-i)) for i in range(stage)]

        layer = nn.ModuleList()
        for i in range(stage - 1):
            module = up_block(channels[i], channels[i + 1], mode=self.mode)
            layer.append(module)

        return layer

    def forward(self, x: dict) -> dict:
        """
        :param x: x is a list contain mediate feature from different stage, and
                  each ele of x is a list contain data from different sites,
                  i.e., x = [stage1, stage2, ....], stage = [(B,C,M,N), (B,C,M,N), ...]
        :return:
        """
        # decoding path
        out = {k: v[-1] for k, v in x.items()}
        for i in range(self.stage - 1):
            out = self.layer[i](out, {k: v[self.stage - i - 2] for k, v in x.items()})

        seg = self.conv_seg(out)
        seg = self.sigmoid(seg)
        if self.training:
            contour = self.conv_contour(out)
            contour = self.sigmoid(contour)
            return seg, contour
        else:
            return seg


class MultiSiteNet(nn.Module):
    def __init__(self, sites, up_block, init_ch=32, stage=4):
        super(MultiSiteNet, self).__init__()
        self.sites = sites if isinstance(sites, set) else set(sites)

        self.uni_encoder = Encoder(img_ch=1, init_ch=init_ch, stage=stage, mode=sites)
        out_ch = self.uni_encoder.out_ch

        self.uni_decoder = Decoder(in_ch=out_ch, out_ch=1, up_block=up_block, stage=stage, mode=sites)

        if len(sites) > 1:
            self.spe_decoders = nn.ModuleDict()
            for i, k in enumerate(sites):
                module = Decoder(in_ch=out_ch, out_ch=1, up_block=up_block, stage=stage, mode=k)
                self.spe_decoders[k] = module

    def forward(self, x: dict):
        out = self.uni_encoder(x)

        uni_sg = self.uni_decoder(out)

        if self.training:
            if len(self.sites) > 1:
                spe_seg = {}
                spe_contour = {}
                for k, v in out.items():
                    output = self.spe_decoders[k]({k: v})
                    spe_seg[k] = output[0][k]
                    spe_contour[k] = output[1][k]

                return uni_sg, (spe_seg, spe_contour)

            else:
                return uni_sg

        else:
            return uni_sg


def ms_att_u_net(modes='BEF', init_ch=32, stage=4, *args, **kwargs):
    return MultiSiteNet(sites=modes, up_block=UpAttConv, init_ch=init_ch, stage=stage)


def ms_u_net(modes='BEF', init_ch=32, stage=4, *args, **kwargs):
    return MultiSiteNet(sites=modes, up_block=UpConv, init_ch=init_ch, stage=stage)
