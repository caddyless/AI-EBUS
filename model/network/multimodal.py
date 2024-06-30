import torch
import torch.nn.functional as F

from torch import nn
from model.network.aggregators import Aggregator
from model.network.extractor import Extractor
from model.network.fusion import FusionModule
from model.utils import white
from itertools import combinations

'''
This file generate the multimodal backbone.
There are few parameter need take into control for ablation study:
The feature extraction part:
    B, F always use SEExtract, E is alternative to the handcrafted feature, i.e., 
    color histogram or SEExtract. Thus, there set a switch parameter for E.
        B: level, 
        F: level, 
        E: is_conv, level
The video aggregation part:
    Five ways for video aggregation:
        Average:
        NetVLAD: num_anchor, 
        NeXtVLAD: num_anchor
        Anchor: num_anchor, regular_type
        Subspace: num_space
The multimodal fusion part:
    Three way for multimodal fusion:
        GatedFusion:
        Concat:
        WeightedConcat:
The classifier part:
    two ways:
        MLP:
        Linear:
'''


class MIConstrains(nn.Module):
    def __init__(self, in_channels: dict, dims: int = 128, mode: str = 'BEF', num_noise: int = 100,
                 noise: str = 'gaussian'):
        super().__init__()
        # if noise == 'gaussian':
        #     noise = distribution.normal.Normal(0, 1)
        # else:
        #     raise ValueError('Unknown noise type %s' % str(noise))
        #
        # self.noise = noise
        self.num_noise = num_noise
        self.dims = dims
        self.head = nn.ModuleDict()
        for m in mode:
            self.head[m] = nn.Sequential(nn.Linear(in_channels[m], 256),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(256, dims))
        self.local_loss = torch.tensor(0, dtype=torch.float, requires_grad=True)

    def forward(self, x: dict):
        # noise = self.noise.sample((self.num_noise, self.dims)).to()
        x = [F.normalize(white(self.head[k](v)), p=2, dim=1) for k, v in x.items()]
        loss = 0
        count = 0
        for item in combinations(x, 2):
            v1, v2 = item
            device = v1.device
            b, d = v1.size()
            cat = torch.cat((v1, v2), dim=0)
            v1 = v1.view(b, 1, d)
            cat = cat.view(1, 2 * b, d)
            inner = (v1 * cat).sum(2)
            mask = 1 - torch.eye(b, dtype=torch.long, device=device)
            mask = torch.cat((mask, torch.ones((b, b), dtype=torch.long, device=device)), dim=1)
            inner = inner * mask
            loss += F.cross_entropy(inner, target=torch.arange(b, 2 * b, dtype=torch.long, device=v1.device))
            count += 1
        self.local_loss = loss / count

        return


class Branch(nn.Module):
    def __init__(self, mode: str, num_class: int, extractor: str, extractor_param: dict, aggregator: str,
                 aggregator_param: dict, unsupervised: bool = False):
        super().__init__()

        extractor_param = {} if extractor_param is None else extractor_param
        aggregator_param = {} if aggregator_param is None else aggregator_param

        print('Mode %s use extractor %s and its param is %s:' % (mode, extractor, str(extractor_param)))

        self._mode = mode
        self.extractor = Extractor(method=extractor, params=extractor_param)
        self.extract_method = extractor

        out_channel = self.extractor.out_channel

        aggregator_param.update({'in_channels': out_channel})
        print('Mode %s use aggregator %s and its param is %s:' % (mode, aggregator, str(aggregator_param)))
        self.aggregator = Aggregator(method=aggregator, params=aggregator_param)

        self.out_channels = self.aggregator.out_channels

        if not unsupervised:
            self.classifier = nn.Linear(self.aggregator.out_channels, num_class)
            # self.classifier = create_model(self.aggregator.out_channels, num_classes=num_class)

    def parse_feature(self, x: tuple):
        data, mask = x
        out = self.extract(data)
        out = self.aggregator(out, mask)
        return out

    def extract(self, x):
        b, n = x.size(0), x.size(1)
        if self.extract_method not in ['linear', 'none', 'mlp']:
            b, n, c, w, h = x.size()
            x = x.view(b * n, c, w, h)
        out = self.extractor(x)
        out = out.view(b, n, -1)
        return out

    def clf(self, feature):
        out = self.classifier(feature)
        return out

    def forward(self, x: dict):
        x = x[self._mode]
        out = self.parse_feature(x)
        if hasattr(self, 'clf'):
            out = self.clf(out)
        return out


class WholeModel(nn.Module):

    def __init__(self, mode='BEF', num_class=2, fusion_way='concat', clf_way='linear', branch_params: dict = None,
                 unsupervised: bool = False):

        """
        :param mode: The mode of the backbone
        :param num_class: The number of category
        :param fusion_way: The fusion method, (GatedFusion, Concat, WeightedConcat)
        :param clf_way: The classification method, (MLP, Linear)
        :param b_params: extractor: (seincept: in_channel, level, attention,
                                     resnet: in_channel, level)
                         aggregator: (anchor: num_anchor, regular,
                                      netvald: num_clusters
                                      nextvlad: num_cluster
                                      subspace: num_space, reduction,
                                      attention:
                                      average:)

        :param f_params: extractor: (seincept: in_channel, level, attention,
                                     resnet: in_channel, level)
                         aggregator: (attention:
                                      average:)

        :param e_params: extractor: (seincept: in_channel, level, attention,
                                     resnet: in_channel, level,
                                     linear: in_channel, out_channel, bias,
                                     none:)
                         aggregator: (anchor: num_cluster, regular,
                                      netvald: num_clusters
                                      nextvlad: num_cluster
                                      subspace: num_space, reduction,
                                      attention:
                                      average:)
        """

        super().__init__()
        self.mode = mode
        self.__unsupervised = unsupervised
        self.branches = nn.ModuleDict()
        for m in mode:
            assert m in ['B', 'E', 'F', 'C'], 'Unknown mode %s!' % m
            self.branches[m] = Branch(mode=m, num_class=num_class, **branch_params[m], unsupervised=unsupervised)
            print(branch_params[m])
        if len(mode) > 1:

            if unsupervised:
                self.constrain = MIConstrains({k: self.branches[k].out_channels for k in mode}, dims=128, mode=mode)

            else:
                out_channel = 0
                for v in self.branches.values():
                    out_channel += v.out_channels
                self.fusion_clf = FusionModule(mode, out_channel, num_class, fusion_way=fusion_way, clf_way=clf_way)

    def collect_loss(self):
        sub_loss = {}
        for name, module in self.named_modules():
            if hasattr(module, 'local_loss'):
                # print('%s have attribute local_loss' % name)
                sub_loss[name] = module.local_loss
        return sub_loss

    def parse_feature(self, x):
        mode = self.mode
        feature = {}
        if len(mode) == 1:
            feature[mode] = self.branches[mode].parse_feature(x[mode])

        else:
            for m in mode:
                feature[m] = self.branches[m].parse_feature(x[m])

            if self.__unsupervised:
                if self.training:
                    self.constrain(feature)

            else:
                fusion_feature = self.fusion_clf.parse_feature(feature)
                feature['Fusion'] = fusion_feature

        return feature

    def forward(self, x):
        feature = self.parse_feature(x)

        if self.__unsupervised:
            out = feature
        else:
            out = self.multimodal_clf(feature)

        if self.training:
            sub_loss = self.collect_loss()
            return sub_loss, out
        else:
            return out

    def multimodal_clf(self, feature: dict):
        out = {}
        for k, v in feature.items():
            if k == 'Fusion':
                out['Fusion'] = self.fusion_clf.clf(v)
            else:
                out[k] = self.branches[k].classifier(v)
        return out

    def accumulate_decision(self, x: tuple, previous_feature: torch.Tensor = None):
        mode, data = x
        feature = self.branches[mode].extract(data)
        mask = torch.ones((1, 1), dtype=torch.int8)
        if previous_feature is None:
            feature = self.branches[mode].aggregator(feature, mask)
        else:
            feature = torch.cat((feature, previous_feature), dim=1)
            mask[0, 0] = feature.size(1)
            feature = self.branches[mode].aggregator(feature, mask)
        score = self.branches[mode].clf(feature)
        return score
