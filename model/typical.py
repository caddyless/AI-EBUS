from model.template import Template
from torch.autograd import Variable
from model.utils import LQ, MAE, CrossEntropyLabelSmooth, GHMC, FocalLoss, CentralLoss
import torch.nn as nn
import torch.nn.functional as F


class TypicalNet(Template):
    def __init__(self, model_func, loss_fn='CCE', weights=[1.0], writer=None, **kwargs):
        super().__init__(model_func, writer=writer, **kwargs)
        # count = [int(n) for n in p_n.split('-')]
        # beta = 10
        # weight = [1 + (1 - s/max(count))/beta for s in count]
        # weight = [1, weights]
        # self.weight = torch.tensor(weight, dtype=torch.float, device=device)
        if loss_fn == 'CCE':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'MAE':
            self.loss_fn = MAE()
        # elif loss_fn == 'LQ':
        #     self.loss_fn = LQ(args.q)
        elif loss_fn == 'smooth':
            self.loss_fn = CrossEntropyLabelSmooth(2, 0.1, self.weight)
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(2, 0.25)
        elif loss_fn == 'central':
            self.loss_fn = CentralLoss(weights, self.writer, kwargs['mode'])
            print('current in central loss %s mode' % kwargs['mode'])
        else:
            raise BaseException('Unknown loss function')

    def set_forward(self, x):
        shape = x.size()
        x = x.cuda()
        if len(shape) == 5:
            x = x.view(-1, *shape[2:])
            scores = self.model(x)
            scores = scores.view(*shape[:2], 2).mean(1)
        else:
            scores = self.model(x)
        return scores

    def set_loss(self, x, label):
        label = Variable(label).cuda()
        x = Variable(x)
        scores = self.set_forward(x)
        return self.loss_fn(scores, label)

    def parse_feature(self, x):
        x = x.to(device)
        feature_map = self.model.module.feature(x)
        feature = F.adaptive_avg_pool2d(feature_map, (1, 1))
        feature = feature.squeeze().unsqueeze(0)
        return feature
