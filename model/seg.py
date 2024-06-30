import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vis_f

from torch.autograd import Variable

from model.template import MultimodalTemplate
from datamanager.dataset import MyDataset
from metric.segment import SGAnalyzer
from model.utils import post_process


def debug(func):
    def wrapper(net, epoch, dataloader, threshold=0.5, record=True):
        num = 10
        dataloader.dataset.dataset.eval()
        for i, (x, label) in enumerate(dataloader):
            if i < num:
                img_tensor = x['B'][0]
                mask_tensor = label['B'][0]
                img = vis_f.to_pil_image(img_tensor.squeeze())
                mask = vis_f.to_pil_image(mask_tensor.squeeze())
                img.save('../debug/img-%d.png' % i)
                mask.save('../debug/mask-%d.png' % i)
            else:
                break
        avg_loss = func(net, epoch, dataloader, threshold, record)
        return avg_loss

    return wrapper


def regular_loss(m_uni: torch.Tensor, p_aux: torch.Tensor):
    assert m_uni.dtype == torch.float and p_aux.dtype == torch.float, 'sg and gt must be float'

    intersect = (m_uni * p_aux).sum()
    union = torch.pow(m_uni, 2).sum() + torch.pow(p_aux, 2).sum()
    return 1 - 2 * intersect / union


def dice_loss(sg: torch.Tensor, gt: torch.Tensor):
    assert sg.dtype == torch.float and gt.dtype == torch.float, 'sg and gt must be float'
    intersect = (sg * gt).sum()
    union = sg.sum() + gt.sum()
    loss = 1 - 2 * intersect / union
    return loss


def dice_bce_loss(sg: torch.Tensor, gt: torch.Tensor):
    assert sg.dtype == torch.float and gt.dtype == torch.float, 'sg and gt must be float'
    batch_size = sg.size(0)
    loss = dice_loss(sg, gt) + F.binary_cross_entropy(sg.view(batch_size, -1), gt.view(batch_size, -1).float())
    return loss


def ms_dice_loss(sg: torch.Tensor, gt: torch.Tensor):
    assert sg.dtype == torch.float and gt.dtype == torch.float, 'sg and gt must be float'
    large_scale = dice_bce_loss(sg, gt)
    sg = F.max_pool2d(sg, 2, 2)
    gt = F.max_pool2d(gt, 2, 2)
    small_scale = dice_bce_loss(sg, gt)
    return large_scale + small_scale


class MSLoss(nn.Module):

    def __init__(self, mode, alpha=0.5, loss_fn='dice'):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
        if loss_fn == 'dice':
            self.loss_fn = dice_loss
        elif loss_fn == 'dice_bce':
            self.loss_fn = dice_bce_loss
        elif loss_fn == 'ms_dice':
            self.loss_fn = ms_dice_loss
        else:
            raise ValueError('Unrecognized loss function %s' % loss_fn)

    def forward(self, uni, spe, label):
        gt, edge = label
        uni_seg, uni_contour = uni
        spe_seg, spe_contour = spe
        assert len(uni_seg) == len(spe_seg), print('this length of uni_seg and spe_sg must be the same')
        loss_fn = self.loss_fn
        l_aux = sum([loss_fn(spe_seg[key], gt[key]) + self.l2_loss(edge[key], spe_contour[key].squeeze())
                     for key in self.mode])
        l_uni = self.alpha * sum([regular_loss(spe_seg[key].detach(), uni_seg[key]) for key in self.mode]) + \
                (1 - self.alpha) * sum([loss_fn(uni_seg[key], gt[key]) +
                                        self.l2_loss(edge[key], uni_contour[key].squeeze()) for key in self.mode])

        return l_aux + l_uni


class MixNet(MultimodalTemplate):
    def __init__(self, model_func, mode, loss_fn='dice', writer=None, is_post_process=True, **kwargs):
        super().__init__(model_func, mode=mode, num_class=2, writer=writer, **kwargs)
        self.loss_func = MSLoss(mode, loss_fn=loss_fn)
        self.is_post_process = is_post_process

    def set_loss(self, x, label):
        img = {}
        gt = ({}, {})
        for key in self._mode:
            img[key] = Variable(x[key]).cuda()
            gt[0][key] = Variable(label[0][key]).cuda()
            gt[1][key] = Variable(label[1][key]).cuda()

        uni_sg, spe_sg = self.model(img)
        return self.loss_func(uni_sg, spe_sg, gt)

    def set_forward(self, x):
        img = {}
        for key in self._mode:
            img[key] = x[key].cuda()

        return self.model(img)

    def train_loop(self, epoch, train_loader, optimizer, writer=None):
        assert isinstance(dataset:=train_loader.dataset.dataset, MyDataset), \
            'The input dataset must be instance of MyDataset'
        dataset.train()
        super().train_loop(epoch, train_loader, optimizer)

    def test_loop(self, epoch, test_loader, threshold=0.5, writer=None):
        sg_keys = ['JS', 'DC', 'score']
        total = len(test_loader.dataset)
        collector = {k: np.zeros((total, len(sg_keys)), dtype=np.float) for k in self._mode}

        index = 0
        assert isinstance(dataset:=test_loader.dataset.dataset, MyDataset), \
            'The input dataset must be instance of MyDataset'
        dataset.eval()
        self.model.eval()
        for i, (image, mask) in enumerate(test_loader):
            mask = mask[0]
            out = self.set_forward(image)
            length = list(out.values())[0].size(0)
            for j in range(length):
                for key in self._mode:
                    gt = mask[key][j].numpy()
                    sg = out[key][j].cpu().detach().squeeze().numpy()
                    sg = (sg > 0.5) + 0
                    if self.is_post_process:
                        sg = post_process(sg)
                    r = SGAnalyzer(gt.reshape(-1), sg.reshape(-1), epoch)
                    for k, m in enumerate(sg_keys):
                        collector[key][index, k] = r.result[m]
                index += 1

        mean = {k: v.mean(0) for k, v in collector.items()}
        std = {k: v.std(0) for k, v in collector.items()}

        for i, key in enumerate(sg_keys):
            self.writer.add_scalars('%s' % key, {k: v[i] for k, v in mean.items()}, epoch)

        for mode in self._mode:
            print('%s: ' % mode + ' | '.join(['%s: %6f +- %6f' % (k, mean[mode][i], std[mode][i])
                                              for i, k in enumerate(sg_keys)]))

        result = {k: {m: (mean[k][i], std[k][i]) for i, m in enumerate(sg_keys)} for k in self._mode}
        return result
