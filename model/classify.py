import torch
import random

import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch import nn
from metric.classify import Analyzer
from model.template import MultimodalTemplate, optimal_threshold
from torch.autograd import Variable
from model.utils import UnionLoss
from torch.distributions.beta import Beta
from torchvision.transforms.functional import to_pil_image


def debug(func):
    def wrapper(*args, **kwargs):
        num = 10
        dataloader = args[2]
        for i, (id_, x, label) in enumerate(dataloader):
            if i < num:
                img_tensor = x['B'][0][0, 5]
                print(img_tensor.size())
                print(img_tensor.max())
                img = to_pil_image(img_tensor.squeeze())
                img.save('../debug/img-%d.png' % i)
            else:
                break

        avg_loss = func(*args, **kwargs)
        return avg_loss

    return wrapper


class MMVideoNet(MultimodalTemplate):

    def __init__(self, model_func, weights, num_class=2, is_parallel=False, **kwargs):
        super().__init__(model_func, num_class=num_class, is_parallel=is_parallel, **kwargs)
        self.mode = list(kwargs['mode'])
        if len(self.mode) > 1:
            self.mode.append('Fusion')
        self.loss_fn = UnionLoss(weights)
        self.remix = Remix(0.6)

    def calibrate_model(self, epoch, test_loader, val_loader):
        print('evaluation on test_loader...')
        result_t = self.test_loop(epoch, test_loader)
        print('\nCalibrating...')
        print('evaluation on val_loader...')
        result_v = self.test_loop(epoch, val_loader)
        for k in result_t.keys():
            optim_t, optim_acc = optimal_threshold(result_v[k].raw_data['y_true'], result_v[k].raw_data['y_score'])
            print('\nOptimal threshold is %f, optimal accuracy on val is %f' % (optim_t, optim_acc))
            print('\nAfter calibrating...')
            print('evaluation on test_loader...')
            result_t[k].result = result_t[k].analyze(optim_t)
            print(' | '.join(result_t[k].output_data()))
        return result_t

    def to_cuda(self, x: dict):
        inputs = {}
        for k, v in x.items():
            if isinstance(v, (list, tuple)):
                if self.training:
                    inputs[k] = (Variable(v[0]).cuda(), v[1].cuda())
                else:
                    inputs[k] = (v[0].cuda(), v[1].cuda())
            else:
                raise ValueError('Unknown type of v %s' % type(v))
        return inputs

    def trace_score(self, scores, label, identity, epoch):
        traces = pd.DataFrame()
        device = label.device
        for m in self.mode:
            category_score = scores[m].detach()
            for j in range(identity.size(0)):
                traces = traces.append({'Identity': identity[j].item(), 'Epoch': epoch, 'Mode': m,
                                        'Category-0': category_score[j][0].item(),
                                        'Category-1': category_score[j][1].item()}, ignore_index=True)
            # prediction = F.softmax(scores[m].detach(), dim=1).to(device)
            # gap = torch.abs(prediction[:, 1] - label)
            # for j in range(identity.size(0)):
            #     traces = traces.append({'Identity': identity[j].item(), 'Epoch': epoch, 'Mode': m,
            #                             'Gap': gap[j].item()}, ignore_index=True)
        return traces

    def set_forward(self, x):
        inputs = self.to_cuda(x)
        return self.model(inputs)

    def set_loss(self, x, label, identity, epoch, is_trace, *args):

        label = Variable(label).cuda()
        x = self.to_cuda(x)
        sub_loss, scores = self.model(x)
        sub_loss = {k: v.mean() for k, v in sub_loss.items()}
        loss = self.loss_fn(scores, label)
        loss.update(sub_loss)
        if is_trace:
            traces = self.trace_score(scores, label, identity, epoch)
            return loss, traces
        else:
            return loss

    def train_loop(self, epoch, train_loader, optimizer, batch_times: int = 1, writer=None, is_trace=False):
        print_freq = self.print_freq
        avg_loss = 0
        loss_record = None
        trace_df = pd.DataFrame(columns=('Identity', 'Epoch', 'Mode', 'Gap'))

        self.model.train()
        self.set_dataset(train_loader, True)
        indicator = batch_times
        print('batch times %d' % batch_times)
        for i, (id_, x, label) in enumerate(train_loader):
            # print(label)
            # optimizer.zero_grad()
            # x, label = self.remix(x, label)
            out = self.set_loss(x, label, id_, epoch, is_trace)
            if is_trace:
                loss, traces = out
                trace_df = pd.concat([trace_df, traces])
            else:
                loss = out

            if loss_record is None:
                loss_record = {k: v.data.item() for k, v in loss.items()}
            else:
                loss_record = {k: (loss_record[k] * i + loss[k].data.item()) / float(i + 1) for k in loss_record.keys()}

            total_loss = sum([v for v in loss.values()])
            total_loss.backward()
            indicator -= 1
            if indicator == 0:
                optimizer.step()
                indicator = batch_times
                optimizer.zero_grad()
            avg_loss = (avg_loss * i + total_loss.data.item()) / float(i + 1)

            if i % print_freq == 0:
                print('Epoch {:3d} | Batch {:3d}/{:3d} | Loss {:6f} | lr {:6f}'.format(
                    epoch, i, len(train_loader), avg_loss, optimizer.param_groups[0]['lr']))

        if writer is not None:
            writer.add_scalars('loss', loss_record, epoch)
            self.write_param(epoch, writer)

        return trace_df

    def test_loop(self, epoch, test_loader, multi_test=1, temperature=1, threshold=0.5, writer=None, is_trace=False):
        length = len(test_loader.dataset)
        y_score = {}
        result = {}
        modes = self.mode
        y_true = np.zeros(length, dtype=np.int)
        identity = np.zeros(length, dtype=np.int)
        trace_df = pd.DataFrame(columns=('Identity', 'Epoch', 'Mode', 'Gap'))
        for m in modes:
            y_score[m] = np.zeros(length, dtype=np.float)

        self.model.eval()
        self.set_dataset(test_loader, False)
        index = 0
        with torch.no_grad():
            for i, (id_, x, label) in enumerate(test_loader):
                label.squeeze_()
                scores = self.set_forward(x)
                num = label.numel()
                identity[index: (index + num)] = id_.numpy()
                y_true[index: (index + num)] = label.numpy()
                if is_trace:
                    traces = self.trace_score(scores, label, id_, epoch)
                    trace_df = pd.concat([trace_df, traces])
                for m in modes:
                    score = F.softmax(scores[m], dim=1)
                    score = score.cpu().detach()
                    y_score[m][index: (index + num)] = score.numpy()[:, 1]
                index += num

        for m in modes:
            result[m] = Analyzer(y_true, y_score[m], identity, epoch, threshold)
            print('%-6s: ' % m + ' | '.join(result[m].output_data()))

        if writer is not None:
            wait_record = {'Accuracy': {}, 'AUC': {}}
            for metric in wait_record.keys():
                for m in modes:
                    wait_record[metric][m] = result[m].result[metric]
                writer.add_scalars(metric, wait_record[metric], epoch)

        if is_trace:
            return result, trace_df
        else:
            return result

    def parse_feature(self, data_loader):
        length = len(data_loader.dataset)
        keys = self.mode
        y_true = np.zeros(length, dtype=np.int8)
        features = {}
        model = self.model.module if hasattr(self.model, 'module') else self.model

        for m in keys:
            if m == 'Fusion':
                dims = model.fusion_clf.out_channel
            else:
                dims = model.branches[m].out_channel
            features[m] = np.zeros((length, dims), dtype=np.float)

        self.model.eval()
        data_loader = self.set_dataset(data_loader, False)
        with torch.no_grad():
            index = 0
            for i, (id_, x, label) in enumerate(data_loader):
                num_sample = label.size(0)
                x = self.to_cuda(x)
                f = model.parse_feature(x)
                for m in keys:
                    features[m][index: index + num_sample] = f[m].cpu().numpy()
                y_true[index: index + num_sample] = label.numpy()
                index += num_sample

        return features, y_true


class Remix(nn.Module):
    def __init__(self, p: float = 0.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.class_weights = class_weights
        self.p = p

    def forward(self, x: dict, label: torch.Tensor):
        p = self.p
        batch_size = x['E'][0].size(0)
        beta_dist = Beta(1, 1)
        new_label = torch.zeros_like(label)
        neg_indices = torch.nonzero(label == 0)
        num_neg = len(neg_indices)
        print(num_neg)
        if num_neg == 0:
            return x, label

        new_output = {k: (torch.zeros_like(v[0]), v[1]) for k, v in x.items()}

        for i in range(batch_size):
            lam = beta_dist.sample()
            if lam < 0.4:
                new_label[i] = 0
            else:
                new_label[i] = 1

            if random.uniform(0, 1) < p and label[i] == 1:
                for k, v in x.items():
                    pos_sample = v[0][i]
                    neg_sample = v[0][random.sample(list(neg_indices), 1)]
                    synthesized_sample = lam * pos_sample + (1 - lam) * neg_sample
                    new_output[k][0][i] = synthesized_sample
        return new_output, new_label
