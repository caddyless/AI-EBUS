import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.utils.data import Subset
from datamanager.dataset import MyDataset
from abc import abstractmethod
from metric.classify import Analyzer


def debug(func):
    def wrapper(net, epoch, dataloader, optimizer):
        num = 10
        for i, (x, label) in enumerate(dataloader):
            if i < num:
                img_tensor = x['B'][0][0, 13]
                print(img_tensor.size())
                print(img_tensor.max())
                img = F.to_pil_image(img_tensor.squeeze())
                print(img.shape)
                img.save('../debug/img-%d.png' % i)
            else:
                break

        avg_loss = func(net, epoch, dataloader, optimizer)
        return avg_loss

    return wrapper


class Template(nn.Module):
    def __init__(self, model_func, is_parallel=True, print_freq=1, **kwargs):
        super().__init__()
        self.is_parallel = is_parallel
        self.print_freq = print_freq
        model = model_func(**kwargs).cuda()
        model = nn.DataParallel(model) if is_parallel else model
        self.model = model

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_loss(self, *args, **kwargs):
        pass

    def load_params(self, params):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(params)
        else:
            self.model.load_state_dict(params)
        return

    def calibrate_model(self, epoch, test_loader, val_loader):
        print('evaluation on test_loader...')
        result_t = self.test_loop(epoch, test_loader)
        print('\nCalibrating...')
        print('evaluation on val_loader...')
        result_v = self.test_loop(epoch, val_loader)
        optim_t, optim_acc = optimal_threshold(result_v.labels, result_v.probability)
        print('Optimal threshold is %f, optimal accuracy on val is %f' % (optim_t, optim_acc))

        print('\nAfter calibrating...')
        print('evaluation on test_loader...')
        new_result_t = Analyzer(result_t.labels, result_t.probability, result_t.number, epoch, optim_t)
        print(' | '.join(new_result_t.output_data()))
        return new_result_t

    def train_loop(self, epoch, train_loader, optimizer, writer=None):
        print_freq = self.print_freq
        avg_loss = 0

        self.model.train()
        for i, (x, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.set_loss(x, label)
            try:
                loss.backward()
            except RuntimeError:
                print(loss.item())
                continue
            optimizer.step()
            avg_loss = (avg_loss * i + loss.data.item()) / float(i + 1)

            if i % print_freq == 0:
                print('Epoch {:3d} | Batch {:3d}/{:3d} | Loss {:6f} | lr {:6f}'.format(
                    epoch, i, len(train_loader), loss.data.item(), optimizer.param_groups[0]['lr']))

        if writer is not None:
            writer.add_scalar('loss', avg_loss, epoch)
            self.write_param(epoch, writer)
        return avg_loss

    def write_param(self, epoch, writer):
        for name, module in self.named_modules():
            if hasattr(module, 'scalars'):
                data = module.scalars()
                for k, v in data.items():
                    writer.add_scalars('ModeSpeBatchNorm/%s-' % name + k, v, epoch)

    def test_loop(self, epoch, test_loader, threshold=0.5, writer=None):

        length = len(test_loader.dataset)
        y_true = np.zeros(length, dtype=np.int)
        y_score = np.zeros(length, dtype=np.float)
        serial = np.zeros(length, dtype=np.int)
        index = 0

        self.model.eval()
        with torch.no_grad():
            for i, (x, label) in enumerate(test_loader):
                label.squeeze_()
                scores = self.set_forward(x)
                scores = F.softmax(scores, dim=1)
                num = label.numel()
                y_true[index: (index + num)] = label.cpu().detach().numpy()
                y_score[index: (index + num)] = scores.cpu().detach().numpy()[:, 1]
                # serial[index: (index + num)] = number
                index += num

        result = Analyzer(y_true, y_score, serial, epoch, threshold)
        print(' | '.join(result.output_data()))

        if writer is not None:
            writer.add_scalar('accuracy', result.data['Accuracy'], epoch)
            writer.add_scalar('AUC', result.data['AUC'], epoch)
            writer.add_histogram('prediction', y_score, epoch)

        return result


class MultimodalTemplate(Template):
    def __init__(self, model_func, mode='BEF', is_parallel=True, print_freq=1, **kwargs):
        super().__init__(model_func, mode=mode, is_parallel=is_parallel, print_freq=print_freq, **kwargs)
        self._mode = mode

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_loss(self, *args, **kwargs):
        pass

    @staticmethod
    def set_dataset(dataloader, train: bool = True):
        dataset = dataloader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if isinstance(dataset, MyDataset):
            dataset.training = train
        return dataloader


def scale_score(scores, threshold):
    pos_index = (scores >= threshold)
    neg_index = (scores < threshold)
    scores[pos_index] = (scores[pos_index] - 1) * 0.5 / (1 - threshold) + 1
    scores[neg_index] = scores[neg_index] * 0.5 / threshold
    return scores


def optimal_threshold(y_true, y_scores):
    best_acc = 0.0
    optimal_t = -1
    score = y_scores
    for s in score:
        acc = eval_acc(y_true, score, s)
        if acc > best_acc:
            best_acc = acc
            optimal_t = s
    return optimal_t, best_acc


def eval_acc(label: np.ndarray, scores: np.ndarray, t):
    pred = (scores > t) + 0
    accuracy = (pred == label).sum() / pred.size
    return accuracy
