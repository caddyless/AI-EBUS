import torch
import numpy as np
from torch import nn as nn
from torchvision import transforms as tf
from torchvision.transforms.functional import resize
from metainfo.default import B_IMAGE_SIZE, E_DIMS

frame_ceil = {'B': 30, 'F': 3, 'E': 50, 'C': 30}


def obtain_transform(data_arrange):
    defined_transform = {}
    for m, dty in data_arrange.items():
        if m == 'B':
            trans = ImageTransform(m, frame_ceil['B'], 1, B_IMAGE_SIZE)
        elif m == 'E':
            if 'hist' in dty:
                trans = ElasticTransform(frame_ceil['E'], E_DIMS, dty, True)
            else:
                trans = ImageTransform(m, frame_ceil['E'], 3, B_IMAGE_SIZE)

        elif m == 'F':
            trans = ImageTransform(m, frame_ceil['F'], 3, B_IMAGE_SIZE)

        elif m == 'C':
            if 'roi' in dty:
                img_size = 32
            else:
                img_size = 224

            if 'max' in dty:
                max_number = 1
            else:
                max_number = 30
            trans = ImageTransform(m, max_number, 1, img_size)

        else:
            raise ValueError('Unknown mode %s' % m)

        defined_transform[m] = trans

    return defined_transform


class ImageTransform(object):
    def __init__(self, mode: str, num_ceil: int, channel: int, size: int = 224):
        if mode == 'C':
            train_trans = [ToTensor('cpu'),
                           BlackBackGround(size),
                           tf.GaussianBlur(3),
                           tf.RandomAffine(degrees=90, translate=(0.2, 0.2), fillcolor=0),
                           Normalize()]
            test_trans = [ToTensor('cpu'),
                          BlackBackGround(size),
                          Normalize()]

        else:
            train_trans = [ToTensor('cpu'),
                           tf.Resize(size),
                           tf.CenterCrop(size),
                           tf.RandomAffine(degrees=90, translate=(0.2, 0.2), fillcolor=0),
                           Normalize()]
            test_trans = [ToTensor('cpu'),
                          tf.Resize(size),
                          tf.CenterCrop(size),
                          Normalize()]

        self.train_transform = torch.nn.Sequential(*train_trans)
        self.test_transform = torch.nn.Sequential(*test_trans)

        self.formed_shape = (num_ceil, channel, size, size)

    def __call__(self, data: np.ndarray, is_training: bool):
        formed_data = torch.zeros(self.formed_shape, dtype=torch.float)

        if is_training:
            transform = self.train_transform
        else:
            transform = self.test_transform

        data = transform(data)
        formed_data[:data.size(0)] = data

        return formed_data


class ElasticTransform(object):
    def __init__(self, num_ceil: int, channel: int, datatype: str, is_normalize: bool):
        if '+' in datatype:
            transform = (ImageTransform(num_ceil, 3), HistTransform(num_ceil, channel, is_normalize))
        elif 'hist' in datatype:
            transform = HistTransform(num_ceil, channel, is_normalize)
        else:
            transform = ImageTransform(num_ceil, 3)
        self.transform = transform

    def __call__(self, data, is_training: bool):
        transform = self.transform
        if isinstance(transform, tuple):
            img, hist = data
            img, hist = transform[0](img, is_training), transform[1](hist, is_training)
            return img, hist
        else:
            return transform(data, is_training)


class HistTransform(object):
    def __init__(self, num_ceil: int, channel: int = 512, is_normalize: bool = True):
        self.is_normalize = is_normalize
        self.formed_shape = (num_ceil, channel)

    def __call__(self, data: np.ndarray, is_training: bool):
        formed_data = torch.zeros(self.formed_shape, dtype=torch.float)
        data = torch.from_numpy(data)
        data = data / (data.sum(1, keepdim=True) + 1e-6)
        if self.is_normalize:
            data = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
        formed_data[:data.size(0)] = data

        return formed_data


class ToTensor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, x: np.ndarray):
        assert len(x.shape) == 4, 'The shape of x expected to be 4, however get %s' % str(x.shape)
        return torch.from_numpy(x).to(device=self.device)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.ByteTensor):
            x = x.float().div(255)
        return x


class BlackBackGround(nn.Module):

    def __init__(self, target_size: int):
        super().__init__()
        self.target_size = target_size
        self.resize = tf.Resize(target_size)

    def forward(self, x: torch.Tensor):
        n, c, w, h = x.size()
        target_size = self.target_size

        if w <= target_size and h <= target_size:
            output = self.embed(x)
        else:
            factor = max(w, h) / target_size
            x = resize(x, [int((w / factor) + 0.5), int((h / factor) + 0.5)])
            output = self.embed(x)
        return output

    def embed(self, x: torch.Tensor):
        n, c, w, h = x.size()
        target_size = self.target_size

        output = torch.zeros((n, c, target_size, target_size), dtype=torch.float)
        left, right = obtain_range(target_size, w)
        top, bottom = obtain_range(target_size, h)
        output[:, :, left: right, top: bottom] = x

        return output


def obtain_range(target_size: int, a: int):
    return int((target_size - a) / 2 + 0.5), int((target_size + a) / 2 + 0.5)
