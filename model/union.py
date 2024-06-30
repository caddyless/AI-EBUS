from model.typical import TypicalNet
from args import device, args
from torch.autograd import Variable
import torch


class MMVideoNet(TypicalNet):

    def __init__(self, model_func, writer=None, **kwargs):
        super().__init__(model_func, writer, **kwargs)
        self.index = 0
        # self.loss_fn = Regularized_CCE()

    def set_forward(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model(x)

    def set_loss(self, x, label):
        label = Variable(label).to(device)
        x = tuple([Variable(item).to(device) for item in x])
        scores = self.model(x)
        # self.writer.add_scalar('c_loss', c_loss.detach().cpu().item(), self.index)
        self.index += 1
        return self.loss_fn(scores, label)

    def parse_feature(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model.feature(x)

    def save_features(self, loader, index=None):
        images = None
        labels = None
        if index is None:
            index = list(range(len(loader)))
        for i, (x, label, number) in enumerate(loader):
            if i in index:
                if images is None:
                    images = x
                else:
                    for j in range(len(images)):
                        images[j] = torch.cat([images[j], x[j]], 0)

                if labels is None:
                    labels = label
                else:
                    labels = torch.cat([labels, label], 0)

        images = tuple([item.to(device) for item in images])
        e, b, f = self.model.module.save_features(images)
        self.writer.add_embedding(e, metadata=labels, label_img=images[0], tag='e')
        self.writer.add_embedding(b, metadata=labels, label_img=images[1], tag='b')
        self.writer.add_embedding(f, metadata=labels, label_img=images[2], tag='f')
        self.writer.add_embedding(torch.cat([e, b, f], 1), metadata=labels, label_img=images[0], tag='all')


class Central(MMVideoNet):

    def __init__(self, model_func, writer=None, **kwargs):
        super().__init__(model_func, writer, **kwargs)

    def set_forward(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model(x)[0]

    def set_loss(self, x, label):
        label = Variable(label).to(device)
        x = tuple([Variable(item).to(device) for item in x])
        scores = self.model(x)
        weights = [args.cw] + [1] * (len(scores) - 1)
        return sum([w * self.loss_fn(s, label) for w, s in zip(weights, scores)])

    def parse_feature(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model.feature(x)

    def save_features(self, loader, index=None):
        images = None
        labels = None
        if index is None:
            index = list(range(len(loader)))
        for i, (x, label, number) in enumerate(loader):
            if i in index:
                if images is None:
                    images = x
                else:
                    for j in range(len(images)):
                        images[j] = torch.cat([images[j], x[j]], 0)

                if labels is None:
                    labels = label
                else:
                    labels = torch.cat([labels, label], 0)

        images = tuple([item.to(device) for item in images])
        hc, e, b, f = self.model.module.save_features(images)
        self.writer.add_embedding(e, metadata=labels, label_img=images[0], tag='e')
        self.writer.add_embedding(b, metadata=labels, label_img=images[1], tag='b')
        self.writer.add_embedding(f, metadata=labels, label_img=images[2], tag='f')
        self.writer.add_embedding(hc, metadata=labels, label_img=images[0], tag='all')


class Auto(MMVideoNet):

    def __init__(self, model_func, writer=None, **kwargs):
        super().__init__(model_func, writer, **kwargs)

    def set_forward(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model(x)[1]

    def set_loss(self, x, label):
        label = Variable(label).to(device)
        x = tuple([Variable(item).to(device) for item in x])
        rebuild, score = self.model(x)
        x = torch.cat(x, 1)
        return self.loss_fn(score, label) + 0.2 * torch.log((rebuild - x).pow(2).sum())

    def parse_feature(self, x):
        x = tuple([item.to(device) for item in x])
        return self.model.feature(x)

    def save_features(self, loader, index=None):
        images = None
        labels = None
        if index is None:
            index = list(range(len(loader)))
        for i, (x, label, number) in enumerate(loader):
            if i in index:
                if images is None:
                    images = x
                else:
                    for j in range(len(images)):
                        images[j] = torch.cat([images[j], x[j]], 0)

                if labels is None:
                    labels = label
                else:
                    labels = torch.cat([labels, label], 0)

        images = tuple([item.to(device) for item in images])
        f = self.model.module.save_features(images)
        self.writer.add_embedding(f, metadata=labels, label_img=images[2], tag='f')
