import torch

import numpy as np

from model.classify import MMVideoNet


class ContrastMMVideo(MMVideoNet):

    def __init__(self, model_func, weights, num_class=2, is_parallel=False, **kwargs):
        super().__init__(model_func, weights, num_class=num_class, is_parallel=is_parallel, **kwargs)

    def set_forward(self, x):
        inputs = self.to_cuda(x)
        return self.model(inputs)

    def set_loss(self, x, label, identity, epoch, is_trace, *args):

        x = self.to_cuda(x)
        _ = self.model(x)
        sub_loss = [m.local_loss for m in self.sub_loss_module]
        loss = sum(sub_loss)
        return loss

    def train_loop(self, epoch, train_loader, optimizer, writer=None, is_trace=False):
        print_freq = self.print_freq
        avg_loss = 0

        self.model.train()
        self.set_dataset(train_loader, True)
        for i, (id_, x, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.set_loss(x, label, id_, epoch, is_trace)
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * i + loss.data.item()) / float(i + 1)

            if i % print_freq == 0:
                print('Epoch {:3d} | Batch {:3d}/{:3d} | Loss {:6f} | lr {:6f}'.format(
                    epoch, i, len(train_loader), avg_loss, optimizer.param_groups[0]['lr']))

        if writer is not None:
            writer.add_scalar('loss', avg_loss, epoch)
            self.write_param(epoch, writer)

        return

    def parse_feature(self, data_loader):
        length = len(data_loader.dataset)
        keys = self.mode
        y_true = np.zeros(length, dtype=np.int8)
        features = {}

        model = self.model.module if hasattr(self.model, 'module') else self.model
        for m in keys:
            if m in 'BEF':
                dims = model.branches[m].out_channels
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
                    if m in 'BEF':
                        features[m][index: index + num_sample] = f[m].cpu().numpy()
                y_true[index: index + num_sample] = label.numpy()
                index += num_sample

        return features, y_true
