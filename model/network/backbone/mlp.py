from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: list, activation: str = 'relu'):
        super().__init__()
        num_uint = [in_channels] + hidden + [out_channels]
        length = len(num_uint) - 1
        if activation == 'none':
            modules = [nn.Linear(num_uint[i], num_uint[i+1]) for i in range(length)]
        else:
            if activation == 'relu':
                act = nn.ReLU
                act_param = {'inplace': True}
            elif activation == 'sigmoid':
                act = nn.Sigmoid
                act_param = {}
            else:
                raise NotImplemented('Unknown activation %s' % activation)

            modules = []
            for i in range(length):
                modules.append(nn.Linear(num_uint[i], num_uint[i+1]))
                if i != (length - 1):
                    modules.append(act(**act_param))

        self.module = nn.Sequential(*modules)
        self.out_channel = out_channels

    def forward(self, x):
        return self.module(x)
