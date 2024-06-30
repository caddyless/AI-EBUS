import pynvml
import torch
import argparse
import os

"""
It mainly comprises of three modules: data, model and method. The options for data is gray, egraph, bflow.
--v indicates generate dataset from videos. --regenerate indicates re-generate the dataset. Model indicates 
the architecture of backbone, such as simplenet and graynet. Method indicates the paradigm of training, such as 
typical.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='e', type=str)
parser.add_argument('--source', default='../data/new/crop_e', type=str)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('-s', action='store_true')
parser.add_argument('-observe', action='store_true')
parser.add_argument('-ag', action='store_true')
parser.add_argument('-record', action='store_true')
parser.add_argument('-calibrate', action='store_true')
parser.add_argument('-resume', action='store_true')
parser.add_argument('--backbone', default='simplenet2', type=str)
parser.add_argument('--dataset', default='folder', type=str)
parser.add_argument('--net', default='typical', type=str)
parser.add_argument('--kfold', default=6, type=int)
parser.add_argument('--loss', default='CCE', type=str)
parser.add_argument('--q', default=0.2, type=float)
parser.add_argument('--loss_weight', default=1.0, type=float)
parser.add_argument('--warm', default=20, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--remark', default='无', type=str)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--multitest', default=0, type=int)
parser.add_argument('--discard', default='no', choices=['black_list', 'random', 'no'])
parser.add_argument('--compare', default='ela')
parser.add_argument('--epoch', default=250, type=int)
parser.add_argument('--cw', default=1.0, type=float)
parser.add_argument('--bw', default=1.2, type=float)
parser.add_argument('--fw', default=1.2, type=float)

args = parser.parse_args()


# 检测GPU情况
def get_device():
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        devicecount = pynvml.nvmlDeviceGetCount()
        available_device = []
        for i in range(devicecount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            print('GPU', i, ':', pynvml.nvmlDeviceGetName(handle))
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            ratio = info.free / info.total
            if ratio < 0.9:
                available_device.append(str(i))
            print(
                'Memory Total:%.1f GB   Memory Free:%.1f GB   Ratio:%.2f' %
                (info.total / 1e9, info.free / 1e9, 1 - info.free / info.total))
        print('Total %d devices are available' % len(available_device))
        if len(available_device) == 0:
            print('All devices are occupied')
            return False
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


command = []
for k, v in args.__dict__.items():
    if v != parser.get_default(k):
        if isinstance(v, bool):
            if v:
                command.append(k)
        elif isinstance(v, str):
            if k == 'source':
                command.append(os.path.basename(v))
            else:
                command.append(v)
        else:
            command.append(str(v))
command = '-'.join(command)
