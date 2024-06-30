import os
import torch
import joblib
import warnings
import argparse

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from metainfo.schedule import get_schedule
from utils.tools import get_current_time, grid_search, convert_dict_yaml, set_seed
from model.interface import obtain_net
from datamanager.interface import DataInterface
from datamanager.dataset import MMVideoDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./conf/train-datatype.yaml', type=str)


def obtain_data_loader(mmv_dataset: MMVideoDataset, k_fold=5, batch_size=4,
                       allocate_file_path='../raw-data/20201001-mmv-5-fold.txt'):
    d = DataInterface(dataset=mmv_dataset, require_ag=True, num_class=2, k_fold=k_fold)

    for data in d.dataset_generator(allocate_file_path):
        train_set, val_set, test_set = data
        train_set = train_set + val_set
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, shuffle=False)

        yield train_loader, test_loader


def train(net, train_loader, epoch=100, lr=1e-3, warm=10, last_e=10, final_lr=1.4e-3,
          save_dir='../models', writer=None):

    optimizer, scheduler = get_schedule('folder', net.model.parameters(), epoch, lr, warm, last_epoch=last_e,
                                        final_lr=final_lr)

    for i in range(epoch):
        net.train_loop(i, train_loader, optimizer, writer)
        scheduler.step()
        params = net.model.module.state_dict() if hasattr(net.model, 'module') else net.model.state_dict()
        checkpoint = {'param': params, 'epoch': i, 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

    return


def evaluation(net, test_loader, model_folder):
    path = os.path.join(model_folder, 'checkpoint.pth')
    resume = torch.load(path)

    if hasattr(net.model, 'module'):
        net.model.module.load_state_dict(resume['param'])
    else:
        net.model.load_state_dict(resume['param'])

    return net.parse_feature(test_loader)


@grid_search
def main(default_param, local_param):
    param = {}
    param.update(default_param)
    param.update(local_param)

    mode = param['mode']
    k_fold = param['k-fold']
    is_parallel = param['is_parallel']

    net_type = param['net']
    epoch = param['epoch']
    batch_size = param['batch_size']
    lr = param['lr']
    warm = param['warm']
    last_e = param['last_e']
    final_lr = param['final_lr']

    dataset_path = param['dataset_path']
    allocate_file_path = param['allocate_file_path']
    database_path = param['database_path']

    e_params = param['e_params']
    b_params = param['b_params']
    f_params = param['f_params']
    branch_params = {'E': e_params, 'B': b_params, 'F': f_params}

    unsupervised = param['unsupervised']
    is_norm = param['normalize']
    dta = param['datatype_arrange']
    weights = param['weights']

    current_time = get_current_time()

    root_dir = Path('../task-checkpoint/video/classification-%s' % current_time)
    if not root_dir.is_dir():
        root_dir.mkdir(parents=True)

    # construct dataset
    mmv_dataset = MMVideoDataset(database_path, mode, normalize=is_norm, datatype_arrange=dta)
    mmv_dataset.load(dataset_path)
    print('There is %d videos here.' % len(mmv_dataset.raw_data))
    mmv_dataset.set_iter_data('BEF')
    print('This dataset length under mode %s is %d' % (str(mode), len(mmv_dataset)))

    for i, data in enumerate(obtain_data_loader(mmv_dataset, k_fold=k_fold, batch_size=batch_size,
                                                allocate_file_path=allocate_file_path)):
        print('*' * 100 + 'Fold %d' % i + '*' * 100)
        train_loader, test_loader = data

        model_dir = root_dir / ('%d' % i) / 'model'

        model_dir.mkdir(exist_ok=True, parents=True)

        tensorboard_dir = root_dir / ('%d' % i) / 'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True, parents=True)

        result_dir = root_dir / ('%d' % i) / 'result'
        result_dir.mkdir(exist_ok=True, parents=True)

        writer = SummaryWriter(tensorboard_dir)

        net_param = {'is_parallel': is_parallel, 'model': 'wholemodel', 'net': net_type, 'weights': weights,
                     'mode': mode, 'branch_params': branch_params, 'unsupervised': unsupervised}

        convert_dict_yaml(diction=net_param, yaml_file=str(model_dir / 'net_config.yaml'))

        net = obtain_net(**net_param)
        joblib.dump(net, str(model_dir / 'initialized.model'))

        train(net, train_loader, epoch=epoch, lr=lr, warm=warm, last_e=last_e, final_lr=final_lr, save_dir=model_dir,
              writer=writer)

        feature, y_true = evaluation(net, test_loader, model_folder=model_dir)

        for m in ['B', 'E', 'F']:
            print(feature[m].shape)
            writer.add_embedding(feature[m], metadata=y_true, tag='Fold-%d-%s' % (i, m))

        np.savez(str(result_dir / 'extracted_feature.npz'), **feature)
        np.save(str(result_dir / 'extracted_label.npy'), y_true)

        writer.close()

    return


if __name__ == '__main__':
    set_seed()
    warnings.filterwarnings('ignore', category=UserWarning)
    args = parser.parse_args()

    param = convert_dict_yaml(yaml_file=args.data)

    basic_param = convert_dict_yaml(yaml_file='conf/train/default-param.yaml')

    search_param = param['search_param']
    repeat_time = param['repeat_time']
    search_mode = param['search_mode']

    for i in range(repeat_time):
        main(basic_param, search_param, search_mode)
