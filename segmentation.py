import os
import torch

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from metainfo.schedule import get_schedule
from model.interface import obtain_net
from datamanager.dataset import SegDataset
from utils.tools import grid_search, get_current_time


def obtain_data_loader(mode, save_path, batch_size=32, paired=True, aug=''):
    dataset = SegDataset(mode, paired, aug=aug)
    dataset.load(save_path)
    dataset.set_iter_data()

    train_set, test_set = dataset.split([0.8, 1.0])

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader


def train(net, train_loader, test_loader, mode: str = 'E', epoch=150, lr=0.005, warm=15,
          save_path='../models/segmentation'):
    best_measure = 0.0
    best_result = None

    optimizer, scheduler = get_schedule('mix', net.model.parameters(), epoch, lr, warm)

    for i in range(epoch):
        net.train_loop(i, train_loader, optimizer)
        scheduler.step()
        if i % 10 == 0:
            result = net.test_loop(i, test_loader)
            new_measure = sum([result[k]['JS'][0] for k in mode]) / len(mode)

            if new_measure > best_measure:
                best_result = result
                best_measure = new_measure

                if net.is_parallel:
                    state = {'param': net.model.module.state_dict(), 'epoch': i,
                             'optimizer': optimizer.state_dict()}
                else:
                    state = {'param': net.model.state_dict(), 'epoch': i,
                             'optimizer': optimizer.state_dict()}

                torch.save(state, save_path)

    return best_result


@grid_search()
def main(param):
    epoch = param['epoch']
    lr = param['lr']
    warm = param['warm']
    mode = param['mode']
    batch_size = param['batch_size']
    stage = param['stage']
    init_ch = param['init_ch']

    is_parallel = param['is_parallel']
    is_post_process = param['is_post_process']
    aug = param['aug']
    paired = param['paired']

    # writer_folder = param['writer_path']
    loss_fn = param['loss_fn']
    model_func = param['model_func']
    dataset_path = param['dataset_path']
    record_file = param['record_file_path']

    current_time = get_current_time()
    model_save_path = os.path.join('../models/segmentation/%s.pth' % current_time)

    writer = SummaryWriter(os.path.join('../tensorboard/video/segmentation-%s' % current_time))
    train_loader, test_loader = obtain_data_loader(mode, dataset_path, batch_size, paired=paired, aug=aug)

    net_param = {'model': model_func, 'net': 'mixnet', 'writer': writer, 'mode': mode, 'is_parallel': is_parallel,
                 'init_ch': init_ch, 'stage': stage, 'loss_fn': loss_fn, 'is_post_process': is_post_process}
    net = obtain_net(**net_param)

    best_result = train(net, train_loader=train_loader, test_loader=test_loader, mode=mode,
                        epoch=epoch, lr=lr, warm=warm, save_path=model_save_path)

    content = {'TimeStamp': [current_time] + [''] * (len(mode) - 1)}
    content.update({k: [str(param[k])] + [''] * (len(mode) - 1)
                    for k in filter(lambda x: 'path' not in x, param.keys())})
    result = {k: ['%.4f+-%.4f' % (best_result[m][k][0], best_result[m][k][1])
                  for m in mode] for k in best_result[mode[0]].keys()}
    content['mode'] = [m for m in mode]
    content.update(result)

    df = pd.DataFrame(content)
    df.to_csv(record_file, mode='a', header=True, index=False)

    return


if __name__ == '__main__':
    basic_param = {'epoch': 251, 'batch_size': 6, 'mode': 'EBF', 'paired': True, 'loss_fn': 'dice_bce',
                   'is_parallel': True, 'is_post_process': False, 'init_ch': 64, 'stage': 5, 'lr': 0.005, 'warm': 15,
                   'model_func': 'msunet', 'aug': 'degrees=90, translate=(0.2, 0.2), scale=(0.6, 1.2)',
                   'dataset_path': '../data/video/mm-ete/20200902-dataset.pkl',
                   'record_file_path': '../record/MultiSite-Segmentation.csv'}

    # experiment for backbone compare
    # search_param = {'model_func': ['msattunet']}
    # main(basic_param, search_param)

    # experiment for multi-modal effectiveness
    # print('experiment for multi-modal start!')
    # search_param = {'mode': ['E', 'B']}
    # main(basic_param, search_param)

    # experiment for data augmentation
    # print('experiment for data augmentation start!')
    # search_param = {'aug': ['degrees=90', 'degrees=0, translate=(0.2, 0.2)', 'degrees=0, scale=(0.6, 1.2)']}
    # main(basic_param, search_param)

    # experiment for loss function
    # search_param = {'loss_fn': ['dice_bce', 'dice', 'ms_dice']}
    # main(basic_param, search_param)

    # experiment for data paired
    # search_param = {'paired': [True, False]}
    # main(basic_param, search_param)

    # print('experiment for post process!!')
    print('experiment for post process start!!')
    search_param = {'is_post_process': [True]}
    main(basic_param, search_param)
