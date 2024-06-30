import os
import torch
import joblib
import warnings
import argparse

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from metainfo.schedule import get_schedule
from utils.tools import get_current_time, grid_search, convert_dict_yaml, set_seed
from model.interface import obtain_net
from datamanager.interface import obtain_data_loader, obtain_dataset
from metric.cv import MultiModalSummary


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./conf/train-datatype.yaml', type=str)


def train(net, train_loader, val_loader, save_keys, epoch=100, lr=1e-3, lr_times=100, warm=10, last_e=10, final_lr=1.4e-3, save_dir='../models', writer=None, batch_times: int = 1, finetune=True):
    save_dict = dict.fromkeys(save_keys, 0.0)

    optimizer, scheduler = get_schedule('folder', net, epoch, lr, warm,
                                        last_epoch=last_e, final_lr=final_lr, finetune=finetune, lr_times=lr_times)

    train_trace = pd.DataFrame(columns=('Identity', 'Epoch', 'Mode', 'Gap'))
    test_trace = pd.DataFrame(columns=('Identity', 'Epoch', 'Mode', 'Gap'))

    for i in range(epoch):
        train_t = net.train_loop(i, train_loader, optimizer, writer=writer, batch_times=batch_times)
        train_trace = train_trace.append(train_t)

        scheduler.step()
        result, test_t = net.test_loop(i, val_loader, writer=writer, is_trace=True)
        test_trace = test_trace.append(test_t)

        existed_mode = list(result.keys())
        if 'Fusion' in existed_mode:
            result = result['Fusion']
        else:
            result = result[existed_mode[0]]

        params = net.model.module.state_dict() if hasattr(net.model, 'module') else net.model.state_dict()
        state = {'param': params, 'epoch': i,
                 'optimizer': optimizer.state_dict()}

        for k in save_dict.keys():
            if result.result[k] > save_dict[k]:
                save_dict[k] = result.result[k]
                path = os.path.join(save_dir, 'best-%s.pth' % k)
                torch.save(state, path)

    return train_trace, test_trace


def evaluation(net, val_loader, test_loader, metrics, modes, model_folder, result_dir, recorder, is_validate):
    for k in metrics:
        print('\n{:*^160}\n'.format('evaluation on best-%s' % k))

        path = os.path.join(model_folder, 'best-%s.pth' % k)
        resume = torch.load(path)

        if hasattr(net.model, 'module'):
            net.model.module.load_state_dict(resume['param'])
        else:
            net.model.load_state_dict(resume['param'])

        if is_validate:
            result = net.calibrate_model(resume['epoch'], test_loader, val_loader)
        else:
            result = net.test_loop(resume['epoch'], test_loader)

        recorder.update(k, result)
        for m in modes:
            result[m].save_data(os.path.join(result_dir, '%s-%s-prediction.csv' % (m, k)))

            print('\n{:*^160}\n'.format('Mode {} best-{} is {:6.4f}'.format(m, k, result[m].result[k])))
    return


@grid_search
def main(default_param, local_param):
    param = {}
    param.update(default_param)
    param.update(local_param)

    mode = param['mode']
    k_fold = param['k-fold']
    is_parallel = param['is_parallel']
    data_way = param['data_way']

    epoch = param['epoch']
    batch_size = param['batch_size']
    batch_times = param['batch_times']
    lr = param['lr']
    warm = param['warm']
    last_e = param['last_e']
    final_lr = param['final_lr']
    lr_times = param['lr_times']

    dataset_path = param['dataset_path']
    allocate_file_path = param['allocate_file_path']
    database_path = param['database_path']
    record_file_path = param['record_file_path']
    remark = param['remark']

    e_params = param['e_params']
    b_params = param['b_params']
    f_params = param['f_params']
    branch_params = {'E': e_params, 'B': b_params, 'F': f_params}
    fusion_way = param['fusion_way']

    if data_way == 'five-fold-with-validate':
        metrics = param['metrics']
    else:
        metrics = ['epoch']

    is_norm = param['normalize']
    dta = param['datatype_arrange']
    weights = param['weights']

    model_save_folder = os.path.join(param['model_save_folder'], str(param['select_fold'])) if not param['from_scratch'] else None

    current_time = get_current_time()
    result_modes = list(mode) if len(mode) == 1 else (list(mode) + ['Fusion'])

    existed_param = list(local_param.keys()) + ['mode', 'k-fold', 'remark']
    residual_param = {k: v for k, v in default_param.items() if k not in existed_param}
    summary = MultiModalSummary(task='classify', modes=result_modes, k_fold=k_fold, timestamp=current_time,
                                remark=remark, select_metric=metrics, default_params=residual_param,
                                ergodic_param=local_param)

    root_dir = Path('../task-checkpoint/video/classification-%s' % current_time)
    if not root_dir.is_dir():
        root_dir.mkdir(parents=True)

    for i, data in enumerate(obtain_data_loader(dataset_path, database_path, mode, is_norm, dta,
                                                k_fold=k_fold, batch_size=batch_size,
                                                allocate_file_path=allocate_file_path)):
        print('*' * 100 + 'Fold %d' % i + '*' * 100)
        if len(data) == 3:
            train_loader, val_loader, test_loader = data
        else:
            train_loader, test_loader = data
            val_loader = test_loader

        recorder = summary.recorders[i]

        model_dir = root_dir / ('%d' % i) / 'model'
        model_dir.mkdir(exist_ok=True, parents=True)

        tensorboard_dir = root_dir / ('%d' % i) / 'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True, parents=True)

        result_dir = root_dir / ('%d' % i) / 'result'
        result_dir.mkdir(exist_ok=True, parents=True)

        writer = SummaryWriter(tensorboard_dir)
        net_param = {'is_parallel': is_parallel, 'model': 'wholemodel', 'net': 'union', 'weights': weights,
                     'mode': mode, 'branch_params': branch_params, 'model_folder': model_save_folder, 'fusion_way': fusion_way}

        convert_dict_yaml(diction=net_param, yaml_file=str(model_dir / 'net_config.yaml'))

        net = obtain_net(**net_param)
        joblib.dump(net, str(model_dir / 'initialized.model'))

        # add the visualization of net
        # dump_input = mmv_dataset[0][1]
        # dump_input = {k: (v[0].unsqueeze_(0), v[1].unsqueeze_(0)) for k, v in dump_input.items()}
        # writer.add_graph(net.model, dump_input)

        with recorder:
            train_trace, test_trace = train(net, train_loader, val_loader, save_keys=metrics, epoch=epoch, lr=lr,
                                            warm=warm, last_e=last_e, final_lr=final_lr, save_dir=model_dir,
                                            writer=writer, batch_times=batch_times, lr_times=lr_times, finetune=not param['from_scratch'])

        train_trace.to_csv(str(result_dir / 'train_trace.csv'))
        test_trace.to_csv(str(result_dir / 'test_trace.csv'))

        evaluation(net, val_loader, test_loader, metrics=metrics, modes=result_modes, model_folder=model_dir,
                   result_dir=result_dir, recorder=recorder, is_validate=(data_way == 'five-fold-with-validate'))

        writer.close()

    summary.write(record_file_path)


if __name__ == '__main__':
    set_seed()
    warnings.filterwarnings('ignore', category=UserWarning)
    args = parser.parse_args()

    params = convert_dict_yaml(yaml_file=args.data)

    basic_param = convert_dict_yaml(yaml_file='conf/train/default-param.yaml')
    search_param = params['search_param']

    trans_param = []
    for k, v in search_param.items():
        if len(v) == 1:
            trans_param.append(k)
    for k in trans_param:
        basic_param.update({k: search_param[k][0]})
        search_param.pop(k, -1)

    repeat_time = params['repeat_time']
    search_mode = params['search_mode']

    for i in range(repeat_time):
        main(basic_param, search_param, search_mode)
