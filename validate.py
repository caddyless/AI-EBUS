import torch
import joblib
import argparse
import random
from os.path import join

import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.tools import convert_dict_yaml
from datamanager import obtain_dataset, obtain_data_loader
from model.interface import obtain_net


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='validation', type=str)


# This file provides validate function to existed model, related params as follow:
# validate type: five-fold, one-fold
# data type: whole, subset, test_loader
# parse feature: true, false


def common_validate(dataset_path, database_path, identity_file, mode, datatype_arrange, model_dir):
    datatype_arrange = {'C': datatype_arrange}
    dataset = obtain_dataset(dataset_path, database_path, mode, datatype_arrange=datatype_arrange)
    map_dict = {item.info['id']: i for i, item in enumerate(dataset.iter_data)}

    with open(identity_file, 'r') as f:
        content = f.read()
        identities = content.split(',')
        identities = map(int, identities)

    indices = [map_dict[id_] for id_ in identities if id_ in map_dict.keys()]
    dataset = Subset(dataset, indices)
    data_loader = DataLoader(dataset, batch_size=16, num_workers=2)

    param_path = join(model_dir, 'model', 'best-epoch.pth')

    result, trace_df = validate(data_loader, model_path=param_path, param_path=param_path, is_trace=True)
    trace_df.to_csv('../record/ct-from-scratch.csv')
    return


def result_validate(model_dir: str, metric: str,  dataset_path, database_path, allocate_file_path):
    mmv_dataset = obtain_dataset(dataset_path, database_path, 'BEF')
    data_loader = obtain_data_loader(mmv_dataset, allocate_file_path=allocate_file_path)
    for i, (train, val, test) in enumerate(data_loader):
        net_path = model_dir + '%d/model/initialized.model' % i
        param_path = model_dir + '%d/model/best-%s.pth' % (i, metric)
        result = validate(test, model_path=net_path, param_path=param_path)
        save_dir = Path('../Final-Model/best-result/fold-%d' % i)
        save_dir.mkdir(parents=True, exist_ok=True)
        for m in ['B', 'F', 'E', 'Fusion']:
            result[m].save_data(str(save_dir / ('%s-epoch-prediction.csv' % m)))
    return


def validate(data_loader, model_path, param_path, is_trace: bool = False):
    model_path = Path(model_path)
    param = torch.load(param_path)['param']
    net_param = convert_dict_yaml(yaml_file=str(model_path.parent / 'net_config.yaml'))
    net = obtain_net(**net_param)
    if hasattr(net.model, 'module'):
        net.model.module.load_state_dict(param)
    else:
        net.model.load_state_dict(param)

    return net.test_loop(epoch=0, test_loader=data_loader, is_trace=is_trace)


def parse_feature(dataset_path, database_path, model_path, param_path):
    data_loader = obtain_data_loader(dataset_path, database_path, 'BEF', True)
    param = torch.load(param_path)['param']
    net = joblib.load(model_path)
    if hasattr(net.model, 'module'):
        net.model.module.load_state_dict(param)
    else:
        net.model.load_state_dict(param)

    feature, y_true = net.parse_feature(data_loader)

    return feature, y_true


def five_fold_parse(model_dir: str, feature_dir: str, writer_dir: str, metric: str, dataset_path: str,
                    database_path: str):
    writer = SummaryWriter(writer_dir)

    task_root_dir = model_dir
    save_dir = Path(feature_dir)
    save_dir.mkdir()
    for i in range(5):
        net_path = task_root_dir + '%d/model/initialized.model' % i
        param_path = task_root_dir + '%d/model/best-%s.pth' % (i, metric)
        # parse_feature('/media/sandisk/20201025-tumor-mmv.dataset', '/media/sandisk/20201025-tumor-database.h5',
        #               model_path=net_path, param_path=param_path)

        feature, y_true = parse_feature(dataset_path, database_path, model_path=net_path, param_path=param_path)
        for m in ['B', 'E', 'F', 'Fusion']:
            print(feature[m].shape)
            writer.add_embedding(feature[m], metadata=y_true, tag='Fold-%d-%s' % (i, m))

        np.savez(str(save_dir / ('%d-extracted_feature.npz' % i)), **feature)
        np.save(str(save_dir / ('%d-extracted_label.npy' % i)), y_true)

    writer.close()
    return


def five_fold_validate(model_dir: str, metric: str, dataset_path: str, database_path: str, k_fold: int = 5):
    dataset = obtain_dataset(dataset_path, database_path, 'BEF')
    data_loader = DataLoader(dataset, batch_size=32, num_workers=2)
    for i in range(k_fold):
        root_dir = model_dir
        net_path = root_dir + '%d/model/initialized.model' % i
        param_path = root_dir + '%d/model/best-%s.pth' % (i, metric)
        validate(data_loader, model_path=net_path, param_path=param_path)
    return


def merge_predictions(y_true, y_score):
    assert (y_true[0] == y_true[1]).all(), 'y_true not aligned!'
    y_true = y_true[0]
    y_score = [np.expand_dims(item, axis=1) for item in y_score]
    y_score = np.concatenate(y_score, axis=1)
    predict = ((y_score > 0.5).mean(1) > 0.5).astype(np.int)
    acc = (predict == y_true).sum() / y_true.shape[0]
    print(acc)
    return


def five_fold_validate_subset(model_dir: str, metric: str, dataset_path: str, database_path: str, save_dir,
                              datatype_arrange, k_fold: int = 5, num_test_sample: int = 30, is_trace: bool = True,
                              mode='BEF', **kwargs):
    dataset = obtain_dataset(dataset_path, database_path, mode, datatype_arrange=datatype_arrange)

    file_path = Path('../raw-data/old-sub_indices.txt')
    if file_path.is_file():
        with open(str(file_path), 'r') as f:
            content = f.readline()
            identities = list(map(int, content.split(',')))
    else:
        id_pool = []
        for item in dataset.iter_data:
            id_pool.append(int(item.info['id']))
        identities = random.sample(id_pool, num_test_sample)
        with open(str(file_path), 'w') as f:
            f.write(','.join(list(map(str, identities))) + '\n')

    # identities += [174, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194]
    # print(identities)
    new_add = [165, 176, 181, 209, 212, 215, 217, 227, 231, 244, 316, 330, 337]
    indices = []
    validate_indices = kwargs['identity_file']
    with open(validate_indices, 'r') as f:
        content = f.read()
        identities = list(map(int, content.split(',')))
    for i, item in enumerate(dataset.iter_data):
        if int(item.info['id']) in identities:
            indices.append(i)
        # if int(item.info['id']) in new_add:
        #     indices.append(i)
    dataset = Subset(dataset, indices)
    print(len(indices))
    data_loader = DataLoader(dataset, batch_size=16, num_workers=2)
    y_true = []
    y_score = []

    result_save_dir = Path(save_dir)
    result_save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(k_fold):
        root_dir = model_dir
        net_path = root_dir + '%d/model/initialized.model' % i
        param_path = root_dir + '%d/model/best-%s.pth' % (i, metric)
        result, trace_df = validate(data_loader, model_path=net_path, param_path=param_path, is_trace=is_trace)
        trace_df.to_csv(str(result_save_dir / ('%d.csv' % i)))
        # if 'Fusion' in result.keys():
        y_true.append(result['C'].raw_data['y_true'])
        y_score.append(result['C'].raw_data['y_score'])

    merge_predictions(y_true, y_score)
    return


if __name__ == '__main__':
    args = parser.parse_args()

    if args.data == 'validation':
        conf_file = 'conf/validate/validate-template.yaml'
    else:
        conf_file = args.data

    param = convert_dict_yaml(yaml_file=conf_file)

    if param['result_validate']:
        result_validate(**param['basic_param'])
    else:
        if param['validate']:
            if param['subset']:
                five_fold_validate_subset(**param['basic_param'])
            else:
                common_validate(**param['basic_param'])
                # five_fold_validate(**param['basic_param'])

        if param['parse_feature']:

            five_fold_parse(**param['basic_param'], **param['parse_feature_param'])
