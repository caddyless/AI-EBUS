import random
import os
import traceback
import torch
import time

import numpy as np
import matplotlib.pyplot as plt

from ruamel import yaml
from timeit import default_timer as timer
from itertools import product, combinations


def convert_dict_yaml(diction: dict = None, yaml_file: str = None):
    left = isinstance(diction, dict)
    right = isinstance(yaml_file, str)

    if (~left) & right:
        # read yaml file to dict
        with open(yaml_file, 'r', encoding='utf-8') as f:
            diction = yaml.load(f.read(), Loader=yaml.Loader)
        return diction
    elif left & right:
        # save diction to yaml_file
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(diction, f, Dumper=yaml.RoundTripDumper)

        return
    else:
        return


def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def combination(params: dict):
    keys = list(params.keys())
    for item in product(*params.values()):
        yield dict(zip(keys, item))


def pair(params: dict):
    keys = list(params.keys())
    length = len(params[keys[0]])
    for i in range(length):
        yield dict(zip(keys, [params[k][i] for k in keys]))


def travel(iterable):
    length = len(iterable)
    for i in range(1, length + 1):
        for item in combinations(iterable, i):
            yield item


def grid_search(func):

    def wrapper(basic_param, search_param, mode):
        info_collector = []
        iterator = combination(search_param) if mode == 'combination' else pair(search_param)
        for term in iterator:
            try:
                func(basic_param, term)
            except Exception:
                error_info = traceback.format_exc()
                info_collector.append((str(term), error_info))
                print(error_info)
                continue
            info_collector.append((str(term), 'Success!'))

        print('The conduct result as follow:')
        for item in info_collector:
            print('Term: %s | Status: %s' % (item[0], item[1]))

        return

    return wrapper


def clock(func):
    def wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print('Time consume %f' % (end - start))
        return result
    return wrapper


def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def display(img_array: (np.ndarray, list), name: list = None, cmap: str = 'viridis') -> None:
    """
    This function display given img array or list with matplotlib.
    If ima_array is ndarray, the dims of img_array must be 3, 4 or 5. If dims == 3, it shows 1 img. if dims == 4,
        it shows images in one row. if dims == 5, it shows images in rows and columns.

    If ima_array is list, it supposed to be a list of ndarray.

    The datatype of input must be uint8

    :rtype: None
    """
    if isinstance(img_array, np.ndarray):

        assert img_array.dtype == np.uint8, 'The dtype of input must be uint8'

        shape = img_array.shape
        assert shape[-1] in (1, 3), 'The channel of img must be 1 or 3'

        if (length := len(shape)) == 3:
            fig, ax = plt.subplots()

            ax.imshow(img_array, cmap=cmap)

        elif length == 4:
            num_img = shape[0]
            fig, axes = plt.subplots(1, num_img, figsize=(num_img * 3, 3))
            for i in range(num_img):
                if name is not None:
                    axes[i].set_title(name[i])
                axes[i].imshow(img_array[i], cmap=cmap)

        elif length == 5:
            num_row = shape[0]
            num_column = shape[1]
            fig, axes = plt.subplots(num_row, num_column, figsize=(num_column * 3, num_row * 3))
            for i in range(num_row):
                for j in range(num_column):
                    axes[i * num_column + j].imshow(img_array[i, j], cmap=cmap)

        else:
            raise ValueError('The dims of input img_array expected to be (3, 4, 5), but found %d' % length)

    elif isinstance(img_array, list):

        num_img = len(img_array)
        fig, axes = plt.subplots(1, num_img, figsize=(num_img * 3, 3))
        for i in range(num_img):
            axes[i].imshow(img_array[i])
            if name is not None:
                axes[i].set_title(name[i], cmap=cmap)

    else:
        raise ValueError('Unknown type %s of img_array' % str(type(img_array)))

    return
