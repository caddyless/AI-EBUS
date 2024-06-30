from random import shuffle

from pathlib import Path


def generate_allocation(indices: list, k_fold: int = 5, with_validate: bool = False):
    shuffle(indices)
    length = len(indices)
    average = length // k_fold
    residual = length % k_fold

    data = []
    index = 0
    for i in range(k_fold):
        if residual > 0:
            sep = average + 1
            residual -= 1
        else:
            sep = average

        data.append(indices[index: (index + sep)])
        index += sep

    if with_validate:
        allocation = [{'train': data[i], 'val': data[(i + 1) % k_fold], 'test': data[(i + 2) % k_fold]}
                      for i in range(k_fold)]
    else:
        allocation = [{'train': data[i], 'val': None, 'test': data[(i + 1) % k_fold]} for i in range(k_fold)]

    return allocation


class Allocations(object):

    def __init__(self, file_path: str, allocation=None):
        self.file_path = file_path
        self.keys = ('train', 'val', 'test')
        file_path = Path(file_path)
        if file_path.is_file():
            self.allocation = self.read_file()
        else:
            self.allocation = allocation
            self.write_file(allocation)

    def generate_data_allocate(self):
        for item in self.allocation:
            data = [t for t in item.values() if t is not None]
            yield data

    def write_file(self, allocation: list):
        k_fold = len(allocation)
        keys = self.keys

        with open(self.file_path, 'w') as f:
            f.write('k-fold: %d\n' % k_fold)

            for i in range(k_fold):
                for j, k in enumerate(keys):
                    f.write('%s:\n' % k)
                    content = allocation[i][k]
                    if content is not None:
                        f.write(','.join(list(map(str, allocation[i][k]))))
                    f.write('\n')
        return

    def read_file(self):
        with open(self.file_path, 'r') as f:
            k_fold = int(f.readline()[-2])
            content = f.readlines()
            allocation = []
            keys = self.keys
            for fold in range(k_fold):
                data = {}
                for j, k in enumerate(keys):
                    current_ele = content[fold * 6 + j * 2 + 1]
                    current_ele = current_ele.replace('\n', '')
                    current_ele = current_ele.split(',')
                    if len(current_ele) == 1 and current_ele[0] == '':
                        data[k] = None
                    else:
                        item = list(map(int, current_ele))
                        data[k] = item
                allocation.append(data)
        return allocation
