import os

from torch.utils.data import DataLoader

from datamanager.dataset import MMVideoDataset
from datamanager.DataAllocation import Allocations, generate_allocation


class DataInterface(object):
    def __init__(self, dataset, k_fold=5):
        self.k_fold = k_fold
        self.dataset = dataset
        self.map_dict = {item.info['id']: i for i, item in enumerate(dataset.iter_data)}

    def dataset_generator(self, allocate_file_path: str, **kwargs):
        dataset = self.dataset
        if os.path.isfile(allocate_file_path):
            a = Allocations(file_path=allocate_file_path, **kwargs)
        else:
            indices = [item.info['id'] for item in self.dataset.iter_data]
            allocation = generate_allocation(indices, self.k_fold)
            a = Allocations(allocate_file_path, allocation=allocation)

        for data in a.generate_data_allocate():
            yield dataset.split(self.id_to_indices(data))

    def id_to_indices(self, identity):
        map_dict = self.map_dict
        new_indices = []
        for item in identity:
            new_indices.append([map_dict[k] for k in item])
        return new_indices


def obtain_dataset(dataset_path: str, database_path: str, mode: str, is_norm: bool = True,
                   datatype_arrange: dict = None):

    if datatype_arrange is None:
        datatype_arrange = {'B': 'raw-img', 'F': 'ROI', 'E': 'elastic-hist', 'C': 'h-roi-max'}

    # construct dataset
    mmv_dataset = MMVideoDataset(database_path, mode, normalize=is_norm, datatype_arrange=datatype_arrange)
    mmv_dataset.load(dataset_path)
    print('There is %d videos here.' % len(mmv_dataset.raw_data))
    mmv_dataset.set_iter_data(mode)
    print('This dataset length under mode %s is %d' % (str(mode), len(mmv_dataset)))

    return mmv_dataset


def obtain_data_loader(dataset_path: str, database_path: str, mode: str, is_norm: bool = True,
                       datatype_arrange: dict = None, k_fold=5, batch_size=4,
                       allocate_file_path='../raw-data/20201001-mmv-5-fold.txt'):
    dataset = obtain_dataset(dataset_path, database_path, mode, is_norm, datatype_arrange)
    d = DataInterface(dataset, k_fold)
    for item in d.dataset_generator(allocate_file_path):
        data_loaders = []
        for i, ds in enumerate(item):
            if i == 0:
                shuffle = True
            else:
                shuffle = False
            data_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4))
        yield data_loaders
