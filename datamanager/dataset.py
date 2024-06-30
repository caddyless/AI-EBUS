import torch
import numpy as np
import os
import joblib

import pandas as pd
import cv2 as cv

from torch.utils.data import Dataset, Subset
from functools import reduce
from multiprocessing import Pool, Manager, cpu_count
from torchvision import transforms as tf
from tqdm import tqdm
from abc import abstractmethod
from utils.tools import travel
from metainfo.default import B_IMAGE_SIZE, F_IMAGE_SIZE, E_DIMS, EXCEL_PATH
from datamanager.utils import read_excel, enhance_intensity
from datamanager.transforms import obtain_transform

from datamanager.video import EBUSVideo


DELETE_GRAY = [(0.2075, 0.2765, 0.3079), (0.1851, 0.2356, 0.2353)]
KEEP_GRAY = [(0.2229, 0.2743, 0.3044), (0.1826, 0.2176, 0.2157)]


class MyDataset(Dataset):

    def __init__(self, device='cuda'):
        self.device = device
        self.data = None
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def split(self, parti_indices):
        return_dataset = []

        if isinstance(parti_indices[0], str):
            for item in parti_indices:
                start, end = item.split('-')[:2]
                start, end = int(start), int(end)
                return_dataset.append(Subset(self, list(range(start, end))))

        elif isinstance(parti_indices[0], list):
            for item in parti_indices:
                return_dataset.append(Subset(self, item))

        elif isinstance(parti_indices[0], float):
            length = self.__len__()
            start = 0
            for item in parti_indices:
                sep = int(length * item)
                return_dataset.append(Subset(self, list(range(start, sep))))
                start = sep

        else:
            raise ValueError('Unknown type of %s for elements in parti_indices!' % type(parti_indices[0]))

        return return_dataset

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class MultiModalDataset(MyDataset):
    def __init__(self, mode='BEF', paired=True, device='cuda'):
        super().__init__(device=device)
        self._mode = mode if isinstance(mode, set) else set(mode)
        self._paired = paired
        self.raw_data = None
        self.iter_data = None
        self.labels = None

    def __repr__(self):
        count = dict.fromkeys(travel(self._mode), 0)
        for k, v in self.raw_data.items():
            for mode in travel(self._mode):
                if len(mode) == 1:
                    if mode[0] in v.modes:
                        count[mode] += 1
                else:
                    if reduce(lambda x, y: x & y, [m in v.modes for m in mode]):
                        count[mode] += 1

        output = 'There is %d mode in this dataset, specifically:\n' % len(count)
        for k, v in count.items():
            output += 'Mode %s, %d; ' % (k, v)
        return output

    @abstractmethod
    def set_iter_data(self):
        pass

    @abstractmethod
    def read_in_data(self, source_dir):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    def set_mode(self, mode=None):
        if mode is None:
            self.set_iter_data()
        else:
            mode = mode if isinstance(mode, set) else set(mode)
            self._mode = mode
            self.set_iter_data()

        return

    def load(self, save_path: str):
        data = joblib.load(save_path)
        self.raw_data = data

    def save(self, save_path):
        if os.path.isfile(save_path):
            string = input('This file existed! Is overwrite? y/n\n')
            if string in ['y', 'Y']:
                joblib.dump(self.raw_data, save_path)
            else:
                flag = input('rename or stop? r/s\n')
                if flag in ['r', 'R']:
                    new_name = input('input the new name, please!\n')
                    joblib.dump(self.raw_data, new_name)
                else:
                    print('stopped!')
                return
        else:
            joblib.dump(self.raw_data, save_path)

    def set_paired(self, new_paired: bool):
        if self._paired == new_paired:
            return
        else:
            self._paired = new_paired
            self.set_mode(self._mode)

    def __len__(self):
        return len(self.iter_data)
        # return max([len(item) for item in self.iter_data.values()])


class SegDataset(MultiModalDataset):

    def __init__(self, mode='BEF', paired=True, aug='degrees=90, translate=(0.2, 0.2), scale=(0.6, 1.2)'):
        super().__init__(mode, paired=paired)

        self.test_tf = tf.Compose([tf.Resize(224),
                                   tf.CenterCrop(224),
                                   tf.ToTensor()])
        augment = eval('[tf.RandomAffine(%s)]' % aug)
        augment.extend([tf.Resize(224),
                        tf.CenterCrop(224),
                        tf.ToTensor()])
        self.train_tf = tf.Compose(augment)
        self.target_intensity = 45

    def read_in_data(self, source_dir: str) -> list:
        """

        :rtype: list of patient data. Patient data is a dict whose key is the mode and value is a list of correspond
                file path.
        """
        data = []
        for patient in os.listdir(source_dir):
            patient_data = {}
            for mode in os.listdir(patient_dir := os.path.join(source_dir, patient)):
                assert mode in ['B', 'E', 'F'], print('The mode in directory %s must be one of the BEF' % patient_dir)
                mode_dir = os.path.join(patient_dir, mode)
                patient_data[mode] = [(os.path.join(mode_dir, file),
                                       os.path.join(mode_dir, file.replace('raw', 'annotation')))
                                      for file in filter(lambda x: 'raw' in x, os.listdir(mode_dir))]
                if length := len(patient_data[mode]) != 3:
                    print('%s shot number is %d, not equal to 3!' % (mode_dir, length))
                    del patient_data[mode]

            data.append(patient_data)

        self.raw_data = data

        return data

    def set_iter_data(self, mode = None):
        if mode is None:
            mode = self._mode

        iter_data = {k: [] for k in mode}

        if self._paired:
            for patient in self.raw_data:
                if mode.issubset(set(patient.keys())):
                    for key in mode:
                        iter_data[key].extend(patient[key])
        else:
            for patient in self.raw_data:
                for key in mode:
                    if key in patient.keys():
                        iter_data[key].extend(patient[key])

        self.iter_data = iter_data
        self._mode = mode
        return

    @staticmethod
    def generate_edge_shot(mask: np.ndarray):
        edge = cv.Canny(mask, 80, 150)
        blurred = cv.GaussianBlur(edge, (5, 5), 0)
        blurred = np.expand_dims(blurred, axis=2)
        return blurred

    def __getitem__(self, item):
        mode = self._mode
        data = dict.fromkeys(mode)
        shot = dict.fromkeys(mode)
        edge = dict.fromkeys(mode)

        if self._paired:
            patient = {key: self.iter_data[key][item] for key in mode}
        else:
            length = self.__len__()
            patient = {key: self.iter_data[key][int(item / length * len(self.iter_data[key]))] for key in mode}

        for key in mode:
            image, mask = patient[key]
            image, mask = cv.imread(image).mean(2, keepdims=True).astype(np.uint8), \
                          cv.imread(mask).mean(2, keepdims=True).astype(np.uint8)
            image = enhance_intensity(image, self.target_intensity)
            blurred = self.generate_edge_shot(mask)

            union = np.concatenate((image, mask, blurred), axis=2)
            if self.training:
                union = self.train_tf(union)
            else:
                union = self.test_tf(union)

            data[key] = union[0].unsqueeze(0)
            mask = union[1].unsqueeze(0)
            mask[mask < 0.5] = 0
            mask[mask > 0.5] = 1
            shot[key] = mask
            maximum, minimum = union[2].max(), union[2].min()
            if 1 >= maximum > 0 and minimum >= 0:
                edge[key] = union[2] / maximum
            else:
                edge[key] = union[2]
                print('The input should be in 0-1')
        return data, (shot, edge)


class ArtifactDataset(MyDataset):
    def __init__(self):
        super().__init__()
        excel_file = '../raw-data/20200822-wait_shot.xlsx'
        df = pd.read_excel(excel_file)
        self.map_dict = {}
        for k, v in zip(df['file name'], df['label']):
            self.map_dict[k] = v

        source_dir = '../data/video/mm-ete/20200810-wait_shot'
        self.files = os.listdir(source_dir)
        self.source_dir = source_dir

        self.transform = tf.Compose([tf.Resize(224),
                                     tf.CenterCrop(224),
                                     tf.ToTensor()])

    def __getitem__(self, item):
        file_name = self.files[item]
        image = cv.imread(os.path.join(self.source_dir, file_name))
        image = self.transform(image)
        label = torch.tensor(self.map_dict[file_name], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.files)


class MMVideoDataset(MultiModalDataset):
    def __init__(self, database_path: str, mode: str = 'E', normalize=False, datatype_arrange=None, device='cpu'):
        super().__init__(mode, paired=True, device=device)
        # save the number of frames of each modal]
        if datatype_arrange is None:
            datatype_arrange = {'B': 'raw-img', 'F': 'roi', 'E': 'elastic-hist', 'C': 'h-raw-volume'}

        self.database_path = database_path
        self.normalize = normalize
        self.datatype_arrange = datatype_arrange
        # reference_dict = {'E': {'img': 'elastic', 'img-roi': 'ROI', 'hist': 'elastic-hist', 'hist-roi': 'ROI-hist'},
        #                   'B': {'img': 'raw-img', 'img-roi': 'ROI'},
        #                   'F': {'img': 'raw-img', 'img-roi': 'ROI'}}
        frame_ceil = {'B': 30, 'F': 3, 'E': 50, 'C': 30}
        self.__FRAME_CEIL = frame_ceil

        self.transforms = obtain_transform(datatype_arrange)

    @staticmethod
    def read_video(args):
        v, lock, mode = args
        v.read_in_data(lock=lock, mode=mode)
        return

    def clean_self(self):
        videos = self.raw_data
        need_delete = []

        for k, v in videos.items():
            v.check()
            if len(v.modes) == 0:
                need_delete.append(k)

        for key in need_delete:
            videos.pop(key)

        self.raw_data = videos

    def read_in_data(self, source_dir: str, excel_path: str = EXCEL_PATH, from_scratch=True):
        if excel_path is None:
            total = read_excel()
        else:
            total = read_excel(excel_path)

        need_handle = []
        if from_scratch or (self.raw_data is None):
            print(len(total))
            for info in total:
                v = EBUSVideo(info, source_dir, self.database_path, self.__FRAME_CEIL)
                if v.flag:
                    need_handle.append((v, v.modes))

        else:
            self.set_iter_data('BEF')
            existed = {v.info['id'] for v in self.iter_data}
            total_dict = {v['id']: v for v in total}
            need_handle_id = set(total_dict.keys()) - existed
            raw_data = self.raw_data
            for id_ in need_handle_id:
                if raw_data.get(id_, -1) == -1:
                    v = EBUSVideo(total_dict[id_], source_dir, self.database_path, self.__FRAME_CEIL)
                    if v.flag:
                        need_handle.append((v, v.modes))
                else:
                    v = raw_data[id_]
                    need_handle_mode = set(v.check()) - set(v.modes)
                    need_handle.append((v, need_handle_mode))

        length = len(need_handle)

        lock = Manager().Lock()
        with tqdm(total=length) as t:
            with Pool(cpu_count()) as p:
                for item in p.imap_unordered(self.read_video, [(v, lock, mode) for v, mode in need_handle], chunksize=5):
                    t.update()

        # check
        videos = {v.info['id']: v for v, mode in need_handle}
        need_remove = []
        for key in videos.keys():
            if videos[key].check_database():
                continue
            else:
                need_remove.append(key)
        for k in need_remove:
            del videos[k]

        if self.raw_data is None:
            self.raw_data = videos
        else:
            self.raw_data.update(videos)

        return

    def set_iter_data(self, mode=None):
        if mode is None:
            mode = self._mode
        data = []
        labels = []
        for v in self.raw_data.values():
            flag = True
            for item in mode:
                flag &= (item in v.modes)
            if flag:
                data.append(v)
                labels.append(v.info['BM'])
        self.iter_data = data
        self.labels = labels
        return

    def __getitem__(self, item):
        mode = self._mode
        dta = self.datatype_arrange
        video = self.iter_data[item]
        device = self.device

        identity = torch.tensor(video.info['id'], dtype=torch.int)
        label = torch.tensor(video.info['BM'], dtype=torch.long)

        item_data = {}
        for m in mode:
            datatype = dta[m]
            if m == 'E' and '+' in datatype:
                img, hist = datatype.split('+')
                data = (video.get_data(m, img, self.database_path), video.get_data(m, hist, self.database_path))
                length = data[0].shape[0]
            else:
                data = video.get_data(m, datatype, self.database_path)
                length = data.shape[0]

            data = self.transforms[m](data, self.training)
            mask = torch.tensor((length, ), dtype=torch.int8, device=device)

            item_data.update({m: (data, mask)})

        return identity, item_data, label
