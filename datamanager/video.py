import base64
import datetime
import json
import os
import shutil
import random
import operator

import cv2 as cv
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from scipy import stats
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as tf
from torchvision.io import read_image
from tqdm import tqdm

from metainfo.default import EXCEL_PATH, tidy_folder, MODE_SIGN, BASIC_FOLDER, IMAGE_FORMAT, CROP_RANGE
from datamanager.utils import read_excel, shape_to_mask, LowQualityError, mini_cut, gaussian_laplacian, read_folder, \
    read_hist, estimate_frame, TooLongVideoError
from model.network.backbone.msnet import ms_u_net
from model.utils import post_process


class FolderDataset(Dataset):
    def __init__(self, source_dir: Path):
        super().__init__()
        self.files = list(source_dir.iterdir())
        self.transform = tf.Compose([tf.Grayscale(num_output_channels=1),
                                     tf.Resize(224),
                                     tf.CenterCrop(224)])

    def __getitem__(self, item):
        img_path = self.files[item]
        img = cv.imread(str(img_path))
        img = torch.from_numpy(img)
        if len(img.size()) == 3:
            img = img.permute(2, 0, 1)
        img = self.transform(img)
        img = img.float() / 255
        return img, img_path.name

    def __len__(self):
        return len(self.files)


# delete confirm wrapper
def delete_confirm(func):
    def wrapper(manager, target_dir, **kwargs):
        need_delete = func(manager, target_dir, **kwargs)

        is_delete = False
        string = input('there is %d need to be deleted, is continue or check detail? y/n/c\n' % len(need_delete))
        if string in ['y', 'Y']:
            is_delete = True

        elif string in ['c', 'C']:
            print('Folders need to be deleted are: ', need_delete, '\n')
            new_string = input('Delete? y/n\n')
            if new_string in ['y', 'Y']:
                is_delete = True

        if is_delete:
            for directory in need_delete:
                shutil.rmtree(directory)

            print('Remove %d folders totally!' % len(need_delete))

        else:
            print('Stop removing')

        return

    return wrapper


# data generate wrapper
def generate_template(func):

    def wrapper(manager, target_dir, from_scratch=True, batch_size: int = 32, device: str = 'cuda'):
        print('Current process %s mode!' % manager.mode)
        excel_file = manager.excel_file
        source_dir = manager.tidy_dir
        print('current used excel: %s' % excel_file)

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        samples = read_excel(excel_file)
        if from_scratch:
            need_handle = samples
        else:
            need_handle = manager.lacked_samples(target_dir, samples)

        print('%d need to handle' % len(need_handle))

        except_clt = {}
        success = 0

        bar = tqdm(need_handle)

        for i, s in enumerate(bar):

            source_dir = Path(source_dir)
            directory = source_dir / (s['date'] + s['name'] + s['local'])
            if not directory.is_dir():
                error = 'Directory %s Not Found' % directory.name
                if 'no_folder' in except_clt.keys():
                    except_clt['no_folder'].append(error)
                else:
                    except_clt['no_folder'] = [error]
                continue

            is_success = func(manager, exception=except_clt, info=s, source_dir=directory, target_dir=target_dir,
                              batch_size=batch_size, device=device)
            if is_success:
                success += 1

            bar.set_description('Current processing %s, Success rate: %f' % (s['date'] + s['name'], success / (i + 1)))

        with open('../record/exception.txt', 'a') as f:
            curr_time = datetime.datetime.now()
            time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S\n')
            f.write('\n' + time_str + ' Process mode: %s' % manager.mode)
            for k, v in except_clt.items():
                f.write('%s ' % k + ','.join(v) + '\n')

        print('Process %d videos successfully.\n' % success +
              '\n'.join(['%s : %d' % (k, len(v)) for k, v in except_clt.items()]))

        return

    return wrapper


class DataManager(object):

    def __init__(self, tidy_dir=None, excel_file=None):
        self._wait_seg_folder = 'raw-img'
        self._raw_img_folder = 'raw-img'
        self._seg_folder = 'ROI'
        self._shot_folder = None
        self.mode = None

        if tidy_dir is None:
            self.tidy_dir = tidy_folder
        else:
            self.tidy_dir = tidy_dir

        if excel_file is None:
            self.excel_file = EXCEL_PATH
        else:
            self.excel_file = excel_file

    @abstractmethod
    def read_video(self, video_path, hospital: str = 'shanghai', batch_size: int = 32, device: str = 'cuda'):
        pass

    @abstractmethod
    def quality_detect(self, images: torch.Tensor, *args, **kwargs):
        pass

    @generate_template
    def generate_v_folder(self, exception: dict, info: dict, source_dir: str, target_dir: str, batch_size: int = 32,
                          device: str = 'cuda'):
        """
        This function used to capture useful frames from raw video, and it is a template where the common operations
        among different modes are presented. The main procedures of handling a video are:
            1. Capture frames in batch from video with certain hyper-parameter by calling the capture_frame function
            2. Read the whole video, and return available frames by calling read_video func
            3. Process and write qualified frames into files by calling write_image func.

        :rtype: None
        """
        patient = os.path.basename(source_dir)

        target_dir = Path(target_dir)
        save_dir = target_dir / self.naming_rule(info) / self.mode

        if save_dir.is_dir():
            shutil.rmtree(str(save_dir))
            save_dir.mkdir()

        try:
            video_path = self.get_video_path(source_dir)
        except FileNotFoundError:
            if 'no_video' in exception.keys():
                exception['no_video'].append(patient)
            else:
                exception['no_video'] = [patient]
            return False

        hospital = info.get('hospital', 'shanghai')
        try:
            available = self.read_video(video_path, hospital, batch_size, device=device)

        except LowQualityError:
            print('%s patient low_quality' % patient)
            if 'low_quality' in exception.keys():
                exception['low_quality'].append(patient)
            else:
                exception['low_quality'] = [patient]
            return False

        except cv.error as e:
            print(e)
            print('%s patient cv error' % patient)

            if 'video_error' in exception.keys():
                exception['video_error'].append(patient)
            else:
                exception['video_error'] = [patient]
            return False

        except TooLongVideoError:
            print('%s patient too long video' % patient)

            if 'too_long_video' in exception.keys():
                exception['too_long_video'].append(patient)
            else:
                exception['too_long_video'] = [patient]
            return False

        self.write_image(save_dir, self._wait_seg_folder, available, info)

        return True

    @generate_template
    def generate_roi_folder(self, info: dict, exception: dict, source_dir: str, target_dir: str):
        patient = os.path.basename(source_dir)

        if os.path.isdir(shot_dir := os.path.join(source_dir, self._shot_folder)):
            if len(os.listdir(shot_dir)) == 0:
                return False
            if os.path.isdir(patient_dir := os.path.join(target_dir, self.naming_rule(info), self.mode)):
                if 'existed' in exception.keys():
                    exception['existed'].append(patient)
                else:
                    exception['existed'] = [patient]
                return False
            else:
                os.makedirs(patient_dir)

                for i, f in enumerate(filter(lambda x: x.endswith('.json'),
                                             os.listdir(shot_dir))):
                    with open(os.path.join(shot_dir, f), 'rb') as js_file:
                        try:
                            data = js_file.read().decode('utf-8', errors='ignore')
                            diction = json.loads(data)
                        except UnicodeDecodeError as e:
                            print(e)
                            continue

                    img_data = diction['imageData']
                    bytes_stream = BytesIO(base64.b64decode(img_data))
                    raw_image = Image.open(bytes_stream)
                    raw_image = np.array(raw_image)

                    shape = raw_image.shape
                    points = diction['shapes'][0]['points']
                    mask = shape_to_mask(shape, points).astype(np.uint8) * 255

                    image = self.pre_cut(raw_image)
                    mask = self.pre_cut(mask)

                    if self.mode == 'B':
                        cv.imwrite(os.path.join(patient_dir, '%d-raw.png' % i), image)
                        cv.imwrite(os.path.join(patient_dir, '%d-annotation.png' % i), mask)

                    elif self.mode == 'E':
                        image_tensor = F.to_tensor(image)
                        mask_tensor = F.to_tensor(mask)
                        elastic = image_tensor[:, :, -self._shift:]
                        gray = image_tensor[:, :, :(-self._shift)]
                        mask = mask_tensor[:, :, :(-self._shift)]
                        mini_range = mini_cut(elastic)
                        image = self.basic_crop(gray, *mini_range[0])
                        mask = self.basic_crop(mask, *mini_range[0])

                        F.to_pil_image(image).save(os.path.join(patient_dir, '%d-raw.png' % i))
                        F.to_pil_image(mask).save(os.path.join(patient_dir, '%d-annotation.png' % i))

                    else:
                        image_tensor = F.to_tensor(image)
                        mask_tensor = F.to_tensor(mask)
                        mini_range = mini_cut(image_tensor)
                        image = self.basic_crop(image_tensor, *mini_range[0])
                        mask = self.basic_crop(mask_tensor, *mini_range[0])

                        F.to_pil_image(image).save(os.path.join(patient_dir, '%d-raw.png' % i))
                        F.to_pil_image(mask).save(os.path.join(patient_dir, '%d-annotation.png' % i))

                return True

        else:
            if 'no_shot' in exception.keys():
                exception['no_shot'].append(patient)
            else:
                exception['no_shot'] = [patient]
            return False

    def generate_seg(self, source_dir: str, max_batch: int = 50,
                     model_path: str = '../models/segmentation/20201010-MS.pth',
                     from_scratch=True):

        #  load the segmentation model
        model = nn.DataParallel(ms_u_net(modes='BEF', init_ch=64, stage=5).cuda())
        params = torch.load(model_path)['param']
        model.module.load_state_dict(params)
        model.eval()
        print('load segmentation model successfully!')

        #  prepare the transform function
        # to_pil_img = tf.ToPILImage()
        transform = tf.Compose([tf.Resize(224),
                                tf.CenterCrop(224)])

        source_dir = Path(source_dir)
        if source_dir.is_dir():
            bar = tqdm(source_dir.iterdir())
            for lymph_node in bar:
                bar.set_description('Current processing %s' % lymph_node.name)
                #  check the pre-segmentation directory and the save directory
                wait_seg_dir = lymph_node / self.mode / self._wait_seg_folder
                if not wait_seg_dir.is_dir():
                    print('wait seg dir for LN %s unavailable' % lymph_node.name)
                    continue
                seg_dir = lymph_node / self.mode / self._seg_folder
                if seg_dir.is_dir():
                    if from_scratch:
                        shutil.rmtree(str(seg_dir))
                    else:
                        print('segmentation folder for lymph node %s exist!' % lymph_node.name)
                        continue

                seg_dir.mkdir(parents=True)

                dataloader = DataLoader(FolderDataset(wait_seg_dir), num_workers=2, batch_size=max_batch)
                for data, name in dataloader:
                    batch_size = data.size(0)
                    data = data.cuda()
                    out = model({self.mode: data})
                    out = out[self.mode] > 0.5 + 0
                    out = out.byte().cpu().permute(0, 2, 3, 1).numpy()

                    for i in range(batch_size):
                        mask = post_process(out[i])
                        mask = torch.from_numpy(mask)
                        if self.mode == 'B':
                            img = data[i].cpu() * 255
                        else:
                            source_file = lymph_node / self.mode / self._raw_img_folder / name[i]
                            img = read_image(str(source_file))
                            img = transform(img)
                        seg = (img * mask).byte()
                        if self.mode != 'B':
                            seg = seg[[2, 1, 0]]
                        cv.imwrite(str(seg_dir / name[i]), seg.permute(1, 2, 0).numpy())

        return

    def capture_frame(self, video_path, hospital='shanghai', ignore=0, gap=0, max_length=1200):
        """
        This function used to capture frame from the video with given hyper-parameter and return whole frames of current
        video.

        :param max_length:
        :param hospital: The hospital this video belong to
        :param video_path: str, The file path of video.
        :param ignore: int, The first "ignore" frames to discard
        :param gap: yield one frame every gap frames

        :rtype: Tensor, the frames in batch.
        """

        if (video_length := estimate_frame(video_path)) > max_length:
            print('%s video is too long!' % video_path)
            raise TooLongVideoError('The video %s is too long (%d frames)' % (video_path, video_length))

        vid_cap = cv.VideoCapture(video_path)

        interval = 0
        total = 0

        buffer = []
        success = True
        while success:
            success, image = vid_cap.read()
            if success:

                if ignore > 0:
                    ignore -= 1
                    continue
                if interval > 0:
                    interval -= 1
                else:
                    interval = gap

                    total += 1
                    if self.mode == 'E':
                        gray, elastic = self.pre_cut(image, hospital)
                        if gaussian_laplacian(elastic):
                            gray, elastic = torch.from_numpy(gray), torch.from_numpy(elastic)

                            c, w, h = elastic.size()
                            gray, elastic = gray.view(1, 1, c, w, h), elastic.view(1, 1, c, w, h)
                            image = torch.cat((elastic, gray), dim=1)
                            buffer.append(image)

                    else:
                        image = self.pre_cut(image, hospital)
                        if self.mode == 'B':
                            the = 20
                        else:
                            the = 100
                        if gaussian_laplacian(image, the):
                            image = torch.from_numpy(image).unsqueeze(0)
                            buffer.append(image)

        vid_cap.release()
        video_length = len(buffer)
        if total == 0:
            raise LowQualityError

        print('gaussian_laplacian keep %f' % (video_length / total))

        if video_length > 0:
            data = torch.cat(buffer, dim=0)
            if self.mode == 'E':
                data = data.permute(0, 1, 4, 2, 3)
                data = data[:, :, [2, 1, 0]]
            elif self.mode == 'B':
                data = data.permute(0, 3, 1, 2)
            else:
                data = data.permute(0, 3, 1, 2)[:, [2, 1, 0]]

            return data
        else:
            raise LowQualityError

    def write_image(self, save_dir: Path, sub_dir: str, images: list, info_dict: dict):

        save_dir = save_dir / sub_dir
        if save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)

        if isinstance(images, (list, tuple)):
            length = len(images)
        elif isinstance(images, torch.Tensor):
            length = images.size(0)
        else:
            raise TypeError('Unknown images type: %s' % type(images))

        for i in range(length):
            filename = str(i) + '.png'
            img = images[i]
            if img.device == 'cuda':
                img = img.cpu()
            img = F.to_pil_image(img.byte())
            img.save(str(save_dir / filename))

        return

    @staticmethod
    def obtain_data_loader(data: torch.Tensor, batch_size=32):
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        return dataloader

    @staticmethod
    def drop_deformity(images: list, channel: int = None):
        length = len(images)
        shapes = np.zeros((length, 3), dtype=np.int)
        for i in range(length):
            shapes[i] = images[i].shape

        mode, count = stats.mode(shapes)

        over_size = np.absolute(shapes - mode).sum(1)
        normal = over_size < 60

        normal_images = [images[i] for i in range(length) if normal[i]]
        standard_shape = mode.squeeze().tolist()
        if channel == 1:
            standard_shape[3] = 1

        reformed_images = []
        for i, img in enumerate(normal_images):
            if channel == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if operator.eq(img.shape, standard_shape):
                reformed_images.append(img)
            else:
                resized = cv.resize(img, (standard_shape[2], standard_shape[1]))
                if len(resized.shape) == 2:
                    resized = np.expand_dims(resized, axis=2)
                reformed_images.append(resized)

        return reformed_images

    @staticmethod
    def naming_rule(ln_info: dict):
        return '%d-%d-%s' % (ln_info['id'], ln_info['BM'], ln_info['name'])

    @staticmethod
    def basic_crop(image: torch.Tensor, top: int, left: int, height: int, width: int):
        return F.crop(image, top, left, height, width)

    @delete_confirm
    def remove(self, target_dir: str):
        target_dir = Path(target_dir)
        if target_dir.is_dir():
            need_delete = []
            for patient_dir in target_dir.iterdir():
                if mode_dir := (patient_dir / self.mode).is_dir():
                    need_delete.append(str(mode_dir))

        return

    @delete_confirm
    def remove_redundancy(self, target_dir: str, total_samples=None):
        if total_samples is None:
            total_samples = read_excel(self.excel_file)
        required_folders = [self.naming_rule(item) for item in total_samples]

        need_delete = []

        target_dir = Path(target_dir)
        for patient_dir in target_dir.iterdir():
            folder = patient_dir.name
            if folder not in required_folders:
                need_delete.append(folder)
                continue

            if (patient_dir / self.mode).is_dir():
                modal_dir = patient_dir / self.mode / self._wait_seg_folder
                if self.remove_condition(modal_dir):
                    need_delete.append(str(modal_dir))

        return need_delete

    def remove_condition(self, directory: Path):
        if len(list(directory.iterdir())) == 0:
            return True
        else:
            return False

    def lacked_samples(self, target_dir: str, total_samples=None):

        if total_samples is None:
            total_samples = read_excel(self.excel_file)

        need_handle = []
        existed = []
        target_dir = Path(target_dir)
        for patient_dir in target_dir.iterdir():
            if patient_dir.is_dir():
                if (patient_dir / self.mode / self._wait_seg_folder).is_dir():
                    existed.append(patient_dir.name)

        for s in total_samples:
            identity = self.naming_rule(s)
            if identity in existed:
                corr_dir = target_dir / identity
                if not (corr_dir / self.mode / self._wait_seg_folder).is_dir():
                    need_handle.append(s)
            else:
                need_handle.append(s)

        return need_handle

    def pre_cut(self, image: np.ndarray, hospital: str = 'shanghai'):
        shape = image.shape
        mode = self.mode
        try:
            crop_range = CROP_RANGE[hospital][str(shape[1])][mode]
        except KeyError as e:
            print('Unknown combination of keys: %s-%s-%s' % (hospital, shape[1], mode))
            raise KeyError

        if mode == 'E':
            left, right, top, bottom, shift = crop_range
            return image[top: bottom, (left - shift): (right - shift)], image[top: bottom, left: right]
        else:
            left, right, top, bottom = crop_range
            return image[top: bottom, left: right]

    def get_video_path(self, directory: str):
        folder = Path(directory)

        video_path = []
        for item in folder.iterdir():
            if item.is_file():
                name = item.stem
                suffix = item.suffix
                if (self.mode in name) and (suffix in ['.mp4', '.MP4', '.avi']):
                    video_path.append(item.name)
        if len(video_path) == 0:
            raise FileNotFoundError('No video found in %s' % directory)

        video_path.sort()
        video_path = [str(folder / item) for item in video_path]
        return tuple(video_path)


class EBUSVideo(object):

    def __init__(self, info: dict, repository: str, database_path: str, frame_ceil: dict):
        # the directory for EBUSVideo of current lymph node
        repository = Path(repository)
        directory = repository / DataManager.naming_rule(info)

        self.flag = True
        self.directory = directory
        modes = self.check()
        if len(modes) == 0:
            self.flag = False
            return

        self.frame_ceil = frame_ceil
        self.info = info
        self.modes = modes
        self.length = {}
        self.database_path = database_path

    def check(self):
        directory = self.directory
        modes = []
        # check whether the given directory legal
        if not directory.is_dir():
            return modes
        # check whether the content of given directory legal

        for subdir in directory.iterdir():
            mode_names = tuple(MODE_SIGN.values())
            if subdir.name in mode_names and subdir.is_dir():
                basic_dir = subdir / BASIC_FOLDER[subdir.name]
                if basic_dir.is_dir() and len(list(basic_dir.rglob('*%s' % IMAGE_FORMAT))) > 0:
                    modes.append(subdir.name)
        return modes

    def read_in_data(self, lock=None, mode=None):
        """
        This function read images in given data folder and save them in h5py file.
        The structure of h5 file as follow:

        h5py -> patient (key: the id of patient; type: group) -> modes (key: the mode name, i.e., B,E,F; type: group)
             -> datatype (key: the type of data, i.e., raw, mask. elastic, gray, elastic_hist, elastic_roi_hist for E
                mode; type: dataset; format: B * C * H * W). The data saved in order RGB.

        :rtype: object
        """

        modes = self.modes if mode is None else mode
        need_write = []
        frame_ceil = self.frame_ceil

        for m in modes:

            if m == 'C':
                need_write.extend(self.read_ct_data())

            else:
                if m == 'E':
                    datatype = ['elastic', 'gray', 'ROI']
                else:
                    datatype = ['raw-img', 'ROI']

                if m == 'B':
                    clean_data = True
                else:
                    clean_data = False

                mode_dir = self.directory / m

                length = len(list((mode_dir / datatype[0]).iterdir()))
                indices = list(range(length))

                ceil = frame_ceil[m]
                if length > ceil:
                    indices = random.sample(indices, ceil)

                for dty in datatype:

                    data_dir = mode_dir / dty

                    if not data_dir.is_dir():
                        break

                    # if not len(list(data_dir.iterdir())) == length:
                    #     print('%s have different number of images in different datatype!!' % str(mode_dir))
                    #     break

                    data_key = '%d/%s/%s' % (self.info['id'], m, dty)

                    try:
                        if dty == 'gray' or m == 'B':
                            data = read_folder(data_dir, channel=1, clean_data=clean_data)
                        else:
                            data = read_folder(data_dir, clean_data=clean_data)
                    except AssertionError as e:
                        print(e)
                        continue

                    if data.shape[0] > length:
                        data = data[indices]

                    need_write.append((data_key, data))

                    if m == 'E' and dty != 'gray':
                        hist_key = '%d/%s/%s-hist' % (self.info['id'], m, dty)
                        hist_data = read_hist(data_dir)
                        need_write.append((hist_key, hist_data[indices]))

        lock.acquire()
        try:
            with h5py.File(self.database_path, 'a') as handler:
                for key, data in need_write:
                    value = handler.get(key, -1)
                    if value != -1:
                        del handler[key]
                    handler.create_dataset(name=key, data=data)
        finally:
            lock.release()

        return

    def get_data(self, mode: str, datatype: str, database_path: str = None) -> np.ndarray:

        database_path = self.database_path if database_path is None else database_path

        with h5py.File(database_path, 'r', swmr=True) as handler:
            try:
                dataset = handler['%d/%s/%s' % (self.info['id'], mode, datatype)]
            except KeyError as e:
                print(e)
                print(self.info['id'])
                raise KeyError
            data = dataset[()]

        return data

    def check_database(self):
        with h5py.File(self.database_path, 'r', swmr=True) as handler:
            for m in self.modes:
                key = '%d/%s' % (self.info['id'], m)
                group = handler.get(key, default=-1)
                if group == -1:
                    return False
        return True

    def get_length(self, mode: str, datatype: str) -> int:

        return self.length['%s-%s' % (mode, datatype)]

    def read_ct_data(self):

        # datatype = ['h-raw-volume', 'h-raw-max', 'h-roi-volume', 'h-roi-max',
        #             'e-raw-volume', 'e-raw-max', 'e-roi-volume', 'e-roi-max']

        mode_dir = self.directory / 'C'
        need_write = []

        for data_type in ['h', 'e']:
            ceil = 30 if data_type == 'h' else 15

            mode_raw_dir = mode_dir / ('%s-raw-volume' % data_type)
            if not mode_raw_dir.is_dir():
                continue

            length = len(list(mode_raw_dir.iterdir()))
            indices = list(range(length))
            if length > ceil:
                indices = random.sample(indices, ceil)
            else:
                indices = list(range(length))

            for data_amount in ['volume', 'max']:

                for data_size in ['raw', 'roi']:
                    dty = '%s-%s-%s' % (data_type, data_size, data_amount)
                    data_dir = mode_dir / dty

                    if not data_dir.is_dir():
                        break

                    data_key = '%d/%s/%s' % (self.info['id'], 'C', dty)

                    try:
                        data = read_folder(data_dir, channel=1, clean_data=True)
                    except AssertionError as e:
                        print(e)
                        continue

                    if data_amount == 'volume':
                        data = data[indices]

                    need_write.append((data_key, data))
        return need_write
