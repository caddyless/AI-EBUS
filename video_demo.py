import os
import time
import cv2
import torch
import joblib
import argparse

import numpy as np
import torchvision.transforms as tf
import torchvision.transforms.functional as F

from torch import nn
from pathlib import Path
from datamanager.transforms import Normalize
from datamanager.utils import gaussian_laplacian, convert_image
from model.utils import post_process
from model.network.backbone.seincept import seincept
from model.network.backbone.msnet import ms_u_net


CROP_RANGE = {
    'shanghai': {'1920': {'B': (1238, 1807, 300, 769), 'E': (1466, 1807, 300, 690, 368), 'F': (1238, 1807, 300, 769)},
                 '1024': {'B': (79, 880, 143, 616), 'E': (88, 520, 56, 512)}},

    'henan': {'1920': {'B': (970, 1540, 136, 560), 'E': (960, 1540, 137, 799, 624), 'F': (577, 1540, 136, 976)}},

    'anhui': {'1280': {'B': (368, 1220, 77, 908), 'E': (639, 1220, 86, 738, 624), 'F': (368, 1220, 86, 908)}},

    'hangzhou': {'1920': {'B': (575, 1540, 155, 970), 'E': (960, 1540, 144, 798, 624), 'F': (576, 1539, 136, 970)}},

    'yantai': {'1280': {'B': (256, 1220, 77, 910), 'E': (640, 1220, 77, 737, 623), 'F': (256, 1220, 77, 910)}},
    }


class DataInstance(object):
    def __init__(self, feature: torch.Tensor, image: np.ndarray):
        self.feature = feature
        self.image = image


class PriorityQueue(object):
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data = []
        self._index = 0

    def push(self, priority, data: DataInstance):
        self._data.append((priority, self._index, data))
        self._data.sort(key=lambda x: x[0])
        self._index += 1
        length = len(self._data)
        if length > self.maxsize:
            self.pop()
        return

    def pop(self):
        return self._data.pop(0)

    def reassign_priority(self, priority: list):
        length = len(self._data)
        assert len(priority) == length, print('The length of reassigned must equal to original')
        data = [(priority[i], self._data[i][1], self._data[i][2]) for i in range(length)]
        data.sort(key=lambda x: x[0])
        self._data = data
        return

    def obtain_feature(self):
        return [item[-1].feature for item in self._data]

    def obtain_top_frame(self, k: int = 1):
        return [self._data[-i][-1].image for i in range(1, k + 1)]

    def __len__(self):
        return len(self._data)


def to_tensor(image: np.array):
    """
    This function convert np.ndarray image to tensor.
    The input image must be W * H * C, the returned tensor is C * W * H
    The returned tensor switch the order of channels of input image (the channel of input image regarded as BGR in
    default, and the returned tensor will be RGB instead).
    Both the input image and returned tensor are of uint8 datatype.
    :rtype: torch.Tensor
    """
    img = torch.from_numpy(image)
    if len(img.size()) == 2:
        img = img.unsqueeze(0)
    else:
        img = img.permute(2, 0, 1)
        if img.size(0) == 3:
            img = img[[2, 1, 0]]
    return img


def to_numpy(data: torch.Tensor):
    """
    This function convert tensor to np.ndarray image.
    The input tensor must be C * W * H, the returned tensor is W * H * C instead.
    The returned tensor switch the order of channels of input image (the channel of input image regarded as RGB in
    default, and the returned image will be BGR instead).
    Both the input image and returned tensor are of uint8 datatype.
    :rtype: object
    """
    c, w, h = data.size()
    if data.max() <= 1:
        data = (data * 255).to(torch.uint8)
    if c == 3:
        data = data[[2, 1, 0], :, :]
    img = data.permute(1, 2, 0).cpu().numpy()
    return img


def is_overlap(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]
    m = y.shape[0]

    distance = np.abs((x.reshape((n, 1, 2)) - y.reshape((1, m, 2)))).sum(2)

    return (distance < 10).any()


def get_bounds(image: np.ndarray):
    """
    This function returns the bound of input batch of images

    :param image: The input image
    :rtype: torch.Tensor
    """
    img = image.astype(np.int)
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    mask = ((g - r) > 50) * ((g - b) > 50) * (g > 90)
    w, h = mask.shape
    lower_bound = w

    mask = np.transpose(np.nonzero(mask))
    if mask.shape[0] == 0:
        return 0, 0, 0, 0
    top = mask[:, 0].min().item()
    bottom = mask[:, 0].max().item()
    left = mask[:, 1].min().item()
    right = mask[:, 1].max().item()
    bottom_indices = mask[mask[:, 0] == bottom]
    left_indices = mask[mask[:, 1] == left]
    right_indices = mask[mask[:, 1] == right]

    if is_overlap(bottom_indices, left_indices) or is_overlap(bottom_indices, right_indices):
        bottom = lower_bound

    width = right - left
    height = bottom - top
    if width < 50 or height < 50:
        raise ValueError('No scanner!')

    return left, right, top, bottom


def color_detect(image: np.ndarray, satu_tsd=35, color_tsd=0.7, intensity_tsd=80, scaled=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float)
    hsv_s = hsv[:, :, 1]
    hsv_v = hsv[:, :, 2]

    colored_crd = hsv_s > satu_tsd  # W * H
    gray_crd = (hsv_s > 0) & (~colored_crd)  # W * H

    num_colored_pixel = colored_crd.sum()
    total_in_scan = (hsv_v > 0).sum()
    ratio = num_colored_pixel / total_in_scan

    mean = hsv_v.mean()
    if scaled:
        refer_mean = hsv_v[gray_crd].mean()
        mean = mean * (90 / refer_mean)
    print(ratio, mean)
    return (ratio > color_tsd) & (mean > intensity_tsd)


def pre_cut(image: np.ndarray, crop_range: tuple, mode: str):
    if mode == 'B':
        left, right, top, bottom = crop_range
        cropped = image[top: bottom, left: right].copy()
        relative_pos = crop_range

    elif mode == 'E':
        left, right, top, bottom, shift = crop_range
        elastic = image[top: bottom, left: right].copy()
        gray = image[top: bottom, (left - shift): (right - shift)].copy()
        bounds = get_bounds(elastic)
        elastic = elastic[bounds[2]: bounds[3], bounds[0]: bounds[1]]
        gray = gray[bounds[2]: bounds[3], bounds[0]: bounds[1]]
        cropped = (elastic, gray)
        relative_pos = [left + bounds[0], left + bounds[1], top + bounds[2], top + bounds[3]]

    elif mode == 'F':
        left, right, top, bottom = crop_range
        cropped = image[top: bottom, left: right].copy()
        bounds = get_bounds(cropped)
        cropped = cropped[bounds[2]: bounds[3], bounds[0]: bounds[1]]
        relative_pos = [left + bounds[0], left + bounds[1], top + bounds[2], top + bounds[3]]

    else:
        raise ValueError('Unknown mode %s' % mode)

    return cropped, relative_pos


def augment(frame, mask, crop_range, color='r', edge=False):
    if edge:
        mask = mask * 255
        mask = cv2.Canny(mask, 100, 200)
        mask[mask > 0] = 1
        mask = mask.astype(bool)
    w, h, c = frame.shape
    left, right, top, bottom = crop_range[:4]
    full_mask = np.zeros((w, h), dtype=bool)
    full_mask[top: bottom, left: right] = mask
    value = np.array([0, 0, 255], dtype=np.uint8)
    frame[full_mask] = value
    return frame


def segment(seg_model, frame: np.ndarray, mode: str):
    frame = to_tensor(frame)
    frame = frame.cuda().unsqueeze(0).float() / 255
    size = frame.size()[-2:]
    cropped = F.resize(frame, (224, 224))
    if cropped.size(1) != 1:
        cropped = cropped.mean(dim=1, keepdim=True)

    out = seg_model({mode: cropped})
    out = out[mode] > 0.5 + 0
    out = F.resize(out, size)
    out = out.byte().cpu().permute(0, 2, 3, 1).numpy()
    mask = post_process(out[0]).astype(np.uint8)
    return mask


def present_scanner(frame: np.ndarray, relative_pos: tuple):
    scanner = np.zeros((relative_pos[3] - relative_pos[2], relative_pos[1] - relative_pos[0]), np.bool)
    scanner[0, :] = True
    scanner[-1, :] = True
    scanner[:, 0] = True
    scanner[:, -1] = True
    frame = augment(frame, scanner, relative_pos)
    return frame


def collect_video_path(video_dir: str, mode: str = 'BEF'):
    #  Find video paths for each mode
    video_dir = Path(video_dir)
    video_path = {}
    for m in mode:
        if m == 'E':
            video = video_dir / 'E1.MP4'
        else:
            video = video_dir / ('%s.MP4' % m)
        if video.is_file():
            video_path[m] = str(video)

    if len(video_path) != len(mode):
        raise FileNotFoundError('Lack of files! Only exist %s' % str(video_path.keys()))

    return video_path


def load_seg_model(model_path: str):
    # load the segmentation model
    seg_model_path = model_path
    model = nn.DataParallel(ms_u_net(modes='BEF', init_ch=64, stage=5).cuda())
    params = torch.load(seg_model_path)['param']
    model.module.load_state_dict(params)
    model.eval()
    return model


def load_artefact_model(save_path: str = '../models/recognize_artefact/2020-09-19 16:55:18-0.9739.pth'):
    model = seincept(in_channel=3).cuda()
    state = torch.load(save_path)
    parameters = {}
    for k, v in state['param'].items():
        parameters[k.replace('module.', '')] = v
    model.load_state_dict(parameters)
    model.eval()
    threshold = state['threshold']
    return model, threshold


def count_pixels(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    num_pixels = (s > 30).sum()
    return num_pixels


def load_clf_model(model_folder: str):
    # load the diagnostic model
    model_folder = Path(model_folder)
    diag_model_path = str(model_folder / 'best-epoch.pth')
    state = torch.load(diag_model_path)
    clf_model = joblib.load(str(model_folder / 'initialized.model'))
    clf_model.model.module.load_state_dict(state['param'])
    # clf_model = obtain_net(**convert_dict_yaml(yaml_file=str(model_folder / 'net_config.yaml')))
    clf_model = clf_model.model
    clf_model = clf_model.module if isinstance(clf_model, nn.DataParallel) else clf_model
    clf_model.eval()
    return clf_model


def embed_represent_image(frame: np.ndarray, image: np.ndarray, position: tuple):
    width, height = image.shape[:2]
    target_height = 245
    scale = target_height / height
    resized_width = int(width * scale)
    embed_image = cv2.resize(image, (target_height, resized_width))
    frame[position[1]: (position[1] + resized_width), position[0]: (position[0] + target_height)] = embed_image
    return frame


class VideoPlayer(object):
    def __init__(self, video_dir: str, hospital: str, datatype_arrange: dict, clf_model_folder: str, mode: str = 'BEF',
                 seg_model_path: str = '../models/segmentation/20201010-MS.pth'):
        self.hospital = hospital
        self.datatype_arrange = datatype_arrange
        self.num_threshold = {'B': (20, 40), 'F': (3, 5), 'E': (5, 51)}
        self.data_transform = torch.nn.Sequential(
            tf.Resize(224),
            tf.CenterCrop(224),
            Normalize()
        )
        self.aggregated_feature = {}

        self.video_path = collect_video_path(video_dir, mode)
        print('Video path loaded')

        self.seg_model = load_seg_model(seg_model_path)
        print('load segmentation model successfully!')

        self.artefact_model = load_artefact_model()
        print('load artefact detect model successfully!')

        self.clf_model = load_clf_model(clf_model_folder)
        print('load clf model successfully!')

    def data_format(self, data: np.ndarray, mode: str):
        datatype = self.datatype_arrange[mode]
        if mode == 'E' and 'hist' in datatype:
            data = convert_image(data)
            data = torch.tensor(data, dtype=torch.float)
        else:
            data = to_tensor(data)
            data = self.data_transform(data).float() / 255
        return data.cuda().unsqueeze(0)

    def artefact_detect(self, frame: np.ndarray):
        data = self.data_format(frame, 'F')
        model, threshold = self.artefact_model
        predict = model(data)
        predict = nn.functional.softmax(predict, dim=1)[:, 1]
        return predict.item() <= threshold

    def multimodal_decision(self, feature: dict):
        fusion_feature = self.clf_model.fusion_clf.parse_feature(feature)
        score = self.clf_model.fusion_clf.clf(fusion_feature)
        score = nn.functional.softmax(score, dim=1)[:, 1]
        return score.item()

    def diagnosis(self):
        for k, v in self.video_path.items():
            self.play(k, v)
        return

    def write_video(self, video_save_path: str, fps: int = 24, frame_save_folder: str = None):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_save_path, fourcc, fps, (1920, 1080), True)

        if not os.path.isdir(frame_save_folder):
            os.mkdir(frame_save_folder)

        for k, v in self.video_path.items():
            for i, frame in enumerate(self.generate_diag_video(k, v)):
                print('Current process mode %s | Frame %d' % (k, i))
                writer.write(frame)
                if frame_save_folder is not None:
                    cv2.imwrite(os.path.join(frame_save_folder, '%s-%d.jpg' % (k, i)), frame)
        writer.release()
        return

    def play(self, mode, video_path):
        cv2.namedWindow('EBUSVideo')
        counter = 0
        start_time = time.time()
        for frame in self.generate_diag_video(mode, video_path):
            counter += 1
            cv2.imshow('EBUSVideo', frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("q"):
                break
            print("FPS: ", counter / (time.time() - start_time))
        cv2.destroyAllWindows()
        return

    def generate_diag_video(self, mode: str, video_path: str):
        cap = cv2.VideoCapture(video_path)  # 导入的视频所在路径
        width = cap.get(3)
        crop_range = CROP_RANGE[self.hospital][str(int(width))][mode]
        datatype = self.datatype_arrange[mode]

        seg_model = self.seg_model
        clf_model = self.clf_model

        aggregator = clf_model.branches[mode].aggregator.aggregator
        aggregated_feature = self.aggregated_feature
        presents_text = 'Collecting infos'

        min_num_frame, max_num_frame = self.num_threshold[mode]
        data_queue = PriorityQueue(max_num_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # cut to mini range
            try:
                cropped, relative_pos = pre_cut(frame, crop_range, mode)

            except ValueError:
                cv2.putText(frame, "Scanner Undetected!",
                            (800, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
                cv2.imshow('frame', frame)
                continue

            frame = present_scanner(frame, relative_pos)
            with torch.no_grad():
                # segment the image to ROI
                if mode == 'E':
                    mask = segment(seg_model, cropped[1], mode)
                else:
                    mask = segment(seg_model, cropped, mode)
                mask = np.expand_dims(mask, 2)

                if mode == 'B':
                    score = mask.sum().item()
                elif mode == 'F':
                    data_for_detect = cropped * mask
                    score = count_pixels(data_for_detect)

                if 'ROI' in datatype:
                    if mode == 'E':
                        data = mask * cropped[0]
                    else:
                        data = cropped * mask
                else:
                    data = cropped

                frame = augment(frame, mask, relative_pos, edge=True)

                # if gaussian_laplacian(cropped):
                un_blur = gaussian_laplacian(cropped[1]) if mode == 'E' else gaussian_laplacian(cropped)

                if un_blur:
                    if mode == 'E':
                        data = data[0]
                        qualified = color_detect(data)
                    elif mode == 'B':
                        qualified = data.astype(np.int).mean() > 30
                    else:
                        qualified = self.artefact_detect(data_for_detect)

                    if qualified:
                        data = self.data_format(data, mode)
                        data = data.unsqueeze(0)
                        if mode == 'B':
                            data = data.mean(2, keepdim=True)
                        feature = clf_model.branches[mode].extract(data)

                        if mode == 'E':
                            data_queue.push(1, DataInstance(feature, cropped[0]))
                        else:
                            data_queue.push(score, DataInstance(feature, cropped))

                        feature_list = data_queue.obtain_feature()
                        length = len(feature_list)
                        if length > 1:
                            feature = torch.cat(feature_list, dim=1)

                        if length >= min_num_frame:
                            if mode == 'E':
                                aggregated, weights = aggregator(feature,
                                                                 torch.ones((1, 1), dtype=torch.int8,
                                                                            device='cuda') * length, with_weights=True)
                                weights = weights.view(-1).cpu().tolist()
                                data_queue.reassign_priority(weights)
                                if len(data_queue) > (max_num_frame - 1):
                                    data_queue.pop()
                            else:
                                aggregated = aggregator(feature, torch.ones((1, 1), dtype=torch.int8,
                                                                            device='cuda') * length)

                            aggregated_feature[mode] = aggregated
                            result = clf_model.branches[mode].clf(aggregated)
                            result = nn.functional.softmax(result, dim=1)[0, 1].item()

                            if result > 0.5:
                                presents_text = 'Tends to Malignant'
                            else:
                                presents_text = 'Tends to Benign'

            if len(aggregated_feature) == 3:
                score = self.multimodal_decision(aggregated_feature)
                if score > 0.5:
                    diagnosis = 'Multimodal tends to Malignant'
                else:
                    diagnosis = 'Multimodal tends to Benign'

                cv2.putText(frame, diagnosis,
                            (1200, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)

            if len(data_queue) > min_num_frame:
                represent_image = data_queue.obtain_top_frame(1)[0]
                frame = embed_represent_image(frame, represent_image, (600, 70))
                cv2.putText(frame, 'The represent image as:',
                            (500, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            cv2.putText(frame, presents_text,
                        (1200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
            yield frame
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--example_video_dir', default='', type=str)
    parser.add_argument('--hospital', default='', type=str)
    parser.add_argument('--demo_video_path', default='', type=str)
    parser.add_argument('--demo_frames_path', default='', type=str)
    parser.add_argument('--mode', default='BFE', type=str)
    parser.add_argument('--pretrained_model', default='', type=str)

    args = parser.parse_args()
    
    vp = VideoPlayer(video_dir=args.example_video_dir,
                     hospital=args.hospital,
                     datatype_arrange={'B': 'raw-img', 'F': 'ROI', 'E': 'elastic-hist'},
                     mode='BFE',
                     clf_model_folder=args.pretrained_model)
    vp.write_video(args.demo_video_path, args.demo_frames_path)
