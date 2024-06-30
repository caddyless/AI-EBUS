import os
import random
import json
import torch
import operator

import cv2 as cv
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from pathlib import Path
from scipy import stats
from PIL import Image, ImageDraw
from scipy.ndimage.filters import convolve
from sklearn.decomposition import PCA
from metainfo.default import EXCEL_PATH, excel_keys, HOSPITAL_KEY


class LowQualityError(Exception):

    def __init__(self, err='Low quality video!'):
        Exception.__init__(self, err)


class VideoSizeError(Exception):

    def __init__(self, err='Unexpected video size!'):
        Exception.__init__(self, err)


class TooLongVideoError(Exception):

    def __init__(self, err='Too long video!'):
        Exception.__init__(self, err)


def interpret_scope(scope):
    if scope == 'all':
        return True

    elif isinstance(scope, str):
        scope_list = []
        scope = scope.replace(' ', '')
        intervals = scope.split(',')
        for item in intervals:
            start, end = item.split('-')[:2]
            start, end = int(start), int(end)
            scope_list += [i for i in range(start, end + 1)]
        return scope_list

    elif isinstance(scope, (list, tuple, set)):
        return scope

    else:
        raise ValueError('Unknown type of scope!')


def pca_reduction(data, dims=None):
    pca = PCA()
    pca.fit(data)
    ratios = pca.explained_variance_ratio_
    if dims is None:
        n = 0
        while ratios[:n].sum() < 0.99:
            n += 1
        dims = n
    directions = pca.components_[:dims, :].T
    rdc_data = np.matmul(data, directions)
    return rdc_data, directions


def lacked_samples(target_dir):
    sample_dicts = read_excel(EXCEL_PATH)
    print(len(sample_dicts))
    existence = [parse_number(f) for f in os.listdir(target_dir)]
    existence = set(existence)
    need_handle = [s for s in sample_dicts if not s['id'] in existence]
    print('Total %d waiting for process' % len(need_handle))
    return need_handle


def enhance_intensity(image: torch.Tensor, target_intensity: int = 40):
    b, c, w, h = image.size()
    image_tensor = image.float()
    current_mean = image_tensor.mean((1, 2, 3))

    weights = target_intensity / current_mean
    weights[weights < 1] = 1

    image_tensor = image_tensor * weights.view(b, 1, 1, 1)

    image_tensor[image_tensor > 255] = 255
    image_tensor = image_tensor.byte()

    return image_tensor


def estimate_frame(video_path):
    try:
        vid_cap = cv.VideoCapture(video_path)
    except cv.error:
        print('%s error or corrupted!' % video_path)
        return 0
    fps = vid_cap.get(cv.CAP_PROP_FPS)
    if fps == 0:
        return 0
    frames = vid_cap.get(cv.CAP_PROP_FRAME_COUNT)
    duration = frames / fps
    real_frames = int(duration * 24)
    vid_cap.release()
    return real_frames


def mean_intensity(image: np.ndarray):
    if len(image.shape) == 3:
        intensity = np.max(image, 2)
    else:
        intensity = image.copy()
    mean_i = intensity[intensity != 0].mean()
    scaled_image = (80 / mean_i) * image.astype(np.float)
    scaled_image[scaled_image > 255] = 255
    image = (scaled_image + 0.5).astype(np.uint8)
    return image


def convert_image(path, aug='none', bin_size=32):
    num_bins = (256 // bin_size) ** 3
    hist = np.zeros(num_bins, dtype=np.float)

    if isinstance(path, str):
        img = cv.imread(path)
    else:
        img = path

    if aug == 'intensity':
        img = mean_intensity(img)

    std = img.std(axis=2)
    pixels = img[std > 30, :]
    count = pixels.size // 3
    if count == 0:
        return hist

    bins = pixels / bin_size
    bins = bins.astype(np.int)
    scale = np.array([1, 8, 64], dtype=np.int)
    bins = (bins * scale).sum(1)
    for j in range(num_bins):
        hist[j] = (bins == j).sum()
    hist = hist / hist.sum()
    return hist


def shape_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) > 2, 'Polygon must have points more than 2'
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def max_range(points):
    positions = [296, 693, 1464, 1834]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    positions[0] = max(int(min(y)), positions[0])
    positions[1] = min(int(max(y)), positions[1])
    positions[2] = max(int(min(x)), positions[2])
    positions[3] = min(int(max(x)), positions[3])
    return positions


def roi_mask(path, shape):
    json_path = path.split('.j')[0] + '.json'
    if not os.path.isfile(json_path):
        raise IOError('json does not exist')
    with open(json_path, encoding='utf-8') as f:
        diction = json.load(f)
        points = diction['shapes'][0]['points']
        positions = max_range(points)
    mask = shape_to_mask(shape, points).astype(int)
    return mask, positions


def is_overlap(x: torch.Tensor, y: torch.Tensor):
    n = x.size(0)
    m = y.size(0)

    distance = (x.view(n, 1, 2) - y.view(1, m, 2)).abs().sum(2)

    return (distance < 10).any()


def get_bounds(img: torch.Tensor):
    """
    This function returns the bound of input batch of images

    :param img: The input image
    :rtype: torch.Tensor
    """
    print(img.size(), img.dtype, img.max(), img.min())
    r = img[:, 0, ...]
    g = img[:, 1, ...]
    b = img[:, 2, ...]
    mask = ((g - r) > 50) * ((g - b) > 50) * (g > 90)

    b, w, h = mask.size()
    lower_bound = w
    mini_range = torch.zeros(b, 4, dtype=torch.short)

    for i in range(b):
        indicator = mask[i]
        indicator = torch.nonzero(indicator)
        if indicator.size(0) == 0:
            continue
        top = indicator[:, 0].min().item()
        bottom = indicator[:, 0].max().item()
        left = indicator[:, 1].min().item()
        right = indicator[:, 1].max().item()

        bottom_indices = indicator[indicator[:, 0] == bottom]
        left_indices = indicator[indicator[:, 1] == left]
        right_indices = indicator[indicator[:, 1] == right]

        if is_overlap(bottom_indices, left_indices) or is_overlap(bottom_indices, right_indices):
            bottom = lower_bound

        mini_range[i] = torch.tensor([top, left, bottom - top, right - left])

    return mini_range


def mini_cut(image: torch.Tensor):
    """
    This function receive torch.Tensor images in color channel R,G,B

    :rtype: torch.Tensor
    """
    if isinstance(image, torch.ByteTensor) or image.dtype == torch.uint8:
        img = image.int()
    elif isinstance(image, torch.FloatTensor):
        if (image > 1).sum() == 0:
            img = (image * 255).short()
        else:
            img = image.short()
    else:
        raise ValueError('Unsupported datatype of %s for image!' % image.dtype)

    if (length := len(image.size())) == 3:
        img = image.unsqueeze(0)
    elif length == 2:
        img = image.unsqueeze(0).unsqueeze(0)

    mini_range = get_bounds(img)

    if mini_range.device == 'cuda':
        mini_range = mini_range.cpu()
        mode, _ = torch.mode(mini_range, dim=0)
        standard_size = [mode[2].item(), mode[3].item()]
        mode = mode.cuda()
    else:
        mode, _ = torch.mode(mini_range, dim=0)
        standard_size = [mode[2].item(), mode[3].item()]

    if any([item < 50 for item in standard_size]):
        raise ValueError('Abnormal standard_size %s!' % str(standard_size))

    residual = (mini_range - mode).abs().sum(1)
    indices = residual < 60
    print('shape keep %f samples' % (indices.sum() / indices.numel()))

    img = image[indices]
    mini_range = mini_range[indices]

    qualified = []
    for i in range(img.size(0)):
        qualified.append(F.resized_crop(img[i].unsqueeze(0), *mini_range[i], standard_size))
    qualified = torch.cat(qualified, dim=0)

    return qualified, mini_range, indices


class HistogramEquation(object):

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        image = np.array(image)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv.equalizeHist(image)
        image = Image.fromarray(equ)
        return image


class IntensityNormal(object):

    def __init__(self, mean=80):
        super().__init__()
        self.mean = mean

    def __call__(self, image):
        image = np.array(image)
        assert len(image.shape) == 2, print('The input to IntensityNormal must be gray image')
        mean_i = image[image != 0].mean()
        scaled_image = (self.mean / mean_i) * image.astype(np.float)
        scaled_image[scaled_image > 255] = 255
        image = (scaled_image + 0.5).astype(np.uint8)
        image = Image.fromarray(image)
        return image


class LocalContrastNormalization(object):
    """
    Conduct a local contrast normalization algorithm by
    Pinto, N., Cox, D. D., and DiCarlo, J. J. (2008). Why is real-world visual object recognition hard?
     PLoS Comput Biol , 4 . 456 (they called this "Local input divisive normalization")

    the kernel size is controllable by argument kernel_size.
    """

    def __init__(self, p=0.5, kernel_size=3, mode='constant', cval=0.0):
        """

        :param kernel_size: int, kernel(window) size.
        :param mode: {'reflect', 'constant', 'nearest', 'mirror', 'warp'}, optional
                        determines how the array borders are handled. The meanings are listed in
                        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.html
                        default is 'constant' as 0, different from the scipy default.

        """
        self.kernel_size = kernel_size
        self.mode = mode
        self.cval = cval
        self.p = p

    def __call__(self, tensor):
        """

        :param tensor: Tensor image os size (C, H, W) to be normalized.
        :return:
            Tensor: Normalized Tensor image, in size (C, H, W).
        """
        p = np.array([1 - self.p, self.p])
        flag = np.random.choice([0, 1], p=p.ravel())
        if flag == 0:
            return tensor
        C, H, W = tensor.size()
        kernel = np.ones((self.kernel_size, self.kernel_size))

        arr = np.array(tensor)
        local_sum_arr = np.array([convolve(arr[c], kernel, mode=self.mode, cval=self.cval)
                                  for c in range(C)])  # An array that has shape(C, H, W)
        # Each element [c, h, w] is the summation of the values
        # in the window that has arr[c,h,w] at the center.
        local_avg_arr = local_sum_arr / (self.kernel_size ** 2)  # The tensor of local averages.

        arr_square = np.square(arr)
        local_sum_arr_square = np.array([convolve(arr_square[c], kernel, mode=self.mode, cval=self.cval)
                                         for c in range(C)])  # An array that has shape(C, H, W)
        # Each element [c, h, w] is the summation of the values
        # in the window that has arr_square[c,h,w] at the center.
        local_norm_arr = np.sqrt(local_sum_arr_square)  # The tensor of local Euclidean norms.

        local_avg_divided_by_norm = local_avg_arr / (1e-8 + local_norm_arr)

        result_arr = np.minimum(local_avg_arr, local_avg_divided_by_norm)
        return torch.Tensor(result_arr)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, threshold={1})'.format(self.kernel_size, self.threshold)


def read_excel(excel=EXCEL_PATH):
    df = pd.DataFrame(pd.read_excel(excel))
    # drop abnormal values
    df.dropna(axis=0, how='all', inplace=True)
    df.rename(columns=excel_keys, inplace=True)
    df = df[~df['BM'].isnull()]
    df = df.fillna(0)
    # replace key
    int_format_keys = ['id', 'age', 'fine', 'coarse', 'BM']
    for k in int_format_keys:
        if k in df.columns:
            df[k] = df[k].astype('int')

    str_format_keys = ['hospital', 'name', 'sex', 'local', 'diag']
    for k in str_format_keys:
        if k in df.columns:
            df[k] = df[k].astype('str')

    df['date'] = df['date'].apply(regular_date)
    # replace hospital name to shorten form
    for k, v in HOSPITAL_KEY.items():
        if 'hospital' in df.columns:
            df['hospital'].replace(k, v, inplace=True)

    lymph_dicts = [dict(df.iloc[i]) for i in range(df.shape[0])]

    return lymph_dicts


def split_data(k):
    df = pd.DataFrame(pd.read_excel(EXCEL_PATH))
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    disease = df['diag']
    number = df['id']
    category = {}
    for d, n in zip(disease, number):
        if d in category.keys():
            category[d].append(n)
        else:
            category[d] = [n]

    def split_numbers(numbers):
        splited = [[] for i in range(k)]
        length = len(numbers)
        basic_len = length // k
        for i in range(k):
            splited[i].extend(numbers[i * basic_len: (i + 1) * basic_len])
        residual = length % k
        handle = random.sample(list(range(k)), residual)
        for i, h in enumerate(handle):
            splited[h].append(numbers[k * basic_len + i])
        return splited

    result = [[] for i in range(k)]
    for c in category.values():
        temp = split_numbers(c)
        for i, ele in enumerate(result):
            ele += temp[i]

    return result


def regular_info(ln_dicts):
    for ln in ln_dicts:
        ln['id'] = ln['id']


def regular_date(date):
    date = str(date)
    date = date.split(' ')[0]
    date = date.replace('-', '')
    return date


def normalize_image(image, mean, std):
    shape = image.shape
    if len(shape) == 3 and shape[-1] == 3:
        image = image.transpose((2, 0, 1))
        mean = mean.reshape((3, 1, 1))
        std = mean.reshape((3, 1, 1))
    elif len(shape) == 4 and shape[-1] == 3:
        image = image.transpose((0, 3, 1, 2))
        mean = mean.reshape((1, 3, 1, 1))
        std = mean.reshape((1, 3, 1, 1))
    image = (image - mean) / std
    return image


def parse_number(name):
    i = 0
    number = -1
    while True:
        try:
            number = int(name.split('-')[i])
            break
        except ValueError:
            i += 1
            continue
    return number


class Rgb2hsv(torch.nn.Module):
    def __init__(self, mode: str = 'rgb'):
        super().__init__()
        if mode == 'rgb':
            self.rgb_index = (0, 1, 2)
        elif mode == 'bgr':
            self.rgb_index = (2, 1, 0)
        else:
            raise ValueError('Unknown mode %s' % mode)

    def forward(self, x: torch.Tensor):
        x = x.float()
        r, g, b = x[:, self.rgb_index[0]], x[:, self.rgb_index[1]], x[:, self.rgb_index[2]]

        value, max_indices = x.max(dim=1)
        minimum, _ = x.min(dim=1)
        delta = value - minimum

        hue = torch.zeros_like(value)
        hue[max_indices == self.rgb_index[0]] = ((g - b) / delta % 6)[max_indices == self.rgb_index[0]]
        hue[max_indices == self.rgb_index[1]] = ((b - r) / delta + 2)[max_indices == self.rgb_index[1]]
        hue[max_indices == self.rgb_index[2]] = ((r - g) / delta + 4)[max_indices == self.rgb_index[2]]
        hue[delta == 0] = 0
        hue /= 6

        saturation = torch.where(value == 0, torch.tensor(0., device=value.device), delta / value)

        hue = hue * 180
        saturation = saturation * 255

        hsv = torch.cat((hue.unsqueeze_(1), saturation.unsqueeze_(1), value.unsqueeze_(1)), dim=1)
        hsv = hsv.byte()

        return hsv


def gaussian_laplacian(img: np.ndarray, threshold: float = 100.0):
    """

    return whether the img normal
    """
    shape = img.shape
    if len(shape) == 3 and shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = cv.resize(img, dsize=(569, 469))

    gauss_blur = cv.GaussianBlur(img, (3, 3), 0)
    transform = cv.convertScaleAbs(gauss_blur)
    grey_laplace = cv.Laplacian(transform, cv.CV_16S, ksize=3)
    result_img = cv.convertScaleAbs(grey_laplace)

    mean, std = cv.meanStdDev(result_img)
    blur_per = std[0][0] ** 2

    return blur_per >= threshold


def laplacian(img: np.ndarray, threshold: tuple = (200, 300)):
    shape = img.shape

    if len(shape) == 3 and shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur_per = cv.Laplacian(img, cv.CV_64F).var()

    return threshold[0] <= blur_per <= threshold[1]


def read_folder(folder: str, channel: int = None, clean_data: bool = False) -> np.ndarray:
    """
    This function used to read images in a given folder. In default case, the shape
    of images in given folder is variant slightly. This function can handle these case,
    and find the mode shape to return.
    If clean_data is True, images in given folder are regarded as in the same size.

    :rtype: object
    """

    folder = Path(folder) if isinstance(folder, str) else folder
    file_names = [item.name for item in folder.iterdir()]
    file_names.sort()
    images = [cv.imread(str(folder / file)) for file in file_names]
    length = len(images)

    if clean_data:

        shape = [length] + list(images[0].shape)
        if channel == 1:
            shape[3] = 1
        image_data = np.zeros(shape, dtype=np.uint8)

        for i, f in enumerate(folder.iterdir()):
            img = cv.imread(str(f))
            if channel == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 2)
            image_data[i] = img

    else:

        shapes = np.zeros((length, 3), dtype=np.int)
        for i in range(length):
            shapes[i] = images[i].shape

        assert (shapes[:, 2] == shapes[0, 2]).all(), 'The channel must be the same in the same folder %s!' % str(folder)

        mode, count = stats.mode(shapes)

        standard_shape = [length] + mode.squeeze().tolist()
        if channel == 1:
            standard_shape[3] = 1

        image_data = np.zeros(standard_shape, dtype=np.uint8)
        for i, img in enumerate(images):
            if channel == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if operator.eq(img.shape, standard_shape):
                image_data[i] = img

            else:
                resized = cv.resize(img, (standard_shape[2], standard_shape[1]))
                if len(resized.shape) == 2:
                    resized = np.expand_dims(resized, axis=2)
                image_data[i] = resized

    # change the order of channel and exchange channel R and B
    image_data = image_data.transpose(0, 3, 1, 2)
    if image_data.shape[1] == 3:
        image_data = image_data[:, [2, 1, 0], :, :]

    return image_data


def read_hist(folder: str):
    folder = Path(folder) if isinstance(folder, str) else folder

    files = [str(file) for file in folder.iterdir()]
    length = len(files)

    data = np.zeros((length, 512), dtype=np.float)

    for i in range(length):
        data[i] = convert_image(files[i], aug='none')

    return data
