import torch
import random

import torchvision.transforms.functional as F

from datamanager.video import DataManager
from datamanager.utils import mini_cut, LowQualityError, Rgb2hsv


class ElasticManager(DataManager):
    def __init__(self, cut_video_param: dict = None, tidy_dir=None, excel_file=None, max_samples: int = 30):
        super().__init__(tidy_dir, excel_file)
        self.mode = 'E'
        self._wait_seg_folder = 'gray'
        self._raw_img_folder = 'elastic'
        self._shot_folder = 'elastic-annotation'
        self.max_samples = max_samples
        if cut_video_param is None:
            self.cvp = {'color_tsd': 0.5,
                        'satu_tsd': 35,
                        'intensity_tsd': 60,
                        'ignore': 0,
                        'gap': 0,
                        'k': 5}
        else:
            self.cvp = cut_video_param

    def read_video(self, video_path: list, hospital='shanghai', batch_size: int = 32, device: str = 'cuda'):
        """
        This function used to read videos. With given video_path, this function read frames from video in batch by
        calling capture_frame function. Besides that, this function process images in further, and return available
        frames.
        :param hospital: The hospital this video belong to
        :param video_path: The file path of videos.
        :param batch_size: The maximum batch size while handling video
        :param device: The device tensor placed

        :rtype: object
        """
        standard = {k: v for k, v in self.cvp.items() if k in ['color_tsd', 'satu_tsd', 'intensity_tsd']}

        if (length := len(video_path)) == 0:
            return []

        elif length > 1:
            e1 = video_path[0]
            e1_q = self.parse_qualified(e1, hospital, batch_size, device, **standard, scaled=True)
            if len(e1_q) > self.cvp['k']:
                qualified = e1_q
                print('\nThe first video satisfies requirements!\n')

            else:
                print('start read other videos!')

                for v in video_path[1:]:
                    e1_q += self.parse_qualified(v, hospital, batch_size, device, **standard, scaled=True)

                if len(e1_q) < self.cvp['k']:
                    raise LowQualityError

                else:
                    qualified = e1_q

        else:
            print('There is only one video for %s!' % video_path[0])
            qualified = self.parse_qualified(video_path[0], hospital, batch_size, device, **standard, scaled=True)

            if len(qualified) < self.cvp['k']:
                raise LowQualityError

        if len(qualified) > self.max_samples:
            qualified = random.sample(qualified, self.max_samples)

        return qualified

    def parse_qualified(self, video_path, hospital='shanghai', batch_size: int = 32, device: str = 'cuda', satu_tsd=30,
                        color_tsd=0.75, intensity_tsd=105, scaled=False):

        frames = self.capture_frame(video_path, hospital, self.cvp['ignore'], self.cvp['gap'])
        frames = frames
        elastic_image = frames[:, 0]
        gray_image = frames[:, 1]

        try:
            elastic_tensor, mini_range, indices = mini_cut(elastic_image)
        except ValueError:
            print('Abnormal shape for video %s' % video_path)
            return []

        gray_image = gray_image[indices]

        qua_index = self.quality_detect(elastic_tensor, satu_tsd, color_tsd, intensity_tsd, batch_size, device, scaled)
        qualified = []
        if qua_index.size(0) > 0:
            elastic_tensor = elastic_tensor[qua_index]
            gray_image = gray_image[qua_index]
            mini_range = mini_range[qua_index]

            shape = elastic_tensor.size()

            for i in range(shape[0]):
                g = F.resized_crop(gray_image[i], *mini_range[i], [shape[2], shape[3]])
                qualified.append((elastic_tensor[i], g))

            return qualified

        else:
            return []

    def quality_detect(self, elastic_image: torch.Tensor, satu_tsd=30, color_tsd=0.75, intensity_tsd=105,
                       batch_size: int = 128, device: str = 'cuda', scaled=True, verbose=True, *args):
        total = elastic_image.size(0)
        max_length = self.max_samples
        total_mean = []
        total_ratio = []
        rgb2hsv = Rgb2hsv('rgb')
        print(elastic_image.size())
        dataloader = self.obtain_data_loader(elastic_image, batch_size)
        for item in dataloader:
            item = item[0]
            item = item.to(device)

            hsv = rgb2hsv(item)
            img = hsv.float()
            hsv_s = img[:, 1, :, :]
            hsv_v = img[:, 2, :, :]

            colored_crd = hsv_s > satu_tsd  # B * W * H
            gray_crd = (hsv_s > 0) & (~colored_crd)  # B * W * H

            num_colored_pixel = colored_crd.sum((1, 2))  # B
            total_in_scan = (hsv_v > 0).sum((1, 2))  # B
            ratio = num_colored_pixel / total_in_scan  # B

            mean = hsv_v.mean((1, 2))
            if scaled:
                refer_mean = hsv_v[gray_crd].mean()
                mean = mean * (90 / refer_mean)

            total_mean.append(mean)
            total_ratio.append(ratio)

        total_mean = torch.cat(total_mean)
        total_ratio = torch.cat(total_ratio)

        bottom_indices = ((total_mean > intensity_tsd) & (total_ratio > color_tsd)).nonzero().squeeze()

        try:
            num_over_line = bottom_indices.size(0)
        except IndexError as e:
            print(e)
            return torch.zeros(0)

        if num_over_line > max_length:
            qualified_score = total_mean[bottom_indices] / 100 + 1.5 * total_ratio[bottom_indices]
            _, indices = qualified_score.topk(max_length, sorted=False)
            bottom_indices = bottom_indices[indices]

        if verbose:
            print('Intensity keep %f, ratio keep %f' % ((total_mean > intensity_tsd).sum() / total_mean.numel(),
                                                        (total_ratio > color_tsd).sum() / total_ratio.numel()))
            print('color_judge keep %d, keep rate %f' % (bottom_indices.size(0),
                                                         bottom_indices.size(0) / total))

        return bottom_indices

    def write_image(self, parent_dir: str, sub_dir: str, images: list, info_dict: dict):

        elastic, gray = zip(*images)

        super().write_image(parent_dir, 'elastic', elastic, info_dict)
        super().write_image(parent_dir, self._wait_seg_folder, gray, info_dict)

        return
