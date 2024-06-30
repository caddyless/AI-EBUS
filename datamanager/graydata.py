import random
import torch

import torch.nn as nn
import numpy as np

from model.network.backbone.msnet import ms_u_net
from model.utils import post_process
from datamanager.video import DataManager, LowQualityError
from pathlib import Path
from torchvision import transforms as tf


class GrayManager(DataManager):
    def __init__(self, num_samples: int = 20, tidy_dir=None, excel_file=None, detect_roi=False):
        super().__init__(tidy_dir, excel_file)
        self.mode = 'B'
        self._shot_folder = 'gray-annotation'
        self.num_samples = num_samples
        self.cut_video_param = {
            'intensity_threshold': 5,
            'expected_intensity': 30
        }
        self.detect_roi = detect_roi
        if detect_roi:
            model = nn.DataParallel(ms_u_net(modes='BEF', init_ch=64, stage=5).cuda())
            params = torch.load('../models/segmentation/20201010-MS.pth')['param']
            model.module.load_state_dict(params)
            model.eval()
            self.model = model
            self.transform = tf.Compose([tf.Resize(224),
                                         tf.CenterCrop(224)])

    def quality_detect(self, image: torch.Tensor):
        mean = image.mean(dim=(1, 2, 3))
        mean_qualified_indices = (mean >= self.cut_video_param['intensity_threshold'])
        return mean_qualified_indices

    def read_video(self, video_path, hospital='shanghai', batch_size: int = 32, device: str = 'cuda'):
        num_samples = self.num_samples
        frames = self.capture_frame(video_path[0], hospital, gap=0)
        print(frames.size())
        # detect the mean value
        image = frames.float()
        mean_qualified_indices = self.quality_detect(image)
        frames = frames[mean_qualified_indices]

        print('The ratio satisfies mean intensity: %f' % (mean_qualified_indices.sum() / mean_qualified_indices.numel()))

        # frames = enhance_intensity(frames, self.cut_video_param['expected_intensity'])

        if mean_qualified_indices.sum().item() == 0:
            raise LowQualityError

        if self.detect_roi:
            with torch.no_grad():
                tensor = self.transform(image[mean_qualified_indices] / 255)
                dataloader = self.obtain_data_loader(tensor, batch_size)

                roi_qualified_indices = []
                for item in dataloader:
                    item = item[0]
                    item = {'B': item.cuda()}
                    output = self.model(item)['B']
                    output = (output > 0.5) + 0
                    output = output.cpu().detach().numpy()
                    for j in range(output.shape[0]):
                        img = output[j]
                        img = np.transpose(img, (1, 2, 0))
                        img = post_process(img)
                        ratio = (img > 0).sum() / img.size
                        if 0.1 < ratio < 0.85:
                            roi_qualified_indices.append(1)
                        else:
                            roi_qualified_indices.append(0)

                qualified = frames[roi_qualified_indices]

        else:
            qualified = frames

        if (length := qualified.size(0)) < num_samples:
            raise LowQualityError

        else:
            sample_indices = random.sample(list(range(length)), num_samples)
            qualified = qualified[sample_indices]
            # qualified = [enhance_intensity(image, self.cut_video_param['expected_intensity'])
            #              for image in qualified]
            return qualified

    def remove_condition(self, directory: Path):
        if len(list(directory.rglob('*.png'))) != self.num_samples:
            return True
        else:
            return False
