import torch

from torchvision import transforms as tf
from model.network.backbone.seincept import seincept
from datamanager.video import DataManager
from datamanager.utils import mini_cut, VideoSizeError, Rgb2hsv, LowQualityError


class BloodManager(DataManager):
    def __init__(self, num_samples: int = 3, tidy_dir = None, excel_file = None):
        super().__init__(tidy_dir, excel_file)
        self.mode = 'F'
        self._shot_folder = 'doppler-annotation'
        self.num_samples = num_samples

        self.model = seincept(in_channel=3).cuda()
        dict_save_path = '../models/recognize_artefact/2020-09-19 16:55:18-0.9739.pth'
        state = torch.load(dict_save_path)
        parameters = {}
        for k, v in state['param'].items():
            parameters[k.replace('module.', '')] = v
        self.model.load_state_dict(parameters)
        self.model.eval()
        self.threshold = state['threshold']
        self.transform = tf.Compose([tf.Resize(224),
                                     tf.CenterCrop(224)])

    def quality_detect(self, images: torch.Tensor, batch_size: int):
        try:
            with torch.no_grad():
                image = images.float() / 255
                image = self.transform(image)
                dataloader = self.obtain_data_loader(image, batch_size)

                qua_index = []
                for item in dataloader:
                    item = item[0]
                    item = item.cuda()
                    output = self.model(item)
                    score = torch.softmax(output, dim=1)[:, 1]
                    # qua_index.append(torch.ones((item.size(0), ), dtype=torch.long))
                    qua_index.append(score < 0.4)
                qua_index = torch.cat(qua_index)

        except VideoSizeError as e:
            print(e)
            return []

        return qua_index

    def read_video(self, video_path, hospital='shanghai', batch_size: int = 32, device: str = 'cuda') -> torch.Tensor:
        rgb2hsv = Rgb2hsv('rgb')
        video_path = video_path[0]
        frames = self.capture_frame(video_path, hospital, gap=2)
        print(frames.size())
        try:
            frames, _, _ = mini_cut(frames)
        except ValueError as e:
            print(e)
            print('patient %s video frames shape abnormal!' % video_path)

        print(frames.size())

        qua_index = self.quality_detect(frames, batch_size)

        if (length := qua_index.sum().item()) >= self.num_samples:
            qualified = frames[qua_index]
            hsv = rgb2hsv(qualified)
            hsv_s = hsv[:, 1]
            num_pixel = (hsv_s > 30).sum((1, 2))  # B
            _, indices = num_pixel.sort(descending=True)
            return qualified[indices[:self.num_samples]]

        else:
            raise LowQualityError('Only %d frames are qualified in video %s' % (length, video_path))
