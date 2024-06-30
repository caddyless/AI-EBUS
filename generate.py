import argparse
import os

from utils.tools import convert_dict_yaml
from datamanager.elastic import ElasticManager
from datamanager.graydata import GrayManager
from datamanager.bloodflow import BloodManager
from datamanager.dataset import MMVideoDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='lymph', type=str)


def generate_specific_mode(generate_dir: str, mode: str = 'E', tidy_dir='../raw-data/tidy-data', excel_file=None,
                           from_scratch=True, is_raw=True, is_seg=True, num_b: int = 30, num_e: int = 50, num_f: int = 3,
                           max_batch_size: int = 32, **kwargs):

    for m in mode:
        if m == 'E':
            data_manager = ElasticManager(tidy_dir=tidy_dir, excel_file=excel_file, max_samples=num_e)
        elif m == 'F':
            data_manager = BloodManager(tidy_dir=tidy_dir, excel_file=excel_file, num_samples=num_f)
        elif m == 'B':
            data_manager = GrayManager(num_samples=num_b, tidy_dir=tidy_dir, excel_file=excel_file)
        else:
            raise ValueError('Unknown mode %s' % m)

        if is_raw:
            data_manager.generate_v_folder(generate_dir, from_scratch=from_scratch, batch_size=max_batch_size,
                                           device='cuda')

        if is_seg:
            data_manager.generate_seg(generate_dir, max_batch=max_batch_size, from_scratch=from_scratch)

    return


def generate(mode='BEF', database_path: str = None, data_folder: str = None, excel_path: str = None, from_scratch=False,
             dataset_save_path: str = None):
    mmv_dataset = MMVideoDataset(mode=mode, database_path=database_path)
    if (not from_scratch) and os.path.isfile(dataset_save_path):
        mmv_dataset.load(dataset_save_path)
    mmv_dataset.read_in_data(data_folder, excel_path, from_scratch)
    mmv_dataset.save(dataset_save_path)

    return


if __name__ == '__main__':
    args = parser.parse_args()

    if args.data == 'lymph':
        conf_file = 'conf/data-generate/118-data-lymph.yaml'
    elif args.data == 'tumor':
        conf_file = 'conf/data-generate/data-tumor.yaml'
    elif args.data == 'validation':
        conf_file = 'conf/data-generate/data-validation.yaml'
    else:
        conf_file = args.data

    diction = convert_dict_yaml(yaml_file=conf_file)

    if diction['need_data_folder']:
        generate_specific_mode(**diction['data_folder_param'])

    if diction['need_dataset']:
        generate(**diction['dataset_param'])
