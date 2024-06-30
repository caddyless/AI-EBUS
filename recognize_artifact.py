import torch
import os

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datamanager.dataset import ArtifactDataset
from model.interface import obtain_net
from metainfo.schedule import get_schedule
from utils.tools import get_current_time


def find_threshold(result, sensitivity: float = 0.999):
    y_true = result.labels
    y_score = result.probability
    total = y_true.sum()

    positive_score = y_score[y_true == 1]
    sorted_score = np.sort(positive_score)
    tolerance = int(total * (1 - sensitivity))
    threshold = round(sorted_score[tolerance], 2)
    return threshold


def train(model_save_dir: str, epoch=50, lr=0.0005, batch_size=128):
    dataset = ArtifactDataset()

    train_set, test_set = ArtifactDataset.split(dataset, [0.7, 1.0])

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)

    writer = SummaryWriter('../tensorboard/recognize_artifact')
    param = {'model': 'seincept', 'net': 'typical', 'writer': writer,
             'in_channel': 3, 'is_parallel': True, 'loss_fn': 'focal'}
    net = obtain_net(**param)

    optimizer, scheduler = get_schedule('folder', net.model.parameters(), epoch, lr, 10)

    best_score = 0.0
    default_path = os.path.join(model_save_dir, 'best-model.pth')
    for e in range(epoch):
        net.train_loop(e, train_loader, optimizer)
        scheduler.step()
        result = net.test_loop(e, test_loader)

        score = result.data['AUC']
        if score > best_score:
            best_score = score
            state = {'param': net.model.state_dict(),
                     'threshold': find_threshold(result)}
            torch.save(state, default_path)
    model_save_path = os.path.join(model_save_dir, '%s-%.4f.pth' % (get_current_time(), best_score))
    os.rename(default_path, model_save_path)

    return best_score, model_save_path


def evaluation(model_save_path: str):
    dataset = ArtifactDataset()

    train_set, test_set = ArtifactDataset.split(dataset, [0.7, 1.0])

    test_loader = DataLoader(test_set, batch_size=32, num_workers=2)

    writer = SummaryWriter('../tensorboard/recognize_artifact')
    param = {'model': 'seincept', 'net': 'typical', 'writer': writer,
             'in_channel': 3, 'is_parallel': True, 'loss_fn': 'focal'}
    net = obtain_net(**param)
    state = torch.load(model_save_path)
    net.model.load_state_dict(state['param'])
    net.test_loop(0, test_loader)
    # result.curve.plot_curve('../')
    # result.save_data('../recognize.csv')
    # transform = tf.Compose([tf.Resize(224),
    #                         tf.CenterCrop(224),
    #                         tf.ToTensor()])
    # excel_file = '../raw-data/artifact-shot.xlsx'
    # df = pd.read_excel(excel_file)
    # map_dict = {}
    # for k, v in zip(df['file name'], df['label']):
    #     map_dict[k] = v
    #
    # writer = SummaryWriter('../tensorboard/recognize_artifact')
    # param = {'model': 'resnet18', 'net': 'typical', 'writer': writer,
    #          'in_channel': 3, 'is_parallel': True, 'loss_fn': 'focal'}
    # net = obtain_net(**param)
    # dict_save_path = '../models/20200811-recognize_artifacts.pth'
    # state = torch.load(dict_save_path)
    # net.model.load_state_dict(state['param'])
    # total = 0
    # correct = 0
    # source_dir = '../data/video/mm-ete/Bloodflow-artifact'
    # with torch.no_grad():
    #     net.eval()
    #     for file in os.listdir(source_dir):
    #         label = map_dict[file]
    #         image_path = os.path.join(source_dir, file)
    #         image = cv.imread(image_path)
    #         # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #
    #         tensor = transform(image).unsqueeze(0).cuda()
    #
    #         scores = net.model(tensor)
    #
    #         predicted = torch.argmax(scores, dim=1).item()
    #
    #         if predicted == label:
    #             correct += 1
    #         total += 1
    #
    # print('acc is %f' % (100 * correct / total))


if __name__ == '__main__':
    save_dir = '../models/recognize_artefact'
    s, save_path = train(model_save_dir=save_dir)
    print('best score is %.4f' % s)
    evaluation(save_path)
