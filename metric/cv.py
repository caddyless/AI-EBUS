import pandas as pd
import numpy as np

from pathlib import Path
from metainfo.default import RESULT_RECORD_FILE
from metric.classify import Recorder, MultiModalRecorder
from metric.utils import classify_sta, classify_record, seg_sta, seg_record


class Summary(object):
    def __init__(self, task: str = 'classify', k_fold: int = 5, timestamp: str = '0-0-0 0:0:0', remark: str = '',
                 select_metric: list = None, default_params: dict = None, ergodic_param: dict = None):

        self.k_fold = k_fold

        if select_metric is None:
            select_metric = ['AUC', 'Accuracy', 'F-score', 'epoch']

        if default_params is None:
            default_params = {}

        if ergodic_param is None:
            ergodic_param = {}

        if task == 'classify':
            statistic_metric = classify_sta
            record_metric = classify_record

        elif task == 'segmentation':
            statistic_metric = seg_sta
            record_metric = seg_record

        else:
            raise ValueError('Unknown keys task!')

        self.select_metric = select_metric  # metrics for select best model in each fold
        self.statistic_metric = statistic_metric  # metrics need calculate average and standard variance
        self.record_metric = record_metric  # metrics need to be recorded

        basic_param = ['Date', 'Task', 'Default Params', 'Ergodic Params', 'Remark']

        self.recorders = [Recorder(select_metric) for i in range(k_fold)]

        self.content = {k: [''] for k in basic_param}
        self.content['Date'][0] = timestamp
        self.content['Task'][0] = task
        self.content['Remark'][0] = remark
        self.content['Default Params'][0] = str(default_params)
        self.content['Ergodic Params'][0] = str(ergodic_param)

    def fill_statistical_content(self):
        # params reserve
        select_metric = self.select_metric
        statistic_metric = self.statistic_metric
        record_metric = self.record_metric
        rows = len(select_metric)
        column = len(statistic_metric)

        # write in statistic part
        scores = np.zeros((rows, self.k_fold, column), dtype=np.float)
        for i, r in enumerate(self.recorders):
            for j, k in enumerate(select_metric):
                for n, sub_k in enumerate(record_metric):
                    self.content[sub_k][i * len(select_metric) + j] = r.save_dict[k].result[sub_k]
                for n, sub_k in enumerate(statistic_metric.values()):
                    scores[j, i, n] = r.save_dict[k].result[sub_k]

        mean = scores.mean(1)
        std = scores.std(1)
        for k in statistic_metric:
            self.content[k] = [''] * rows
        for i in range(rows):
            for j, k in enumerate(statistic_metric.keys()):
                self.content[k][i] = '{:.4f} +- {:.4f}'.format(mean[i, j], std[i, j])

        return

    def update_from_recorders(self):
        # write in content in recorder
        self.content['k-fold'] = ['']
        for i, r in enumerate(self.recorders):
            output, length = r.output()
            self.content['k-fold'].extend([''] * length)
            self.content['k-fold'][i * length] = '%d' % i
            for k in output:
                if k in self.content.keys():
                    self.content[k].extend(output[k])
                else:
                    self.content[k] = output[k]

    def write(self, save_path: str = RESULT_RECORD_FILE):

        save_path = Path(save_path)
        parent = save_path.parent
        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True)

        self.fill_statistical_content()
        self.update_from_recorders()

        # complement the content length
        max_length = max([len(v) for k, v in self.content.items()])
        for k in self.content.keys():
            current_length = len(self.content[k])
            if current_length < max_length:
                self.content[k].extend([''] * (max_length - current_length))

        df = pd.DataFrame(self.content)
        df.to_csv(save_path, header=True, index=False, mode='a')
        return

    def plot_roc(self, save_dir):
        for r in self.recorders:
            r.save_dict['AUC'].curve.plot_curve(save_dir)
            r.save_dict['Accuracy'].curve.plot_curve(save_dir)
        return


class MultiModalSummary(Summary):
    def __init__(self, task: str = 'classify', modes: list = 'BEF', k_fold: int = 5, timestamp: str = '0-0-0 0:0:0',
                 remark: str = '', select_metric: list = None, **kwargs):
        super().__init__(task, k_fold, timestamp, remark, select_metric, **kwargs)

        self.modes = modes
        self.recorders = [MultiModalRecorder(modes, self.select_metric) for i in range(k_fold)]

    def fill_statistical_content(self):
        # params reserve
        select_metric = self.select_metric
        statistic_metric = self.statistic_metric
        rows = len(select_metric)
        column = len(statistic_metric)
        num_modes = len(self.modes)

        # write in statistic part
        scores = np.zeros((self.k_fold, num_modes, rows, column), dtype=np.float)
        for i, r in enumerate(self.recorders):
            for q, w in enumerate(self.select_metric):
                for j, m in enumerate(self.modes):
                    for e, t in enumerate(self.statistic_metric.values()):
                        scores[i, j, q, e] = r.save_dict[w][m].result[t]

        mean = scores.mean(0).reshape((num_modes * rows, column))
        std = scores.std(0).reshape((num_modes * rows, column))
        self.content['Statistic Modes'] = [''] * (rows * num_modes)
        for k in statistic_metric:
            self.content[k] = [''] * (rows * num_modes)

        for i in range(rows * num_modes):
            if i % rows == 0:
                self.content['Statistic Modes'][i] = self.modes[i // rows]
            for j, k in enumerate(statistic_metric.keys()):
                self.content[k][i] = '{:.4f} +- {:.4f}'.format(mean[i, j], std[i, j])

        return
