import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from metric.utils import classify_record
from operator import eq
from timeit import default_timer as timer


def evaluation(y_true, y_pred):
    sen, spe, pos, neg, fs = sen_spe_pos_neg_fs(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return [acc, fs, sen, spe, pos, neg]


def sen_spe_pos_neg_fs(y_true, y_pred, beta=1.0, task='binary', average='binary'):
    try:
        if task == 'binary':
            cm = confusion_matrix(y_true, y_pred)
            tp = cm[1, 1]
            fp = cm[0, 1]
            tn = cm[0, 0]
            fn = cm[1, 0]
            sen = tp / (tp + fn)
            spe = tn / (tn + fp)
            pos = tp / (tp + fp)
            neg = tn / (tn + fn)

        else:
            mcm = multilabel_confusion_matrix(y_true, y_pred)
            if average == 'micro':
                tp = mcm[:, 1, 1].sum()
                fp = mcm[:, 0, 1].sum()
                tn = mcm[:, 0, 0].sum()
                fn = mcm[:, 1, 0].sum()
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                pos = tp / (tp + fp)
                neg = tn / (tn + fn)
            elif average == 'macro':
                tp = mcm[:, 1, 1]
                fp = mcm[:, 0, 1]
                tn = mcm[:, 0, 0]
                fn = mcm[:, 1, 0]
                sen = (tp / (fn + tp)).mean()
                spe = (tn / (tn + fp)).mean()
                pos = (tp / (tp + fp)).mean()
                neg = (tn / (tn + fn)).mean()

            else:
                raise ValueError('average must be micro or macro')

        f_score = (1 + beta ** 2) * pos * sen / (beta ** 2 * pos + sen)

    except ZeroDivisionError:
        sen = spe = pos = neg = f_score = 0.0

    return sen, spe, pos, neg, f_score


class ReDiagram(object):
    def __init__(self, m=5, **arguments):
        self.m = m
        self.ece = 1.0
        self.confidence = np.zeros(m, dtype=np.float64)
        self.accuracy = np.zeros(m, dtype=np.float64)

        if len(arguments.keys()) == 2:
            self.analyze(**arguments)

    def analyze(self, y_true, y_score):
        m = self.m
        count = m
        for i in range(m):
            index = (i / m <= y_score[:, 1]) * (y_score[:, 1] < (i + 1) / m)
            if not index.any():
                self.confidence[i] = 0
                self.accuracy[i] = 0
                count -= 1
            else:
                self.confidence[i] = y_score[index].mean()
                self.accuracy[i] = y_true[index].mean()
        self.ece = np.abs(self.confidence - self.accuracy).sum() / count
        return self.ece

    def plot_diagonal(self, save_dir):
        m = self.m
        x = [i / 10 + 0.05 for i in range(m)]
        plt.figure()
        plt.plot(self.confidence, self.accuracy, 'r', x, x, 'b')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.savefig(os.path.join(save_dir, 'rd-{:.4f}.png'.format(self.ece)))
        return


class Analyzer(object):
    r"""
    This class is used to calculate classification metrics from the output of model.
    Only support binary or multiclass classification task now.

    Arguments:
        :param y_true: The ground truth label, expect to be a np.ndarray whose shape is (n_samples,), dtype=np.int
        :param y_score: The output probability of model. It's expected to be np.ndarray with shape (n_samples,) for
            binary classification and np.ndarray with shape (n_samples, n_class) for multiclass classification.
            dtype=np.float.
        :param identity: The id of samples. It's expected to be np.ndarray with shape (n_samples,), dtype=np.int
        :param epoch: The epoch where current result returned. type=int
        :param threshold: The decision threshold, default to be 0.5, and it only be effective in binary classification
            scenario.

    Attributes:
        raw_data: save the raw input y_true, y_score, identity and threshold in dict.
        result: save the analyze result. type=dict
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray, identity: np.ndarray = None, epoch: int = 0,
                 threshold: float = 0.5, **kwargs):
        self.raw_data = locals()
        self.result = self.analyze(threshold)

    def analyze(self, threshold: float = 0.5, average: str = 'macro', beta: float = 1.0) -> dict:
        """
        :param threshold: The decision threshold for binary classification
        :param average: The average method for multiclass scenario
        :param beta: The beta hyper-parameter for F_score

        :rtype: dict
        """

        raw_data = self.raw_data
        y_true, y_score, epoch = raw_data['y_true'], raw_data['y_score'], raw_data['epoch']

        result = {'epoch': epoch, 'threshold': threshold, 'AUC': roc_auc_score(y_true, y_score, average=average)}

        if len(shape := y_score.shape) == 1:
            task = 'binary'
        elif len(shape) == 2:
            if shape[1] == 2:
                task = 'binary'
                y_score = y_score[:, 1]
                self.raw_data['y_score'] = y_score
            else:
                task = 'multiclass'
        else:
            raise ValueError('unexpected shape of y_score %s' % str(shape))

        if task == 'binary':
            y_pred = (y_score > threshold).astype(np.int)

        else:
            y_pred = y_score.argmax(axis=1)

        result.update({'Accuracy': accuracy_score(y_true, y_pred)})

        sen, spe, pos, neg, f_score = sen_spe_pos_neg_fs(y_true, y_pred, beta=beta, task=task, average=average)
        result.update({'Sensitivity': sen, 'Specificity': spe, 'Positivity': pos, 'Negativity': neg,
                       'F-score': f_score})

        return result

    def plot_roc_curve(self, save_path: str):
        """
        This function is used to plot roc curve

        :param save_path: The save path for roc curve

        :rtype: None
        """
        y_true, y_score = self.raw_data['y_true'], self.raw_data['y_score']

        fpr, tpr, threshold = roc_curve(y_true, y_score)

        fig = plt.figure(figsize=(8, 8), dpi=400)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.plot(fpr, tpr)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        fig.suptitle('ROC curve')
        fig.savefig(os.path.join(save_path, '{:.4f}.png'.format(self.result['AUC'])))
        return

    def output_data(self, without_key=False, clean=False):
        result = self.result
        out = []

        if clean:
            out.append('{:.2f}%'.format(100 * result['Accuracy']))
            out.append('{:.4f}'.format(result['AUC']))

        for k in classify_record:
            v = result[k]
            if k in ['AUC', 'F-score', 'threshold']:
                if without_key:
                    out.append('{:6.4f}'.format(v))
                else:
                    out.append('{}:{:6.4f}'.format(k, v))
            elif k == 'epoch':
                if without_key:
                    out.append('{:4d}'.format(int(v)))
                else:
                    out.append('{}:{:4d}'.format(k, int(v)))
            else:
                try:
                    if without_key:
                        out.append('{:6.2f}'.format(100 * v))
                    else:
                        out.append('{}:{:6.2f}%'.format(k, 100 * v))
                except TypeError as e:
                    print(e)
                    out.append('{}:{:6.2f}%'.format(k, 0))

        return out

    def save_data(self, save_path):
        data = {'y_true': self.raw_data['y_true']}
        if (identity:= self.raw_data['identity']) is not None:
            data.update({'identity': identity})

        y_score = self.raw_data['y_score']

        if len(shape:= y_score.shape) == 1:
            data['y_score'] = y_score
        else:
            for i in range(shape[1]):
                data['class %d' % i] = y_score[:, i]

        df = pd.DataFrame(data)
        df.to_csv(save_path)

        return


class Recorder(object):
    def __init__(self, save_keys):
        self.save_keys = save_keys

        self.time_consume = ''
        self.save_dict = None
        self.start = None
        self.length = None

    def _type_check(self, key, result):
        if key not in self.save_keys:
            print('Warning, the given key %s not included by this recorder! '
                  'The receive filed are %s.' % (key, str(self.save_keys)))
            return False

        if not isinstance(result, Analyzer):
            print('The positional parameter result expected to be instance of Result, '
                  'but got %s!' % type(result))
            return False

        return True

    def update(self, key, result):

        if self._type_check(key, result):

            if self.save_dict is None:
                self.save_dict = {key: result}
            else:
                self.save_dict.update({key: result})
        return

    def output(self):
        if self.save_dict is None:
            raise ValueError('The save_dict has not been updated!')

        length = len(self.save_keys)
        out = {'Time Consume': [self.time_consume] + [''] * (length - 1),
               'Select Metric': [str(item) for item in self.save_keys]}

        numerical_result = {}
        for key in self.save_keys:
            result = self.save_dict[key]
            content = result.output_data()
            for item in content:
                k, v = item.split(':')
                if k in numerical_result.keys():
                    numerical_result[k].append(v)
                else:
                    numerical_result[k] = [v]

        out.update(numerical_result)

        return out, length

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = timer() - self.start
        h = int(elapsed / 3600)
        m = int((elapsed % 3600) / 60)
        s = int((elapsed % 60))
        self.time_consume = '{:1d}h{:2d}m{:2d}s'.format(h, m, s)


class MultiModalRecorder(Recorder):
    def __init__(self, modes, save_keys):
        super().__init__(save_keys)
        self.modes = modes

    def _type_check(self, key, result):
        if key not in self.save_keys:
            print('Warning, the given key %s not included by this recorder! '
                  'The receive filed are %s.' % (key, str(self.save_keys)))
            return False

        if not isinstance(result, dict):
            print('The positional parameter result expected to be instance of dict, '
                  'but got %s!' % type(result))
            return False

        if not eq(list(result.keys()), self.modes):
            print('The input result should have the same mode as this recorder, expected %s, '
                  'but got %s!' % (str(self.modes), str(result.keys())))
            return False

        for m in self.modes:
            if not isinstance(result[m], Analyzer):
                print('The positional parameter result expected to be instance of Result, '
                      'but got %s!' % type(result[m]))
                return False

        return True

    def output(self):
        if self.save_dict is None:
            raise ValueError('The save_dict has not been updated!')

        num_metrics = len(self.save_keys)
        num_modes = len(self.modes)
        total = num_metrics * num_modes

        out = {'Time Consume': [self.time_consume] + [''] * (total - 1)}
        mode_content = [''] * total
        for i in range(total):
            if i % num_metrics == 0:
                mode_content[i] = self.modes[i // num_metrics]

        out['Modes'] = mode_content
        out['Select Metric'] = [str(item) for item in self.save_keys] + [''] * (total - num_metrics)

        numerical_result = {}
        for m in self.modes:
            for key in self.save_keys:
                result = self.save_dict[key][m]
                content = result.output_data()
                for item in content:
                    k, v = item.split(':')
                    if k in numerical_result.keys():
                        numerical_result[k].append(v)
                    else:
                        numerical_result[k] = [v]

        out.update(numerical_result)

        return out, total
