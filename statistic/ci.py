from scipy import stats
from statistic.dl_test import ci_auc
from sklearn.metrics import confusion_matrix
import numpy as np

from matplotlib import pyplot as  plt


def normal(group: np.ndarray, alpha: float = 0.95):
    m = group.mean()
    s = group.std()
    size = group.size
    if s == 0:
        return m, m, m
    if size > 30:
        left, right = stats.norm.interval(loc=m, scale=s / (size ** 0.5), alpha=alpha)
    else:
        left, right = stats.t.interval(df=(size - 1), loc=m, scale=s / (size ** 0.5), alpha=0.95)
    return m, left, right


def auc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.95):
    a, ci = ci_auc(y_true, y_score, alpha)
    return a, ci[0], ci[1]


def binomial(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.95):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]
    metrics = ('Sensitivity', 'Specificity', 'Positivity', 'Negativity', 'F-score', 'Accuracy')
    sen = (tp + fn, tp / (tp + fn))
    spe = (tn + fp, tn / (tn + fp))
    pos = (tp + fp, tp / (tp + fp))
    neg = (tn + fn, tn / (tn + fn))
    acc = (tn + tp + fp + fn, (tp + tn) / (tn + tp + fp + fn))
    f_score = (int(tp + 0.5 * (fp + fn)), tp / (tp + 0.5 * (fp + fn)))
    data = []
    for m, v in zip(metrics, [sen, spe, pos, neg, f_score, acc]):
        total = v[0]
        p = v[1]
        try:
            success = int(total * p)
        except ValueError as e:
            print(e)
            success = 0
        left = stats.beta.ppf((1 - alpha) / 2, success, total - success + 1)
        right = stats.beta.ppf((1 + alpha) / 2, success + 1, total - success)
        data.append((m, p, left, right))
    return data
