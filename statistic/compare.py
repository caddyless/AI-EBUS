import numpy as np
from scipy import stats
from statistic.dl_test import delong_roc_test
from sklearn.metrics import confusion_matrix


def t2sample(group1: np.ndarray, group2: np.ndarray, equal_var=False):
    static, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
    return p


def dl_t(y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray):
    return delong_roc_test(y_true, y_score1, y_score2)


# def get_f_score_v(y_true, y_pred):
#     tp = (y_true == y_pred).sum()
#     fp = ((y_pred == 1) * (y_true == 0)).sum()
#     fn = ((y_pred == 0) * (y_true == 1)).sum()

#     total = int(tp + 0.5 * (fp + fn) + 0.5)

#     vector = np.zeros(total, dtype=np.int)


def observation_from_metric(y_true, y_pred_a, y_pred_b, m):
    if m == 'Sensitivity':
        v_a = y_pred_a[y_true == 1]
        v_b = y_pred_b[y_true == 1]
    
    elif m == 'Specificity':
        v_a = 1 - y_pred_a[y_true == 0]
        v_b = 1 - y_pred_b[y_true == 0]
    
    # elif m == 'Positivity':
    #     v_a = y_true[y_pred_a == 1]
    #     v_b = y_true[y_pred_b == 1]
    
    # elif m == 'Negativity':
    #     v_a = 1 - y_true[y_pred_a == 0]
    #     v_b = 1 - y_true[y_pred_b == 0]
    
    elif m == 'Accuracy':
        v_a = (y_pred_a == y_true) - 0
        v_b = (y_pred_b == y_true) - 0
        # print(m, v_a.mean(), v_b.mean())
    
    else:
        raise ValueError('Unsupported metric %s' % m)
    
    # print(v_a, v_b)
    # print(v_a.shape)
    cm = confusion_matrix(v_a, v_b)
    # print(cm[1, 1] == ((v_a == 1) * (v_b == 1)).sum())
    return cm    


def chi2_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b, alpha: float = 0.95):
    metrics = ('Sensitivity', 'Specificity', 'Accuracy')

    results = {}
    for m in metrics:
        cm = observation_from_metric(y_true, y_pred_a, y_pred_b, m)
        # print(m)
        # print(cm)
        b = cm[0, 1]
        c = cm[1, 0]

        if b + c >= 40:
            chi2 = (b - c) ** 2 / (b + c)
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, 1)
        # chi2, p, dof, expected = stats.chi2_contingency(cm, True)
        results[m] = p
    
    return results

