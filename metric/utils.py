import numpy as np

classify_sta = {'Average F-score': 'F-score', 'Average AUC': 'AUC', 'Average accuracy': 'Accuracy'}
classify_record = ('Sensitivity', 'Specificity', 'Negativity', 'Positivity', 'AUC', 'Accuracy', 'F-score', 'epoch',
                   'threshold')

seg_sta = {'Average JS': 'JS', 'Average Dice': 'DC', 'Average Score': 'score'}
seg_record = ('Sensitivity', 'Specificity', 'Negativity', 'Positivity', 'AUC', 'Accuracy', 'JS', 'DC', 'score', 'epoch')


def soft_max(scores: np.ndarray, axis=1):
    exp_value = np.exp(scores)
    probability = exp_value / np.sum(exp_value, axis=axis, keepdims=True)
    return probability
