import pandas as pd
import numpy as np
from pathlib import Path
from metric import Analyzer


mode = ['Fusion', 'B', 'F', 'E']


def read_prediction(file_path):
    df = pd.read_csv(file_path)
    y_true = np.array(df['y_true'].astype(int))
    y_score = np.array(df['y_score'].astype(float))
    identity = np.array(df['identity'].astype(int))
    result = Analyzer(y_true, y_score, identity)
    return result


def statistic_result(folder):
    folder = Path(folder)
    results = {}
    for i in range(5):
        for m in mode:
            csv_file = folder / ('%d' % i) / 'result' / ('%s-epoch-prediction.csv' % m)
            r = read_prediction(str(csv_file))
            if results.get(m, -1) == -1:
                results[m] = [r]
            else:
                results[m].append(r)

    wait_statistic_keys = ['AUC', 'F-score', 'Accuracy', 'Sensitivity', 'Specificity', 'Positivity', 'Negativity']

    data = np.zeros((4, 5, 7), dtype=np.float)

    for i, m in enumerate(mode):
        for j in range(5):
            for p, k in enumerate(wait_statistic_keys):
                data[i, j, p] = results[m][j].result[k]

    mean = data.mean(1)
    std = data.std(1)

    for i, m in enumerate(mode):
        content = []
        for j, k in enumerate(wait_statistic_keys):
            interval = std[i, j] * (1.96 / 5 ** 0.5)
            if k in ['AUC', 'F-score']:
                output = '{:s}: {:.4f}({:.4f}-{:.4f})'.format(k, mean[i, j], mean[i, j] - interval,
                                                              mean[i, j] + interval)
            else:
                output = '{:s}: {:4.2f}%({:4.2f}%-{:4.2f}%)'.format(k, mean[i, j] * 100, (mean[i, j] - interval) * 100,
                                                                    (mean[i, j] + interval) * 100)
            content.append(output)

        print('Mode %6s ' % m + ' | '.join(content))
    return
