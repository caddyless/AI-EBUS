import torch
import ci
import compare

import torch.functional as F
import pandas as pd
import numpy as np

from pathlib import Path
from metric import Analyzer
from datamanager import MMVideoDataset


info_keys = ['Identity', 'Hospital', 'Date', 'Name', 'Sex', 'Age', 'Position', 'Diagnosis', 'BM']
map_dict = {'Identity': 'id', 'Hospital': 'hospital', 'Date': 'date', 'Name': 'name',
            'Sex': 'sex', 'Age': 'age', 'Position': 'local', 'Diagnosis': 'diag', 'BM': 'BM'}
# mode = ['Fusion', 'B', 'F', 'E']


def obtain_dataset(dataset_path, database_path=''):
    dataset = MMVideoDataset(database_path, 'BEF')
    dataset.load(dataset_path)
    dataset.set_iter_data('BEF')
    print(dataset)
    print(len(dataset))
    return dataset


def read_prediction(file_path):
    df = pd.read_csv(file_path)
    y_true = np.array(df['y_true'].astype(int))
    y_score = np.array(df['y_score'].astype(float))
    identity = np.array(df['identity'].astype(int))
    result = Analyzer(y_true, y_score, identity)
    return result


def read_trace_file(file_path, mode, reduced=True):
    for m in mode:
        df = pd.read_csv(file_path)
        part_df = df[df['Identity'] == i]
        row_df = part_df[part_df['Mode'] == m]
        c1, c2 = row_df['Category-0'].item(), row_df['Category-1'].item()
        c1, c2 = float(c1), float(c2)
        score = F.softmax(torch.tensor([c1, c2]), dim=0)[1].item()
        if reduced:
            data.update({('%s-%d-score' % (m, j)): ('%.4f' % score)})
        else:
            data.update({('%s-%d-score' % (m, j)): ('%.4f' % score), ('%s-%d-C1' % (m, j)): ('%.4f' % c1),
                         ('%s-%d-C2' % (m, j)): ('%.4f' % c2)})
    new_df = new_df.append(data, ignore_index=True)
    return


def read_five_fold_csv(result_folder: str):
    result_folder = Path(result_folder)
    dfs = []
    for i in range(5):
        file_path = result_folder / ('%d.csv' % i)
        df = pd.read_csv(str(file_path))
        dfs.append(df)
    return dfs


def merge_dfs(dataset, dfs, mode='BEF', reduced=True):
    data_keys = []
    for m in mode:
        for i in range(5):
            if reduced:
                data_keys += ['%s-%d-score' % (m, i)]
            else:
                data_keys += ['%s-%d-score' % (m, i), '%s-%d-C1' % (m, i), '%s-%d-C2' % (m, i)]
    df_keys = info_keys + data_keys
    raw_data = dataset.raw_data

    new_df = pd.DataFrame(columns=tuple(df_keys))
    ids = set(dfs[0]['Identity'].tolist())
    for i in ids:
        info = raw_data[i].info
        data = {k: info[map_dict[k]] for k in info_keys}

        for m in mode:
            for j in range(5):
                df = dfs[j]
                part_df = df[df['Identity'] == i]
                row_df = part_df[part_df['Mode'] == m]
                c1, c2 = row_df['Category-0'].item(), row_df['Category-1'].item()
                c1, c2 = float(c1), float(c2)
                score = F.softmax(torch.tensor([c1, c2]), dim=0)[1].item()
                if reduced:
                    data.update({('%s-%d-score' % (m, j)): ('%.4f' % score)})
                else:
                    data.update({('%s-%d-score' % (m, j)): ('%.4f' % score), ('%s-%d-C1' % (m, j)): ('%.4f' % c1),
                                 ('%s-%d-C2' % (m, j)): ('%.4f' % c2)})
        new_df = new_df.append(data, ignore_index=True)
    return new_df


def ensemble_prediction(df, mode='BEF', way: str = 'average_predict'):
    for i, m in enumerate(mode):
        raw_score = df[['%s-%d-score' % (m, j) for j in range(5)]]
        raw_score = np.array(raw_score)
        raw_score = raw_score.astype(float)
        if way == 'average_predict':
            raw_predict = (raw_score > 0.5).astype(int)
            raw_predict = raw_predict.mean(axis=1)
            predict = (raw_predict > 0.5).astype(int)
            score = np.zeros(predict.shape[0], dtype=np.float)
            print(score.shape)
            for j in range(score.shape[0]):
                rs = raw_score[j]
                if predict[j] == 1:
                    score[j] = rs[rs >= 0.5].mean()
                else:
                    score[j] = rs[rs < 0.5].mean()

        else:
            score = raw_score.mean(axis=1)
            predict = (score > 0.5).astype(int)
        df.insert(9 + i * 2, '%s-predict' % m, predict)
        df.insert(10 + i * 2, '%s-score' % m, score)

    return df


def evaluate_acc(df, mode='Fusion-score', threshold=0.5):
    label = df['BM']
    label = np.array(label).astype(int)
    score = df[mode]
    score = np.array(score).astype(float)
    result = Analyzer(label, score, threshold=threshold)
    data = result.output_data()
    print(' | '.join(data))
    return


def generate_df(dataset, result_folder):
    dfs = read_five_fold_csv(result_folder)
    new_df = merge_dfs(dataset, dfs)
    ense_df = ensemble_prediction(new_df)
    return ense_df


def statistic_result(folder, mode='BEF'):
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

    df = pd.DataFrame(columns=tuple(['Mode'] + wait_statistic_keys))

    for i, m in enumerate(mode):
        row1 = {'Mode': m}
        row2 = {'Mode': ''}
        for j, k in enumerate(wait_statistic_keys):
            value, left, right = ci.normal(data[i, :, j])
            if k in ['AUC', 'F-score']:
                row1[k] = '{:.4f}'.format(value)
                row2[k] = '{:.4f}-{:.4f}'.format(left, right)
            else:
                row1[k] = '{:4.2f}'.format(value * 100)
                row2[k] = '{:4.2f}-{:4.2f}'.format(left * 100, right * 100)
        df = df.append([row1, row2], ignore_index=True)
    print(df)
    return df


def statistic_p(folder, mode='BEF'):
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

    for i, m1 in enumerate(mode):
        for j, m2 in enumerate(mode):
            if m1 == m2:
                p = 1
            else:
                p = compare.t2sample(data[i, :, 0], data[j, :, 0])
                print('%s-%s p=%f' % (m1, m2, p))
    return


def conduct_statistic():
    dfs = read_dfs(result_folder)
    new_df = merge_dfs(mmv_dataset, dfs)
    new_df = ensemble_prediction(new_df)
    new_df.to_csv('../record/20210309-2-total-prospective-result.csv')
    #     statistic_result('../Final-Model/classification-2020-12-28 17:11:29/')
    return
