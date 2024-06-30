import os
import pandas as pd
import numpy as np
import math
from scipy.stats import norm, chi2


def read_csv(file):
    df = pd.read_csv(file)
    data = df.values
    return data


def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    acc = correct / y_true.shape[0]
    return acc


def alignment(csv_file='./single-modal/e/0.csv'):
    data = read_csv(csv_file)
    number = np.unique(data[:, :2], axis=0).astype(np.int)
    return number


def collect(folder, number, score=False):
    register = np.zeros((49, 7), np.int)
    if score:
        predicted = np.zeros((49, 27), np.float)
    else:
        predicted = np.zeros((49, 27), np.int)
    register[:, :2] = number
    predicted[:, :2] = number
    with open(os.path.join(folder, 'thresholds.txt'), 'r') as f:
        content = f.read()
        thresholds = [float(item) for item in content.split(',')]
        thresholds = np.array(thresholds)
    for i in range(5):
        csv_file = os.path.join(folder, '%d.csv' % i)
        data = read_csv(csv_file)
        p = np.zeros((49, 5), np.float)
        for j in range(245):
            clumn = j // 49
            row = predicted[:, 0] == data[j, 0]
            if score:
                p[row, clumn] = data[j, 2]
            else:
                if data[j, 2] < thresholds[i]:
                    p[row, clumn] = 0
                else:
                    p[row, clumn] = 1
        predicted[:, 2+i*5:2+(i+1)*5] = p
        register[:, i+2] = p.mean(axis=1) > 0.5
#     y_true = predicted[:, 1]
#     for i in range(5):
#         acc = 0
#         for j in range(5):
#             y_pred = predicted[:, 2+i*5+j]
#             acc += accuracy(y_true, y_pred)
#         print(acc / 5)
    return register, predicted


def extract_number(name):
    path, item = os.path.split(name)
    number = int(item.split('-')[0])
    return number


def experts(file, number, score=False):
    register = np.zeros((49, 8), np.int)
    if score:
        predicted = np.zeros((49, 32), np.float)
    else:
        predicted = np.zeros((49, 32), np.int)
    register[:, :2] = number
    predicted[:, :2] = number
    data = read_csv(file)
    prediction = data[:, 3:]
    name = data[:, 1].tolist()
    name = np.array(list(map(extract_number, name)))
    for index, n in enumerate(list(number[:, 0])):
        d = prediction[name == n, :]
        for i in range(6):
            for j in range(5):
                if score:
                    predicted[index, 2 + i * 5 + j] = (d[j, i*2 + 1] - 1) / 4
                else:
                    predicted[index, 2 + i * 5 + j] = (d[j, i*2] - 1)
    for i in range(6):
        p = predicted[:, 2+i*5:2+(i+1)*5]
        register[:, i+2] = p.mean(axis=1) > 0.5
    return register, predicted


def theta(Xi, Y, order=True):
    assert Xi.shape[0] == Y.shape[1], print('error!')
    J = Xi.shape[0]
    n = Y.shape[0]
    X = np.repeat(np.reshape(Xi, (1, J, 1)), J, axis=2)
    Y = np.reshape(Y, (Y.shape[0], 1, J))
    if order:
        residual = Y - X
    else:
        residual = X - Y
    result = np.zeros_like(residual)
    result[residual > 0] = 1
    result[residual == 0] = 0.5
    th = result.sum() / (n * J * J)
    return th


def theta_bar(X, Y):
    assert X.shape[1] == Y.shape[1], print('error!')
    m = X.shape[0]
    n = Y.shape[0]
    J = X.shape[1]
    x = np.repeat(np.reshape(X, (m, 1, J, 1)), J, 3)
    print(x.shape)
    y = np.reshape(Y, (1, n, J, 1))
    residual = y - x
    print(residual.shape)
    result = np.zeros_like(residual)
    result[residual > 0] = 1
    result[residual == 0] = 0.5
    return result.sum() / (m * n * J * J)


def read(matrix):
    x_row = matrix[:, 1] == 0
    y_row = matrix[:, 1] == 1
    X = np.reshape(matrix[x_row, 2:], (100, -1), 'F')
    Y = np.reshape(matrix[y_row, 2:], (145, -1), 'F')
    return X, Y


def sigma(X, Y, t_b, t1, t2):
    m = X[0].shape[0]
    n = Y[0].shape[0]
    sigma1 = 0
    sigma0 = 0
    for i in range(m):
        sigma1 += ((theta(X[t1][i, :], Y[t1]) - t_b[t1]) * (theta(X[t2][i, :], Y[t2]) - t_b[t2]))
    sigma1 = sigma1 / (m - 1)
    for k in range(n):
        sigma0 += ((theta(Y[t1][k, :], X[t1], False) - t_b[t1]) * (theta(Y[t2][k, :], X[t2], False) - t_b[t2]))
    sigma0 = sigma0 / (n - 1)
    return sigma1, sigma0


def covariance(machine, human):
    X = []
    Y = []
    x, y = read(machine)
    m = x.shape[0]
    n = y.shape[0]
    X.append(x)
    Y.append(y)
    x, y = read(human)
    X.append(x)
    Y.append(y)
    t_b = [theta_bar(X[i], Y[i]) for i in range(2)]
    matrix10 = np.zeros((2, 2), np.float)
    matrix01 = np.zeros((2, 2), np.float)
    for i in range(2):
        for j in range(2):
            s1, s0 = sigma(X, Y, t_b, i, j)
            matrix10[i, j] = s1
            matrix01[i, j] = s0
    return (matrix10 / m + matrix01 / n), t_b


number = alignment()


def compare(data='ebf', mode='experts'):
    if len(data) > 1:
        r, machine = collect('./multi-modal/%s' % data, number, True)
    else:
        print('单模态')
        r, machine = collect('./single-modal/%s' % data, number, True)
    r, human = experts('./6名医生EBUS图像测试/%s-summary.csv' % data, number, True)
    print(human[:10, :7])
    if mode == 'all':
        h = human
    elif mode == 'experts':
        h = np.zeros((49, 17), np.float)
        h[:, :2] = human[:, :2]
        for i, index in enumerate([0, 4, 5]):
            h[:, 2 + i * 5:2 + (i + 1) * 5] = human[:, 2 + index * 5:2 + (index + 1) * 5]
    elif mode == 'trainee':
        h = np.zeros((49, 17), np.float)
        h[:, :2] = human[:, :2]
        for i, index in enumerate([1, 2, 3]):
            h[:, 2 + i * 5:2 + (i + 1) * 5] = human[:, 2 + index * 5:2 + (index + 1) * 5]

    else:
        raise ValueError('mdoe error!')
    return machine, h


def p_value(b, matrix, tb):
    b = np.reshape(b, (1, 2))
    t = np.array(tb)
    t = np.reshape(t, (1, 2))
    s = np.dot(b, t.T) / math.sqrt(np.dot(np.dot(b, matrix), b.T))
    print(s)
    return 1 - chi2.cdf(s, 1)


def gw_test(y_true, y_score1, y_score2):
    matrix, tb = covariance(y_score1, y_score2)
    b = np.array([1, -1])
    p = p_value(b, matrix, tb)
    return p
