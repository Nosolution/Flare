import numpy as np
import random
import operator
import copy

'''
西瓜书3.3题编程实现
'''

path_prefix = "F:/Subject/专业课/机器学习导论/data/"
data_path = "Breast Cancer Coimbra/dataR2.csv"

data_set = []


def load_data_set():
    with open(path_prefix + data_path) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i == 0:
                continue
            line = lines[i]
            data_set.append(np.array(list(map(float, line.split(',')))))
    for data in data_set:
        if data[9] == 2:
            data[9] = 0
    sorted(data_set, key=lambda x: x[9], reverse=True)


def split_data_set(k: int = 10):
    # 10折分割，使用k种不同的分割方法
    # split_res = [folds1, folds2...]
    split_res = []
    length = len(data_set)
    nega_index = 0
    # 找到第一个反例的序号，便于类别平衡分类
    for i in range(length):
        if data_set[i][9] == 0:
            nega_index = i
            break

    pd = nega_index // 10
    nd = (length - nega_index) // 10

    posi_set = data_set[:nega_index]
    nega_set = data_set[nega_index:]

    for i in range(k):
        # 重复打乱顺序，直至与之前的结果无重复为止
        random.shuffle(posi_set)
        random.shuffle(nega_set)
        # 假设随机扰乱后不会出现与之前相同的结果，不进行重复性检测
        folds = []
        for j in range(10):
            e1 = min((j + 1) * pd, nega_index)
            e2 = min(nega_index + (j + 1) * nd, length)
            fold = [*(posi_set[j * pd:e1]), *(nega_set[nega_index + j * nd:e2])]
            random.shuffle(fold)
            folds.append(copy.copy(fold))
        split_res.append(folds)

    return split_res


def likelihood(w, ds):
    res = 0
    for data in ds:
        x = np.array([*(data[:9]), 1])
        y = data[9]
        wx = w.dot(x)
        res += -y * wx + np.log(1 + np.exp(wx))
    return res


def llh_derivative(w, ds):
    res = np.zeros(10)
    for data in ds:
        x = np.array([*(data[:9]), 1])
        y = data[9]
        res += x * (y - 1 + 1 / (1 + np.exp(w.dot(x))))
    return res


def train(folds: list):
    w_list = []
    for i in range(len(folds)):
        train_set = []
        # 去除测试集
        for j in range(len(folds)):
            if j != i:
                train_set.extend(folds[j])

        w = np.zeros(10)
        a = 1
        max_round = 1000
        max_detect = 100
        min_a = 0.0000001
        round_count = 0

        # 开始训练
        while round_count < max_round and a > min_a:
            round_count += 1
            origin_llh = likelihood(w, train_set)
            origin_w = w
            detect_count = 0
            delta = llh_derivative(origin_w, train_set)

            # 找出使llh减少的最大步长
            while detect_count < max_detect:
                detect_count += 1
                w = origin_w + a * delta
                if likelihood(w, train_set) < origin_llh:
                    break
                a /= 2

        print("w is: {}".format(w))
        test_set = folds[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data in test_set:
            x = np.array([*(data[:9]), 1])
            odds = np.exp(x.dot(w))
            pred = 1 if odds > 1 else 0
            if pred == data[9]:
                if pred == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    fn += 1
        print("TP is: {}, FP is: {}, TN is: {}, FN is: {}".format(tp, fp, tn, fn))

        w_list.append(w)
    # 返回的w所使用的测试集按顺序为data_set的除第一折，第二折，第三折...
    # r = np.zeros(9)
    # for w in w_list:
    #     r += w

    return 0


def test(w: list, ds):
    np.set_printoptions(precision=5, suppress=True)
    print("w is: {}".format(w))
    test_res = []
    for data in ds:
        x = np.array([*(data[:8]), 1])
        odds = np.exp(x.dot(w))
        test_res.append(odds)
    print("the test result of w is: {}".format(test_res))
    test_res = list(map(lambda x: 1 if x > 1 else 0, test_res))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(10):
        if test_res[i] == data_set[i][8]:
            if test_res[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if test_res[i] == 1:
                fp += 1
            else:
                fn += 1
    print("TP is: {}, FP is: {}, TN is: {}, FN is: {}".format(tp, fp, tn, fn))


if __name__ == "__main__":
    # load_data_set()
    # data_sets = split_data_set()
    # for ds in data_sets:
    #     w = train(ds)
    #     # test(w, ds)
    print(1)
