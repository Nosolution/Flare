import copy
import math
import random
from auxiliary.helper import *


class DataHandler(object):
    def __init__(self, path: str = "", split_token: str = ","):
        self.data_path = path
        self.data_set = []
        self.split_token = split_token
        self.__load()

    def refresh_data_set(self, path: str = "", split_token: str = ","):
        self.data_path = path
        self.data_set = []
        self.split_token = split_token
        self.__load()

    def __load(self):
        self.data_set = []
        with open(self.data_path) as f:
            lines = f.readlines()
            for line in lines:
                self.data_set.append(line.split(self.split_token))

    def split_tt_sets(self, train_ratio: float) -> (list, list):
        # 分配训练集与测试集
        # rate为训练集所占比例
        assert 0 < train_ratio <= 1
        assert self.data_set
        length = len(self.data_set)
        tmp = math.floor(length * train_ratio)
        train_num = tmp if tmp > 0 else math.ceil(length * train_ratio)
        d = copy.copy(self.data_set)
        for i in range(random.randint(1, 100)):
            random.shuffle(d)
        train_set = d[:train_num]
        test_set = d[train_num:]
        return train_set, test_set

    def split_folds(self, c: int, k: int) -> list:
        # 分割子集，用于交叉验证法
        # c为折数，k为使用的分割方法的数量
        assert c > 0 and k > 0
        d = copy.copy(self.data_set)
        # 找出所有分类第一个样例的下标
        class_num = count_class_num(d)
        indices = get_class_indices(d)

        # 按类分割数据集
        d_classified = []
        for i in range(class_num):
            d_classified.append(d[indices[i]: indices[i + 1]])

        folds_list = []
        for kind in range(k):
            # 打乱排序
            for sub_set in d_classified:
                for i in range(random.randint[1, 100]):
                    random.shuffle(sub_set)

            folds = []
            for i in range(c):
                folds.append([])

            # 每一折平均分配所有种类的样例
            for i in range(c):
                for sub_set in d_classified:
                    s = i * (len(sub_set) // c)
                    e = min((i + 1) * (len(sub_set) // c), len(sub_set))
                    folds[i].append(sub_set[s:e])
            folds_list.append(folds)

        return folds_list

    def bootstrap(self):
        pass
