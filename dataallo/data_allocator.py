import copy
import random
from typing import List

import math

from auxiliary.helper import *

any_type = 0  # 任何类型
integer = 1  # 整数型
real = 2  # 实数型
categorical = 3  # 类别型

__all__ = ['DataHandler', 'any_type', 'integer', 'real', 'categorical']


class DataHandler(object):
    """
    DataHandler类，负责数据的读取与分割
    """

    def __init__(self, path: str = "", split_token: str = ",", dtype: int = any_type,
                 del_col: List[int] = None) -> None:
        """
        初始化DataHandler实例
        :param path: 数据集所在路径
        :param split_token: 属性间分隔符
        :param dtype: 属性的数据类型，用于属性的预处理
        """
        self.data_path = path
        self.data_set = []
        self.split_token = split_token
        self.dtype = dtype
        self.del_col = del_col
        self.__load()

    def refresh_data_set(self, path: str = "", split_token: str = ",", dtype: int = any_type,
                         del_col: List[int] = None) -> None:
        """
        刷新数据，与实例化该类的逻辑类似，读入新的数据集
        :param path: 数据集所在路径
        :param split_token: 属性间分隔符
        :param dtype: 属性的数据类型，用于属性的预处理
        :param del_col: 选择删除的属性下标
        :return:
        """
        self.data_path = path
        self.data_set = []
        self.split_token = split_token
        self.dtype = dtype
        self.del_col = del_col
        self.__load()

    def __load(self) -> None:
        """
        读取数据, 不读取第一行
        """
        self.data_set = []
        with open(self.data_path, encoding="UTF-8") as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                attrs = line.split(self.split_token)
                if self.del_col:
                    v = len(attrs)
                    keep_col = list(filter(lambda i: i not in self.del_col, range(v)))
                    attrs = list(map(lambda j: attrs[j], keep_col))
                self.data_set.append(attrs)

        def cast_func(u):
            return u

        def map_categorical(data_set: list):
            """
            将类别属性映射为0,1序列
            :param data_set: 待使用的数据集
            :return: 映射后的数据集
            """
            m = len(data_set[0]) - 1
            attr_values = []
            res_set = []
            for k in range(m):
                attr_values.append(sorted(list(set(list(map(lambda x: x[k], data_set))))))
            for data in data_set:
                mapping = []
                for k in range(m):
                    values = attr_values[k]
                    mapping.extend(
                        list(map(lambda j: 1 if data[k] == values[j] else 0, range(len(values)))))
                mapping.append(data[-1])
                res_set.append(mapping)
            return res_set

        if self.dtype == integer:
            cast_func = int
        elif self.dtype == real:
            cast_func = float

        if self.dtype == integer or self.dtype == real:
            for i in range(len(self.data_set)):
                self.data_set[i] = list(map(cast_func, self.data_set[i]))

        if self.dtype == categorical:
            self.data_set = map_categorical(self.data_set)

    def get_data_set(self):
        """
        直接返回读取到的数据集
        :return: 实例已读取的数据集
        """
        return self.data_set

    def split_tt_sets(self, train_ratio: float) -> (list, list):
        """
        分配训练集与测试集
        :param train_ratio: 训练集占总数据集的比例
        :return: 二元tuple，第一项为训练集，第二项为测试集
        """
        assert 0 < train_ratio <= 1
        assert self.data_set
        length = len(self.data_set)
        tmp = math.floor(length * train_ratio)
        train_num = tmp if tmp > 0 else math.ceil(length * train_ratio)
        d = copy.deepcopy(self.data_set)
        for i in range(random.randint(1, 100)):
            random.shuffle(d)
        train_set = d[:train_num]
        test_set = d[train_num:]
        return train_set, test_set

    def split_folds(self, c: int, k: int) -> list:
        """
        分割子集，用于交叉验证法
        :param c: 折数
        :param k: 使用的分割方法的数量，即产生多少个c折数据集
        :return: 包含k个c折数据的列表. 格式[[fold_1, fold_2,..., fold_c],..., [fold_1, fold_2,..., fold_c]]
        """
        assert c > 0 and k > 0
        d = copy.deepcopy(self.data_set)
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
                for i in range(random.randint(1, 100)):
                    random.shuffle(sub_set)

            folds = []
            for i in range(c):
                folds.append([])

            # 每一折平均分配所有种类的样例
            for i in range(c):
                for sub_set in d_classified:
                    s = i * (len(sub_set) // c)
                    e = min((i + 1) * (len(sub_set) // c), len(sub_set))
                    folds[i].extend(sub_set[s:e])
            folds_list.append(folds)

        return folds_list

    def bootstrap(self, k: int = 1000):
        """
        自助法，随机挑选原数据集中的数据作为新数据集返回
        :param k: 随机挑选次数
        :return: 新数据集
        """
        n = len(self.data_set)
        res = list(map(lambda i: random.randint(0, n - 1), range(k)))
        return list(map(lambda i: self.data_set[i], set(res)))  # 去重返回
