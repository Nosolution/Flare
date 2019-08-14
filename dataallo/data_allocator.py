import copy
import random

import math

from auxiliary.helper import *

any_type = 0  # 任何类型
integer = 1  # 整数型
real = 2  # 实数型
categorical = 3  # 类别型


class DataHandler(object):
    """
    DataHandler类，负责数据的读取与分割
    """

    def __init__(self, path: str = "", split_token: str = ",", dtype: int = any_type) -> None:
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
        self.__load()

    def refresh_data_set(self, path: str = "", split_token: str = ",", dtype: int = any_type) -> None:
        """
        刷新数据，与实例化该类的逻辑类似，读入新的数据集
        :param path: 数据集所在路径
        :param split_token: 属性间分隔符
        :param dtype: 属性的数据类型，用于属性的预处理
        :return:
        """
        self.data_path = path
        self.data_set = []
        self.split_token = split_token
        self.dtype = dtype
        self.__load()

    def __load(self) -> None:
        """
        读取数据, 不读取第一行
        """
        self.data_set = []
        with open(self.data_path) as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                self.data_set.append(line.split(self.split_token))

        def cast_func(u):
            return u

        if self.dtype == integer:
            cast_func = int
        elif self.dtype == real:
            cast_func = float
        # TODO 属性数据类型为categorical时的处理

        if self.dtype == integer or self.dtype == real:
            for i in range(len(self.data_set)):
                self.data_set[i] = list(map(cast_func, self.data_set[i]))

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

    def bootstrap(self):
        # TODO 待实现自助法
        pass
