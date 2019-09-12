from collections import Counter
import math

"""
一些各模块需要使用到的函数
"""


def count_class_num(data_set: list) -> int:
    """
    计算数据集的总类别数目
    :param data_set: 数据集, 要求格式[[x_1, x_2, x_3,..., x_n, label],..., [x_1, x_2, x_3,..., x_n, label]]
    """
    return len(Counter(list(map(lambda x: x[-1], data_set))))


def get_class_indices(data_set: list) -> list:
    """
    对数据集进行排序，查找各类第一个数据的下标并返回包含其数值的列表
    :param data_set: 数据集, 要求格式[[x_1, x_2, x_3,..., x_n, label],..., [x_1, x_2, x_3,..., x_n, label]], 其中label为整数且大于0
    :return: 包含各类第一个数据的下标的列表
    """
    # sort(data_set, key=(lambda x: x[-1]))  # 此处会对原列表进行排序
    data_set.sort(key=lambda x: x[-1])
    class_num = count_class_num(data_set)
    indices = [0]  # 第一个下标必为0
    for i in range(1, class_num):
        for j in range(indices[i - 1] + 1, len(data_set)):
            if data_set[j][-1] != data_set[indices[i - 1]][-1]:
                indices.append(j)
                break
    indices.append(len(data_set))
    return indices


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
