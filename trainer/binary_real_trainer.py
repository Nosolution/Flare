from algorithm.classify import binary_real as br
from trainer.binaryapplier import BinaryApplier
from auxiliary.helper import *

"""
二分类实数分类器训练模块, 没必要写成类就直接写了个方法
"""


def train(train_set: list, algorithm: int, debug_mode: bool = False, **kwargs) -> BinaryApplier:
    """
    二分类实数分类器训练函数
    :param train_set: 训练集, 要求格式要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param algorithm: 指定的训练算法
    :param debug_mode: 产生debug用输出
    :return: 二分类实数分类器模型应用器
    """
    ts = list(map(lambda data: list(map(float, data)), train_set))
    if algorithm == linear_regression:
        return br.linear_regression(ts, debug_mode, **kwargs)
    elif algorithm == logistic_regression:
        return br.logistic_regression(ts, debug_mode, **kwargs)
    elif algorithm == lda:
        return br.lda(ts, debug_mode, **kwargs)
    return BinaryApplier([])
