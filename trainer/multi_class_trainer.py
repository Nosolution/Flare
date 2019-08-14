from algorithm.classify import binary_real as br, multi_real_wrapper as ss
from auxiliary.helper import *
from trainer.applier.binaryapplier import MultiClassApplier

"""
多分类实数分类器训练模块, 没必要写成类就直接写了个方法
"""


def train(train_set: list, algorithm: int, strategy: int = OvR, debug_mode: bool = False,
          **kwargs) -> MultiClassApplier:
    """
    二分类实数分类器训练函数
    :param train_set: 训练集, 要求格式要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param algorithm: 指定的训练算法
    :param strategy: 指定的多分类策略
    :param debug_mode: 产生debug用输出
    :return: 多分类实数分类器模型应用器
    """
    ts = list(map(lambda data: list(map(float, data)), train_set))
    if strategy == OvO:
        stra = ss.ovo_train
    elif strategy == OvR:
        stra = ss.ovr_train
    else:
        stra = ss.mvm_train

    if algorithm == linear_regression:
        algo = br.linear_regression
    elif algorithm == logistic_regression:
        algo = br.logistic_regression
    else:
        algo = br.lda

    return stra(ts, algo, debug_mode, **kwargs)
