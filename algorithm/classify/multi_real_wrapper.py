import copy
from itertools import permutations
from typing import Callable, Any

from applier.binaryapplier import *
from applier.multiapplier import *
from auxiliary.helper import *

"""
多类别实数分类器，通过组合多个二分类实数分类器进行决策
"""


def ovo_train(train_set: list, algorithm: Callable[[list, bool, Any], BinaryApplier],
              debug_mode: bool = False, **kwargs) -> MultiClassApplier:
    """
    一对一训练，为每两个类别生成一个分类器，总共有n(n-1)个分类器
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param algorithm: 二分类实数分类器所使用算法
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """

    class_num = count_class_num(train_set)
    if debug_mode:
        print('OvO training:\n class_num is: {}'.format(class_num))
    indices = get_class_indices(train_set)
    perm = list(filter(lambda x: x[0] < x[1], permutations(range(class_num), 2)))
    classifiers = []  # 最终格式 : [[w_1, p_1, n_1], [w_2, p_2, n_2],...,[w_n, p_n, n_n]]
    for p in perm:
        # 最后两个参数为正类与反类代表的真正种类
        if debug_mode:
            print('current positive class is: {}, negative class is: {}'.format(train_set[indices[p[1]]][-1],
                                                                                train_set[indices[p[0]]][-1]))
        classifier = [0, train_set[indices[p[1]]][-1], train_set[indices[p[0]]][-1]]
        set0 = train_set[indices[p[0]]: indices[p[0] + 1]]
        set1 = train_set[indices[p[1]]: indices[p[1] + 1]]
        # 重设标志，便于二分类训练
        for data in set0:
            data[-1] = 0
        for data in set1:
            data[-1] = 1
        train_set = [*set0, *set1]
        random.shuffle(train_set)
        applier = algorithm(train_set, debug_mode=debug_mode, **kwargs)
        classifier[0] = applier
        classifiers.append(classifier)
    return OvOApplier(classifiers)


def ovr_train(train_set: list, algorithm: Callable[[list, bool, Any], BinaryApplier],
              debug_mode: bool = False, **kwargs) -> MultiClassApplier:
    """
    一对多训练，以一个类别作为正类，其他类别作为反类来训练，总共有n个分类器
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param algorithm: 二分类实数分类器所使用算法
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """
    class_num = count_class_num(train_set)
    if debug_mode:
        print('OvR training:\n class_num is: {}'.format(class_num))
    indices = get_class_indices(train_set)
    # 所有标志都设为0，待某个类设为正类后在设为1
    d = copy.deepcopy(train_set)
    for data in d:
        data[-1] = 0
    classifiers = []  # 返回格式 : [[w1, t1], [w2, t2],...,[wn, tn]]
    for i in range(1, class_num + 1):
        if debug_mode:
            print('current positive class is: {}'.format(train_set[indices[i - 1]][-1]))
        for j in range(indices[i - 1], indices[i]):
            d[j][-1] = 1
        applier = algorithm(d, debug_mode=debug_mode, **kwargs)
        for j in range(indices[i - 1], indices[i]):
            d[j][-1] = 0
        classifiers.append([applier, train_set[indices[i - 1]][-1]])
    return OvRApplier(classifiers)


def mvm_train(train_set: list, algorithm: Callable[[list, bool, Any], BinaryApplier],
              debug_mode: bool = False, **kwargs) -> MultiClassApplier:
    """
    多对多训练，以一些类别作为正类，另外一些类别作为反类来训练，总分类器个数依赖于类别数量，通过对比最终生成的二元码序列与代表各类的纠错码的海明距离来进行判断
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param algorithm: 二分类实数分类器所使用算法
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """
    # 假设类别数所在区间为[3,7]
    class_num = count_class_num(train_set)
    if debug_mode:
        print('MvM training:\n class_num is: {}'.format(class_num))
    indices = get_class_indices(train_set)
    if 3 <= class_num <= 7:
        codewords = []
        # 加入codeword
        for i in range(class_num):
            # 第0行为全1，特殊情况
            if i == 0:
                codewords.append(list(np.ones(2 ** (class_num - 1) - 1)))
            else:
                codeword = []
                flag = False
                # 交替插入全1序列和全0序列
                for j in range(2 ** i):
                    length = min(2 ** (class_num - i - 1), 2 ** i - len(codeword))
                    codeword.extend(
                        list(np.ones(length)) if flag else list(np.zeros(length)))
                    flag = not flag
                codewords.append(codeword)

        # 按列训练数据
        classifiers = []
        for f in range(2 ** (class_num - 1)):
            set0 = []
            set1 = []
            for i in range(class_num):
                if codewords[i][f] == 0:
                    set0.append(train_set[indices[i]: indices[i + 1]])
                else:
                    set1.append(train_set[indices[i]: indices[i + 1]])
            for data in set0:
                data[-1] = 0
            for data in set1:
                data[-1] = 1
            applier = algorithm([*set0, *set1], debug_mode=debug_mode, **kwargs)
            classifiers.append(applier)
        mca = MultiClassApplier(classifiers)
        for i in range(class_num):
            codewords[i] = tuple((codewords[i], train_set[indices[i]][-1]))
        if debug_mode:
            print('codewords and  theirs classes are: \n{}'.format(codewords))
        mca.codewords = codewords
        return mca
    # TODO 待实现class_num>7的情况
