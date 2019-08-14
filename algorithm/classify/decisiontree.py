from collections import Counter

import numpy as np

from applier.multiapplier import MultiClassApplier

"""
决策树实现模块
"""

inf_ent_gain = 21
info_gain_ratio = 22
gini_inx = 23


def __inf_entropy(data_set: np.ndarray) -> float:
    """
    根据数据集的标记分布情况计算信息熵. 算式: Ent(D) = -sum_{k=1}^abs(m)(p_k*log_2{p_k})
    :param data_set: 待使用数据集
    :return: 数据集的类别的信息熵
    """
    s = len(data_set)
    c = Counter(list(map(lambda x: x[-1], data_set))).values()
    c = list(map(lambda x: x / s, c))
    return -sum(list(map(lambda x: x * np.log2(x), c)))


def __info_ent_gain(data_set: np.ndarray, i: int) -> float:
    """
    计算通过属性a_i分割数据集获得的信息增益
    :param data_set: 待分割计算的数据集
    :param i: 属性a_i的下标
    :return: 通过分割数据集获得的信息增益
    """
    assert 0 <= i < len(data_set[0])
    m = len(data_set)
    before = __inf_entropy(data_set)
    values = Counter(list(map(lambda x: x[i], data_set))).keys()
    after = 0
    for value in values:
        sub_set = list(filter(lambda x: x[i] == value, data_set))
        after += len(sub_set) / m * __inf_entropy(np.array(sub_set))
    return before - after


def __gain_ratio(data_set: np.ndarray, i: int) -> float:
    """
    计算通过属性a_i分割数据集获得的信息增益率
    :param data_set: 待分割计算的数据集
    :param i: 属性a_i的下标
    :return: 通过分割数据集获得的信息增益率
    """
    gain = __info_ent_gain(data_set, i)
    value_counts = Counter(list(map(lambda x: x[i], data_set))).values()
    return gain / sum(map(lambda x: -x * np.log2(x), value_counts))


def __gini(data_set: np.ndarray) -> float:
    """
    对数据集进行基尼值计算
    :param data_set: 待计算的数据集
    :return: 该数据集的基尼值
    """
    m = len(data_set)
    d = sorted(data_set, key=lambda x: x[-1])
    i, j = 0, 0
    res = 1
    while j < m:
        if d[j][-1] != d[i][-1]:
            res -= ((j - i) / m) ** 2
            i = j
        j += 1
    return res


def __gini_index(data_set: np.ndarray, i: int) -> float:
    """
    对数据集在属性a_i上进行基尼指数计算
    :param data_set: 待计算的数据集
    :param i: 属性a_i的下标
    :return: 该数据集的基尼指数
    """
    m = len(data_set)
    res = 0
    values = Counter(list(map(lambda x: x[i], data_set))).keys()
    for value in values:
        sub_set = list(filter(lambda x: x[i] == value, data_set))
        res += len(sub_set) / m * __gini(np.array(sub_set))
    return res


class IntTreeNode:
    """
    决策树节点
    """

    def __init__(self):
        self.label = -1  # 叶节点对应的标签
        self.ai = 0  # 该非叶节点决策对应的属性下标
        self.children = {}  # 该非叶节点的所有子节点，字典格式: {attr_value:child}

    def decide(self, x):
        if not self.children:
            return self.label
        else:
            return self.children[x[self.ai]].decide(x)


def generate_int_tree(train_set: list, attrs: dict, strat: int) -> MultiClassApplier:
    """
    决策树生成法的包装方法
    :param train_set:
    :param attrs:
    :param strat:
    :return:
    """
    return IntTreeApplier(__generate_int_tree(train_set=train_set, attrs=attrs, strat=strat))


def __generate_int_tree(train_set: list, attrs: dict, strat: int) -> IntTreeNode:
    """
    递归生成决策树，返回根节点
    :param train_set: 训练集, 要求属性值都为整数, 格式: [(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param attrs: 参与决策的所有属性, 格式: {attr1:[value_1,..., value_k], attr2:[value_1,..., value_k],...}
    :param strat: 采用的学习策略
    :return: 生成的决策树的根节点
    """
    assert train_set
    attr_keys = attrs.keys()
    node = IntTreeNode()

    # 如果训练集中的样例属于同一类，标记社为该类，返回根节点
    if len(Counter(list(map(lambda x: x[-1], train_set))).keys()) == 1:
        node.label = train_set[0][-1]
        return node
    else:
        same_flag = True
        # 判断是不是在参与决策的所有属性上，训练集的类别都一致
        for i in attr_keys:
            if not len(Counter(list(map(lambda x: x[i], train_set))).keys()) == 1:
                same_flag = False
                break
        # 或者是否属性已使用完
        if not attrs or same_flag:
            labels = sorted(Counter(list(map(lambda x: x[-1], train_set))).items(), key=lambda x: x[1], reverse=True)
            node.label = labels[0][0]
            return node

    # 偷懒写法
    def measure(x, i):
        return 0

    if strat == inf_ent_gain:
        measure = __info_ent_gain
    elif strat == info_gain_ratio:
        measure = __info_ent_gain
    elif strat == gini_inx:
        measure = __gini_index

    final_i = 0
    max_measure = 0
    for item in attrs.items():
        # 判断收益最大的属性
        if measure(train_set, item[0]) > max_measure:
            final_i = item[0]
    # 按该属性上的取值对数据集进行分类
    node.ai = final_i
    value_counts = {}
    for value in attrs[final_i]:
        value_counts[value] = list(filter(lambda x: x[final_i] == value, train_set))

    attrs.pop(final_i)  # 移除该属性
    # 对每一类进行判断
    for value in value_counts:
        child = IntTreeNode()
        # 如果在该取值上无样例
        if not value_counts[value]:
            # 赋予原数据集中比例最大的标签
            labels = sorted(Counter(list(map(lambda x: x[-1], train_set))).items(), key=lambda x: x[1], reverse=True)
            child.label = labels[0][0]
        else:
            sub_set = value_counts[value]
            child = __generate_int_tree(sub_set, attrs.copy(), strat)
        node.children[value] = child
    return node


class IntTreeApplier(MultiClassApplier):
    """
    决策树的模型应用器
    """

    def __init__(self, root: IntTreeNode, model: list = None):
        super().__init__(model)
        self.root = root

    def predict(self, x: list) -> int:
        return self.root.decide(x)
